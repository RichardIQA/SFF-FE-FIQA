import torch
import torch.nn as nn
import torchvision.models as models
from thop import profile
import numpy as np
import torch.nn.functional as F
from torch.nn import init
from model.FIQAMamba import FIQAMambaBlock, FIQAMambaModule


# 自定义的Transformer编码层（兼容标准实现）
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, src):
        # 自注意力
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 前馈
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


# 修改后的Rearrange封装类（支持动态尺寸）
class DynamicRearrange(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 自动推导空间尺寸
        b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        return x.view(b, c, h * w).permute(0, 2, 1)  # 转换为 (batch, seq_len, features)

class FusionBlock(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()
        # 不同空洞率的卷积分支
        self.dconv1 = nn.Conv2d(channels, out_channels, 3, padding=1, dilation=1)
        self.dconv2 = nn.Conv2d(channels, out_channels, 3, padding=2, dilation=2)
        self.dconv3 = nn.Conv2d(channels, out_channels, 3, padding=3, dilation=3)
        #
        # self.pooling = nn.AvgPool2d(2, stride=2)
        # self.conv_final = nn.Conv2d(channels * 3, channels * 2, 1)

    def forward(self, x1):
        # 处理第一个输入特征
        x1_1 = self.dconv1(x1)
        # print(f"x1_1 shape: {x1_1.shape}")
        x1_2 = self.dconv2(x1)
        # print(f"x1_2 shape: {x1_2.shape}")
        x1_3 = self.dconv3(x1)
        # print(f"x1_3 shape: {x1_3.shape}")
        x1 = x1_1 + x1_2 + x1_3  # 特征融合方式1：相加
        # print(f"x1 shape: {x1.shape}")
        # print(f"x1 shape: {x1.shape}")
        # x = torch.cat([x1_1, x1_2, x1_3], dim=1)  # 特征融合方式2：拼接


        return x1

class CombinedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, stride=1, num_heads=4):
        super(CombinedConvBlock, self).__init__()

        # 残差连接捷径
        self.shortcut = nn.Sequential()
        if  downsample is not None or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )

        # 深度可分离卷积（改进通道注意力）
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        # self.FIQAMambaBlock = FIQAMambaBlock(out_channels , stride=stride)
        self.FIQAMambaBlock = FIQAMambaModule(out_channels, stride=stride)
        self.se = nn.Sequential(  # Squeeze-and-Excitation通道注意力
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )

        # 大核卷积（加入空间注意力）
        # self.large_kernel = nn.Conv2d(in_channels, out_channels, 7, stride, 3, groups=in_channels)
        # self.DConv = FusionBlock(in_channels, out_channels)  # 不同空洞率的卷积分支
        self.sa = SpatialAttention()  # 空间注意力模块

        # 新增Transformer分支（轻量级设计）
        self.transformer = nn.Sequential(
            DynamicRearrange(),  # 代替原先的固定尺寸Rearrange
            TransformerEncoderLayer(
                d_model=out_channels,
                nhead=num_heads,
                dim_feedforward=out_channels * 2,
                dropout=0.,
                activation='gelu'),
        )

        # 统一标准化层
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.shortcut(x)
        # print(residual.shape)
        # print(residual.shape)
        # 深度路径：DepthWise + PointWise + SE
        depthwise = self.depthwise(x)
        pointwise = self.pointwise(depthwise)
        # print(pointwise.shape)
        pointwise1 = self.FIQAMambaBlock(pointwise)
        channel_att = self.se(pointwise1)
        depth_path = pointwise1 * channel_att
        # print(depth_path.shape)

        # 大核路径：大核卷积 + 空间注意力
        # large_kernel = self.large_kernel(x)
        # large_kernel = self.DConv(x)
        spatial_att = self.sa(pointwise)
        kernel_path = pointwise * spatial_att
        # print(kernel_path.shape)

        # Transformer路径
        trans_path = (self.transformer(kernel_path).permute(0, 2, 1)).reshape(kernel_path.shape[0], kernel_path.shape[1], kernel_path.shape[2], kernel_path.shape[3])
        # print(trans_path.shape)
        # 三重特征融合
        combined = depth_path + kernel_path + trans_path + residual
        return self.relu(self.bn(combined))



# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CombinedNet(nn.Module):
    def __init__(self, width_multiplier=0.25, layers=None):
        super(CombinedNet, self).__init__()
        # self.num_classes = num_classes
        if layers is None:
            layers = [1, 2, 2, 1]
        self.width_multiplier = width_multiplier
        self.layers = layers

        # 定义网络的各层，这里可以使用 CombinedConvBlock 构建多个层级
        # 第一层，大核卷积用于初始特征提取，类似 ConvNeXt
        self.conv1 = nn.Conv2d(3, int(64 * width_multiplier), kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(int(64 * width_multiplier))
        self.relu = nn.ReLU()

        # 构建结合 ResNet 残差思想和深度可分离卷积的中间层
        self.layer1 = self._make_layer(int(64 * width_multiplier), int(128 * width_multiplier), layers[0])
        self.layer2 = self._make_layer(int(128 * width_multiplier), int(256 * width_multiplier), layers[1])
        self.layer3 = self._make_layer(int(256 * width_multiplier), int(512 * width_multiplier), layers[2])
        self.layer4 = self._make_layer(int(512 * width_multiplier), int(1024 * width_multiplier), layers[3])

        # 全连接层用于分类
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(int(512 * width_multiplier), num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用 He 初始化方法对卷积层的权重进行初始化
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # 批量归一化层的权重初始化为 1，偏置初始化为 0
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        # 下采样操作，如果输入和输出通道数不同或者步长为 2，则需要进行下采样
        downsample = None
        if in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        layers.append(CombinedConvBlock(in_channels, out_channels, stride=2, downsample=downsample))

        for _ in range(num_blocks - 1):
            layers.append(CombinedConvBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        # x = self.fc(x)
        return x


class mobileNetV3(torch.nn.Module):
    def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(mobileNetV3, self).__init__()

        mobileNet = models.mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.DEFAULT')
        # self.mobileNet = torch.nn.Sequential(*list(mobileNet.children())[:-1])
        self.mobileNet = mobileNet.features
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.Linear = nn.Linear(in_features=120, out_features=576)
        self.base_quality_regression = base_quality_regression(in_channels=576, middle_channels=128, out_channels=1)
        self.device = device

    def forward(self, x, y):

        x_size = x.shape
        B, T, C, H, W = x_size[0], x_size[1], x_size[2], x_size[3], x_size[4]
        y = y.unsqueeze(1)
        x = torch.cat((x, y), dim=1)

        x = self.avg_pool(self.mobileNet(x.reshape(-1, C, H, W)))
        x = x.reshape(B, T+1, -1)
        x = torch.mean(x, dim=1)
        x = self.base_quality_regression(x)
        return x.squeeze(1)


class shufflenet_v2_x1_0(torch.nn.Module):
    def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(shufflenet_v2_x1_0, self).__init__()

        mobileNet = models.shufflenet_v2_x0_5(weights='ShuffleNet_V2_X0_5_Weights.DEFAULT')
        # self.mobileNet = torch.nn.Sequential(*list(mobileNet.children())[:-1])
        self.mobileNet = nn.Sequential(*list(mobileNet.children())[:-1])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.Linear = nn.Linear(in_features=120, out_features=576)
        self.base_quality_regression = base_quality_regression(in_channels=1024, middle_channels=128, out_channels=1)
        self.device = device

    def forward(self, x, y):

        x_size = x.shape
        B, T, C, H, W = x_size[0], x_size[1], x_size[2], x_size[3], x_size[4]
        y = y.unsqueeze(1)
        x = torch.cat((x, y), dim=1)

        x = self.avg_pool(self.mobileNet(x.reshape(-1, C, H, W)))
        # print(x.shape)
        x = x.reshape(B, T+1, -1)
        x = torch.mean(x, dim=1)
        x = self.base_quality_regression(x)
        return x.squeeze(1)


class ConvNeXt(torch.nn.Module):
    def __init__(self):
        super(ConvNeXt, self).__init__()
        convNeXt = models.convnext_tiny(weights='ConvNeXt_Tiny_Weights.DEFAULT')
        # self.convNeXt = torch.nn.Sequential(*list(convNeXt.children())[:-1])
        # self.convNeXt = torch.nn.Sequential(*list(convNeXt.children())[:-1])
        self.convNeXt = convNeXt.features

        # print(self.convNeXt)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.base_quality_regression = base_quality_regression(in_channels=768, middle_channels=128,
                                                               out_channels=1)

    def forward(self, x, y):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        B, T, C, H, W = x_size[0], x_size[1], x_size[2], x_size[3], x_size[4]
        y = y.unsqueeze(1)
        x = torch.cat((x, y), dim=1)
        x = self.avg_pool(self.convNeXt(x.reshape(-1, C, H, W)))
        # print(x.shape)

        x = torch.mean(x.reshape(B, T + 1, -1), dim=1)
        x = self.base_quality_regression(x)

        return x.squeeze(1)

def base_quality_regression(in_channels, middle_channels, out_channels):
    regression_block = nn.Sequential(
        nn.Linear(in_channels, middle_channels),
        nn.ReLU(),
        nn.Linear(middle_channels, out_channels),
        nn.Sigmoid(),

    )
    return regression_block


class FeatureExtractor:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.outputs = {}
        self.hooks = []

        # 注册前向传播钩子
        for layer_idx in target_layers:  # 包含最后的输出层
            layer = model[layer_idx]
            self.hooks.append(
                layer.register_forward_hook(self._save_output(layer_idx))
            )

    def _save_output(self, layer_id):
        def hook(module, input, output):
            self.outputs[layer_id] = output.detach()

        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()



class MobileNetV3_Conv_Mamba(torch.nn.Module):
    def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(MobileNetV3_Conv_Mamba, self).__init__()

        mobileNet = models.mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.DEFAULT')
        shufflenet_v2 = models.shufflenet_v2_x0_5(weights='ShuffleNet_V2_X0_5_Weights.DEFAULT')
        self.shufflenet = nn.Sequential(*list(shufflenet_v2.children())[:-2])
        # self.mobileNet = torch.nn.Sequential(*list(mobileNet.children())[:-1])
        # self.Linear = nn.Linear(in_features=1024, out_features=576)
        self.mobileNet = mobileNet.features
        self.extractor = FeatureExtractor(self.mobileNet, target_layers=[8, 10])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.Linear = nn.Linear(in_features=120, out_features=576)
        # self.combineNet1 = CombinedConvBlock(in_channels=24, out_channels=96)
        self.combineNet2 = CombinedConvBlock(in_channels=48, out_channels=96)
        self.combineNet3 = CombinedConvBlock(in_channels=96, out_channels=192)

        self.base_quality_regression = base_quality_regression(in_channels=1056, middle_channels=128, out_channels=1)
        self.device = device

    def forward(self, x, y):

        x_size = x.shape
        B, T, C, H, W = x_size[0], x_size[1], x_size[2], x_size[3], x_size[4]


        y = self.avg_pool(self.shufflenet(y)).reshape(B, -1)


        x = self.avg_pool(self.mobileNet(x.reshape(-1, C, H, W)))
        # layer3_feature = self.extractor.outputs[3]
        layer6_feature = self.extractor.outputs[8]
        layer9_feature = self.extractor.outputs[10]
        # layer3_feature = self.avg_pool(self.combineNet1(layer3_feature))

        layer6_feature = self.avg_pool(self.combineNet2(layer6_feature))
        layer9_feature = self.avg_pool(self.combineNet3(layer9_feature))

        x = torch.cat([layer6_feature, layer9_feature, x], dim=1)
        # x = torch.cat([ layer9_feature, x], dim=1)
        # x = torch.cat([layer3_feature, layer6_feature, layer9_feature, x], dim=1)
        # x= layer6_feature + layer9_feature + x
        x = x.reshape(B, T, -1)
        x = torch.cat([torch.mean(x, dim=1), y], dim=1)

        x = self.base_quality_regression(x)
        return x.squeeze(1)



class MobileNetV3_Conv_Mamba_Crop_Only(torch.nn.Module):
    def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(MobileNetV3_Conv_Mamba_Crop_Only, self).__init__()

        mobileNet = models.mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.DEFAULT')

        self.mobileNet = mobileNet.features
        self.extractor = FeatureExtractor(self.mobileNet, target_layers=[3, 8, 10])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.Linear = nn.Linear(in_features=120, out_features=576)
        # self.combineNet1 = CombinedConvBlock(in_channels=24, out_channels=96)
        self.combineNet2 = CombinedConvBlock(in_channels=48, out_channels=96)
        self.combineNet3 = CombinedConvBlock(in_channels=96, out_channels=192)

        self.base_quality_regression = base_quality_regression(in_channels=864, middle_channels=128, out_channels=1)
        self.device = device

    def forward(self, x):

        x_size = x.shape
        B, T, C, H, W = x_size[0], x_size[1], x_size[2], x_size[3], x_size[4]
        x = self.avg_pool(self.mobileNet(x.reshape(-1, C, H, W)))
        # layer3_feature = self.extractor.outputs[3]
        layer6_feature = self.extractor.outputs[8]
        layer9_feature = self.extractor.outputs[10]
        # layer3_feature = self.avg_pool(self.combineNet1(layer3_feature))
        layer6_feature = self.avg_pool(self.combineNet2(layer6_feature))
        layer9_feature = self.avg_pool(self.combineNet3(layer9_feature))

        x = torch.cat([layer6_feature, layer9_feature, x], dim=1)
        # x = torch.cat([ layer9_feature, x], dim=1)
        # x = torch.cat([layer3_feature, layer6_feature, layer9_feature, x], dim=1)
        # x= layer6_feature + layer9_feature + x
        x = x.reshape(B, T, -1)
        x =torch.mean(x, dim=1)

        x = self.base_quality_regression(x)
        return x.squeeze(1)

class MobileNetV3_Conv_Mamba_all(torch.nn.Module):
    def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(MobileNetV3_Conv_Mamba_all, self).__init__()

        mobileNet = models.mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.DEFAULT')
        shufflenet_v2 = models.shufflenet_v2_x0_5(weights='ShuffleNet_V2_X0_5_Weights.DEFAULT')
        self.shufflenet = nn.Sequential(*list(shufflenet_v2.children())[:-2])
        # self.mobileNet = torch.nn.Sequential(*list(mobileNet.children())[:-1])
        # self.Linear = nn.Linear(in_features=1024, out_features=576)
        self.mobileNet = mobileNet.features
        self.extractor = FeatureExtractor(self.mobileNet, target_layers=[8, 10])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.Linear = nn.Linear(in_features=120, out_features=576)
        # self.combineNet1 = CombinedConvBlock(in_channels=24, out_channels=48)
        self.combineNet2 = CombinedConvBlock(in_channels=48, out_channels=96)
        self.combineNet3 = CombinedConvBlock(in_channels=96, out_channels=192)

        self.base_quality_regression = base_quality_regression(in_channels=1056, middle_channels=128, out_channels=1)
        self.device = device

    def forward(self, x):

        y = self.avg_pool(self.shufflenet(x[:, -1, :, :, :])).reshape(x.shape[0], -1)
        x = x[:, :-1, :, :, :]
        x_size = x.shape
        B, T, C, H, W = x_size[0], x_size[1], x_size[2], x_size[3], x_size[4]

        x = self.avg_pool(self.mobileNet(x.reshape(-1, C, H, W)))
        # layer3_feature = self.extractor.outputs[3]
        layer6_feature = self.extractor.outputs[8]
        layer9_feature = self.extractor.outputs[10]
        # layer3_feature = self.avg_pool(self.combineNet1(layer3_feature))
        # z = self.combineNet2(layer6_feature)

        layer6_feature = self.avg_pool(self.combineNet2(layer6_feature))
        layer9_feature = self.avg_pool(self.combineNet3(layer9_feature))
        x = torch.cat([layer6_feature, layer9_feature, x], dim=1)
        # x = torch.cat([ layer9_feature, x], dim=1)
        # x = torch.cat([layer3_feature, layer6_feature, layer9_feature, x], dim=1)
        # x= layer6_feature + layer9_feature + x
        x = x.reshape(B, T, -1)
        x = torch.cat([torch.mean(x, dim=1), y], dim=1)

        x = self.base_quality_regression(x)
        return x.squeeze(1)

class MobileNetV3_Conv_Mamba_all_V2(torch.nn.Module):
    def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(MobileNetV3_Conv_Mamba_all_V2, self).__init__()

        mobileNet = models.mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.DEFAULT')
        shufflenet_v2 = models.shufflenet_v2_x0_5(weights='ShuffleNet_V2_X0_5_Weights.DEFAULT')
        self.shufflenet = nn.Sequential(*list(shufflenet_v2.children())[:-2])
        # self.mobileNet = torch.nn.Sequential(*list(mobileNet.children())[:-1])
        # self.Linear = nn.Linear(in_features=1024, out_features=576)
        self.mobileNet = mobileNet.features
        self.extractor = FeatureExtractor(self.mobileNet, target_layers=[7, 9])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.Linear = nn.Linear(in_features=120, out_features=576)
        # self.combineNet1 = CombinedConvBlock(in_channels=24, out_channels=48)
        self.combineNet2 = CombinedConvBlock(in_channels=48, out_channels=96)
        self.combineNet3 = CombinedConvBlock(in_channels=96, out_channels=192)

        self.base_quality_regression = base_quality_regression(in_channels=1056, middle_channels=128, out_channels=1)
        self.device = device

    def forward(self, x):

        y = self.avg_pool(self.shufflenet(x[:, -1, :, :, :])).reshape(x.shape[0], -1)
        x = x[:, :-1, :, :, :]
        x_size = x.shape
        B, T, C, H, W = x_size[0], x_size[1], x_size[2], x_size[3], x_size[4]

        x = self.avg_pool(self.mobileNet(x.reshape(-1, C, H, W)))
        # layer3_feature = self.extractor.outputs[3]
        layer6_feature = self.extractor.outputs[7]
        layer9_feature = self.extractor.outputs[9]
        # layer3_feature = self.avg_pool(self.combineNet1(layer3_feature))
        layer6_feature = self.combineNet2(layer6_feature)
        Muti_scale_feature = self.pooling(layer6_feature)
        layer6_feature  = self.avg_pool(layer6_feature)
        layer9_feature = self.avg_pool(self.combineNet3(layer9_feature+Muti_scale_feature))

        x = torch.cat([layer6_feature, layer9_feature, x], dim=1)
        # x = torch.cat([ layer9_feature, x], dim=1)
        # x = torch.cat([layer3_feature, layer6_feature, layer9_feature, x], dim=1)
        # x= layer6_feature + layer9_feature + x
        x = x.reshape(B, T, -1)
        x = torch.cat([torch.mean(x, dim=1), y], dim=1)

        x = self.base_quality_regression(x)
        return x.squeeze(1)


class MobileNetV3_Conv_Mamba_all_V3(torch.nn.Module):
    def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(MobileNetV3_Conv_Mamba_all_V3, self).__init__()

        mobileNet = models.mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.DEFAULT')
        shufflenet_v2 = models.shufflenet_v2_x0_5(weights='ShuffleNet_V2_X0_5_Weights.DEFAULT')
        self.shufflenet = nn.Sequential(*list(shufflenet_v2.children())[:-2])
        # self.mobileNet = torch.nn.Sequential(*list(mobileNet.children())[:-1])
        # self.Linear = nn.Linear(in_features=1024, out_features=576)
        self.mobileNet = mobileNet.features
        self.extractor = FeatureExtractor(self.mobileNet, target_layers=[9])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.Linear = nn.Linear(in_features=120, out_features=576)
        # self.combineNet1 = CombinedConvBlock(in_channels=24, out_channels=48)
        # self.combineNet2 = CombinedConvBlock(in_channels=48, out_channels=96)
        self.combineNet3 = CombinedConvBlock(in_channels=96, out_channels=256)

        self.base_quality_regression = base_quality_regression(in_channels=1024, middle_channels=128, out_channels=1)
        self.device = device

    def forward(self, x):

        y = self.avg_pool(self.shufflenet(x[:, -1, :, :, :])).reshape(x.shape[0], -1)
        x = x[:, :-1, :, :, :]
        x_size = x.shape
        B, T, C, H, W = x_size[0], x_size[1], x_size[2], x_size[3], x_size[4]

        x = self.avg_pool(self.mobileNet(x.reshape(-1, C, H, W)))
        # layer3_feature = self.extractor.outputs[3]
        # layer6_feature = self.extractor.outputs[8]
        layer9_feature = self.extractor.outputs[9]
        # layer3_feature = self.avg_pool(self.combineNet1(layer3_feature))
        # layer6_feature = self.combineNet2(layer6_feature)
        # Muti_scale_feature = self.pooling(layer6_feature)
        # layer6_feature  = self.avg_pool(layer6_feature)
        layer9_feature = self.avg_pool(self.combineNet3(layer9_feature))

        # x = torch.cat([layer6_feature, layer9_feature, x], dim=1)
        x = torch.cat([ layer9_feature, x], dim=1)
        # x = torch.cat([layer3_feature, layer6_feature, layer9_feature, x], dim=1)
        # x= layer6_feature + layer9_feature + x
        x = x.reshape(B, T, -1)
        x = torch.cat([torch.mean(x, dim=1), y], dim=1)

        x = self.base_quality_regression(x)
        return x.squeeze(1)

if __name__ == '__main__':
    # Use a pipeline as a high-level helper

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV3_Conv_Mamba_all_V2().to(device)
    model = model.float()
    # input_sample = (torch.randn(1, 3, 3, 224, 224).to(device),  # x
    #                 torch.randn(1, 3, 224, 224).to(device))  # z
    input_sample = (torch.randn(1, 5, 3, 224, 224).to(device), )  # z

    flops, params = profile(model, inputs=input_sample)
    # flops= count_gflops_with_fvcore(model, input_sample)
    print(f"Total MFLOPs: {flops / 1e6}")
    print(f"Total params: {params / 1e6}")
