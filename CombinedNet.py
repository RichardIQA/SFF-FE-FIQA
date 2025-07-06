import numpy as np
import torch
import torch.nn as nn
from thop import profile
from torch.nn import init
import copy
from utils import plcc_loss, plcc_rank_loss, plcc_l1_loss
from torchvision import transforms
from utils import performance_fit, L1RankLoss
from dataset import ImageDataset, ImageDataset_VQC

class CombinedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(CombinedConvBlock, self).__init__()
        # 残差连接的捷径分支
        self.shortcut = nn.Sequential()
        if downsample is not None:
            self.shortcut = downsample

        # 深度可分离卷积（类似 MobileNet）
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # 大核卷积部分（类似 ConvNeXt 的部分设计理念，用于增强感知能力）
        self.large_kernel = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3,
                                      groups=in_channels)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.shortcut(x)

        # 深度可分离卷积路径
        depthwise_out = self.depthwise(x)
        pointwise_out = self.pointwise(depthwise_out)

        # 大核卷积路径
        large_kernel_out = self.large_kernel(x)

        # 将两个路径的输出相加，并通过批量归一化和激活函数
        out = self.relu(self.bn(pointwise_out + large_kernel_out + residual))
        # print(out.shape)
        return out


class CombinedNet(nn.Module):
    def __init__(self, width_multiplier=1.0, layers=None):
        super(CombinedNet, self).__init__()
        # self.num_classes = num_classes
        if layers is None:
            layers = [2, 2, 2, 2]
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


def base_quality_regression(in_channels, middle_channels, out_channels):
    regression_block = nn.Sequential(
        nn.Linear(in_channels, middle_channels),
        nn.ReLU(),
        nn.Linear(middle_channels, out_channels),
        nn.Sigmoid(),

    )
    return regression_block


class CombinedNetV3(torch.nn.Module):
    def __init__(self, width_multiplier=1.0, layers=None, in_channels=1024):
        super(CombinedNetV3, self).__init__()
        self.in_channels = in_channels
        if layers is None:
            layers = [2, 2, 2, 2]
        self.width_multiplier = width_multiplier
        self.layers = layers
        self.CombinedNet = CombinedNet(self.width_multiplier, self.layers)
        # print(self.CombinedNet)
        self.base_quality_regression = base_quality_regression(self.in_channels, middle_channels=128, out_channels=1)

    def forward(self, x, y):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        B, T, C, H, W = x_size[0], x_size[1], x_size[2], x_size[3], x_size[4]
        y = y.unsqueeze(1)
        x = torch.cat((x, y), dim=1)
        x = x.reshape(-1, C, H, W)
        x = self.CombinedNet(x)
        x = x.reshape(B, T + 1, -1)
        # x1 = torch.mean(x, dim=1)
        # print(x1.shape)
        x = torch.mean(x, dim=1)
        # print(x.shape)
        x = self.base_quality_regression(x)
        return x.squeeze(1)


# 定义NetAdapt算法类
class NetAdapt:
    def __init__(self, model, resource_limit, short_term_epochs=5, long_term_epochs=50):
        self.model = model
        self.resource_limit = resource_limit  # 资源限制，如最大延迟或计算量
        self.short_term_epochs = short_term_epochs  # 短期微调的轮数
        self.long_term_epochs = long_term_epochs  # 长期微调的轮数

    # 剪裁filter函数
    def prune_filters(self, layer, num_filters_to_prune):
        # 获取当前层的权重
        weight = layer.weight.data
        # 计算每个filter的L2范数
        filter_norms = torch.norm(weight.view(weight.shape[0], -1), dim=1)
        # 获取要剪裁的filter的索引
        prune_indices = torch.argsort(filter_norms)[:num_filters_to_prune]
        # 保留未被剪裁的filter
        new_weight = weight[prune_indices]
        # 创建新的卷积层
        new_layer = nn.Conv2d(new_weight.shape[1], new_weight.shape[0], kernel_size=layer.kernel_size,
                              stride=layer.stride, padding=layer.padding, groups=layer.groups,
                              bias=layer.bias is not None)
        # 将剪裁后的权重复制到新层
        new_layer.weight.data = new_weight
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data[prune_indices]
        return new_layer

    # NetAdapt算法主函数
    def optimize(self, dataloader, criterion, optimizer, device):
        best_model = copy.deepcopy(self.model)
        best_accuracy = 0.0

        for iteration in range(10):  # 迭代次数可根据需要调整
            candidate_models = []
            for i, layer in enumerate(self.model.modules()):
                if isinstance(layer, nn.Conv2d):
                    # 确定要剪裁的filter数量（这里简单起见，每次剪裁1个，实际应基于资源限制等因素计算）
                    num_filters = layer.out_channels
                    if num_filters > 1:  # 确保至少保留一个filter
                        num_filters_to_prune = 1
                    else:
                        continue

                    # 剪裁filter并创建候选模型
                    candidate_model = copy.deepcopy(self.model)
                    pruned_layer = self.prune_filters(candidate_model.layers[i], num_filters_to_prune)
                    candidate_model.layers[i] = pruned_layer

                    # 对候选模型进行短期微调
                    self.fine_tune(candidate_model, dataloader, criterion, optimizer, self.short_term_epochs, device)

                    # 计算候选模型的精度和资源消耗
                    accuracy, _ = self.evaluate(candidate_model, dataloader, device)
                    resource_consumption = self.calculate_resource_consumption(candidate_model)  # 需实现资源消耗计算

                    # 保存候选模型及其信息
                    candidate_models.append({
                        'model': candidate_model,
                        'accuracy': accuracy,
                        'resource_consumption': resource_consumption
                    })

            # 选择精度最高且满足资源限制的候选模型
            valid_candidates = [c for c in candidate_models if c['resource_consumption'] <= self.resource_limit]
            if valid_candidates:
                best_candidate = max(valid_candidates, key=lambda x: x['accuracy'])
                if best_candidate['accuracy'] > best_accuracy:
                    best_model = best_candidate['model']
                    best_accuracy = best_candidate['accuracy']

        # 对最终模型进行长期微调
        self.fine_tune(best_model, dataloader, criterion, optimizer, self.long_term_epochs, device)

        return best_model

    # 微调函数
    def fine_tune(self, model, dataloader, criterion, optimizer, epochs, device):
        model.train()
        for epoch in range(epochs):
            for inputs, rs, labels, _ in dataloader:
                inputs, labels, rs = inputs.to(device), labels.to(device), rs.to(device)
                optimizer.zero_grad()
                outputs = model(inputs, rs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    # 评估函数
    def evaluate(self, model, dataloader, device):
        model.eval()
        label = np.zeros([len(valset)])
        y_output = np.zeros([len(valset)])
        with torch.no_grad():
            for i, (inputs, rs, labels, _)in dataloader:
                inputs, labels, rs = inputs.to(device), labels.to(device), rs.to(device)
                # outputs = model(inputs, rs)
                label[i] = labels.item()
                outputs = model(inputs, rs)
                y_output[i] = outputs.item()
                val_PLCC, val_SRCC, val_KRCC, val_RMSE = performance_fit(label, y_output)

                print(
                    'The result on the validation databaset: SRCC: {:.4f}, KRCC: {:.4f}, '
                    'PLCC: {:.4f}, and RMSE: {:.4f}'.format(val_SRCC, val_KRCC, val_PLCC, val_RMSE))
        return val_PLCC, val_SRCC

    # 计算资源消耗函数（需根据具体资源限制实现）
    def calculate_resource_consumption(self, model):
        # 这里简单示例，实际计算应根据具体资源限制（如FLOPs、延迟等）进行
        flops = 0
        input_tensor = torch.randn(1, 3, 224, 224)
        flops, params = profile(model, inputs=(input_tensor,))
        return flops


# 使用示例
if __name__ == '__main__':
    # 创建一个简单的卷积神经网络
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU()
                # nn.Flatten(),
                # nn.Linear(256 * 28 * 28, 1)  # 假设输入图像大小为28x28
            )

        def forward(self, x, y):
            return torch.mean(self.layers(x.reshape(-1, 3, 224, 224)).reshape(y.shape[0], -1), dim=1)


    # 初始化模型、优化器、损失函数等
    model = SimpleCNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 创建NetAdapt优化器实例
    resource_limit = 500 * 1e6
    netadapt = NetAdapt(model, resource_limit)
    transformations_train = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    resize = 224
    frames = 3
    train_batch_size = 16
    num_workers = 4
    trainset = ImageDataset_VQC('FIQA/train', 'data/train.csv', transformations_train, resize, frames, 'train')
    valset = ImageDataset_VQC('FIQA/train', 'data/train.csv', transformations_train, resize, frames, 'val')
    # testset = VideoDataset(config.test_dir, config.test_datainfo, transformations_train, config.resize, config.frames)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=num_workers)

    # 执行NetAdapt优化
    optimized_model = netadapt.optimize(train_loader, criterion, optimizer, device)

    print("优化完成，最终模型的卷积层通道数：")
    for layer in optimized_model.modules():
        if isinstance(layer, nn.Conv2d):
            print(layer)




# # 测试代码
# model = CombinedNet(width_multiplier=0.25)
# print(model)
# input_tensor = torch.randn(1, 3, 224, 224)
# flops, params = profile(model, inputs=(input_tensor,))  # flops= count_gflops_with_fvcore(model, input_sample)
# print(f"Total MFLOPs: {flops / 1e6}")
# print(f"Total params: {params / 1e6}")
# output = model(input_tensor)
# # print(output.shape)
