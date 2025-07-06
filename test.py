import os
import cv2
from torch.utils import data
from PIL import Image
import pandas as pd
import torch
from torchvision import transforms

import numpy as np
import argparse
# import Fiqa_Model
from utils import performance_fit
import time


class ImageDataset(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, data_dir, filename_path, transform, resize, frames):
        super(ImageDataset, self).__init__()

        dataInfo = pd.read_csv(filename_path)
        self.Image_names = dataInfo['Image_name'].tolist()
        self.resize = resize
        self.videos_dir = data_dir
        self.transform = transform
        self.length = len(self.Image_names)
        self.frames = frames

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        Image_name = str(self.Image_names[idx]) + '.png'
        # Image_name_str = str(self.Image_names[idx])
        frames_path = os.path.join(self.videos_dir, Image_name)
        image = Image.open(frames_path)
        resize_transform = transforms.Resize((self.resize, self.resize))
        resized_image_tensor = resize_transform(image)
        resized_image_tensor = self.transform(resized_image_tensor)  # 应用相同的变换（如归一化）
        # if image.width > image.height:
        #     new_width = 600
        #     new_height = int(image.height * (new_width / image.width))
        # else:
        #     new_height = 600
        #     new_width = int(image.width * (new_height / image.height))

        # image_cropped = image.resize((new_width, new_height))

        # 转换为tensor
        # image_tensor = self.transform(image_cropped)
        # 转换为tensor
        image_tensor = self.transform(image)

        # 随机裁剪三个224×224的patch
        crop = transforms.RandomCrop((self.resize, self.resize))
        transformed_rs = torch.zeros([self.frames, 3, self.resize, self.resize])

        for i in range(self.frames):
            cropped_image = crop(image_tensor)
            transformed_rs[i] = cropped_image

        return transformed_rs, resized_image_tensor, Image_name

import random
class GDataset_original_rand(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, data_dir, filename_path, transform, resize, frames):
        super(GDataset_original_rand, self).__init__()

        dataInfo = pd.read_csv(filename_path)
        self.Image_names = dataInfo['Image_name'].tolist()
        self.resize = resize
        self.videos_dir = data_dir
        self.transform = transform
        self.length = len(self.Image_names)
        self.frames = frames
        self.dots = [(190, 431), (129, 207), (353, 185)]

    def __len__(self):
        return self.length

    def get_random_offsets(self, num_points=3, max_offset=50):
        """
        Generate random offsets for each point within a specified range.

        Args:
            num_points (int): Number of points to generate offsets for.
            max_offset (int): Maximum absolute value of the offset.

        Returns:
            list: A list containing tuples of (x_offset, y_offset) for each point.
        """
        return [(random.randint(-max_offset, max_offset), random.randint(-max_offset, max_offset)) for _ in
                range(num_points)]
    def extract_patches(self, image, top_left_coords, patch_size=(224, 224)):
        if isinstance(image, Image.Image):
            image = np.array(image)

        H, W, _ = image.shape
        ph, pw = patch_size

        patches = []
        for x, y in top_left_coords:
            x1 = max(0, min(x, W - pw))
            y1 = max(0, min(y, H - ph))
            x2 = x1 + pw
            y2 = y1 + ph

            patch = image[y1:y2, x1:x2]
            patch = Image.fromarray(patch).resize(patch_size, Image.Resampling.LANCZOS)
            patches.append(patch)

        # 确保有3个patches，不足则重复最后一个patch
        while len(patches) < 3:
            patches.append(patches[-1])

        tensor_patches = [transforms.ToTensor()(patch).float() for patch in patches]

        return torch.stack(tensor_patches, dim=0)

    def resize_image(self, image, target_width=600):
        ow, oh = image.size
        scale = target_width / float(ow)
        nh = int(oh * scale)
        resized_img = image.resize((target_width, nh), Image.Resampling.LANCZOS)
        return resized_img

    def __getitem__(self, idx):
        image_name = str(self.Image_names[idx]) + '.png'
        frames_path = os.path.join(self.videos_dir, image_name)

        image = Image.open(frames_path).convert('RGB')

        base_width, base_height = 600, 800
        ow, oh = image.size
        scale_w = ow / float(base_width)
        scale_h = oh / float(base_height)
        # dots = [(190, 431), (129, 207), (353, 185)]
        # dots = get_most_dense_regions()
        dots = self.dots
        adjusted_dots = [(int(x * scale_w), int(y * scale_h)) for x, y in dots]

        offsets = self.get_random_offsets(num_points=len(adjusted_dots), max_offset=50)
        adjusted_dots_with_offset = [(x + dx, y + dy) for (x, y), (dx, dy) in zip(adjusted_dots, offsets)]

        transformed_rs = self.extract_patches(image, adjusted_dots_with_offset)

        resize_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            self.transform
        ])
        resized_image_tensor = resize_transform(image)

        return transformed_rs, resized_image_tensor, image_name

import FIQAModel

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Fiqa_Model.mobileNetV3_conv()
    model = FIQAModel.MobileNetV3_Conv_Mamba()

    pretrained_weights_path = config.Model_weights_path
    if pretrained_weights_path:
        model.load_state_dict(torch.load(pretrained_weights_path, map_location=device, weights_only=True), strict=True)
        print(f"成功加载预训练权重: {pretrained_weights_path}")

    model.to(device)
    model.float()

    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    testset = GDataset_original_rand(config.videos_dir, config.datainfo, transformations, config.resize, config.frames)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=config.num_workers)

    with torch.no_grad():
        model.eval()
        session_start_test = time.time()
        y_output = np.zeros([len(testset)])
        names = []

        for i, (video, rs, name) in enumerate(test_loader):
            video = video.to(device)
            rs = rs.to(device)
            # names.append('test/' + name[0])
            names.append(name[0])
            # label[i] = score.item()
            outputs = model(video, rs)
            y_output[i] = outputs.item()

        session_end_test = time.time()
        # val_PLCC, val_SRCC, val_KRCC, val_RMSE = performance_fit(label, y_output)
        # print(
        #     'completed. The result : SRCC: {:.4f}, KRCC: {:.4f}, '
        #     'PLCC: {:.4f}, and RMSE: {:.4f}'.format(val_SRCC, val_KRCC, val_PLCC, val_RMSE))

        data = {
            'Image': names,
            'MOS': y_output
        }

        df = pd.DataFrame(data)
        df.to_csv('results.csv', index=False)
        print(f'CostTime: {session_end_test - session_start_test:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--videos_dir', type=str, default='val')
    parser.add_argument('--datainfo', type=str, default='data/val.csv')
    parser.add_argument('--frames', type=int, default=3)
    parser.add_argument('--Model_weights_path', type=str, default='MobileNetV3_Conv_Mamba.pth', help='模型权重路径')
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=8)
    config = parser.parse_args()
    main(config)
