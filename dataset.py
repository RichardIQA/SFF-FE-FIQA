import os
import random
import pandas as pd
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import crop as functional_crop

# def get_random_offsets(num_points=3, max_offset=50):
#     """
#     Generate random offsets for each point within a specified range.
#
#     Args:
#         num_points (int): Number of points to generate offsets for.
#         max_offset (int): Maximum absolute value of the offset.
#
#     Returns:
#         list: A list containing tuples of (x_offset, y_offset) for each point.
#     """
#     return [(random.randint(-max_offset, max_offset), random.randint(-max_offset, max_offset)) for _ in
#             range(num_points)]
def get_random_offsets(num_points=3, max_offset=50):
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


def extract_patches(image, dots, patch_size=(224, 224)):
    _, C, H, W = image.shape
    ph, pw = patch_size
    base_width, base_height = 600, 800
    scale_w = W / float(base_width)
    scale_h = H / float(base_height)

    adjusted_dots = [(int(x * scale_w), int(y * scale_h)) for x, y in dots]

    offsets = get_random_offsets(num_points=len(adjusted_dots), max_offset=50)
    adjusted_dots_with_offset = [(x + dx, y + dy) for (x, y), (dx, dy) in zip(adjusted_dots, offsets)]

    patches = []
    for x, y in adjusted_dots_with_offset:
        x1 = max(0, min(x, W - pw))
        y1 = max(0, min(y, H - ph))
        x2 = x1 + pw
        y2 = y1 + ph

        patch = image[..., y1:y2, x1:x2]
        patches.append(patch)

    # 确保有3个patches，不足则重复最后一个patch
    while len(patches) < 3:
        patches.append(patches[-1])

    tensor_patches = torch.cat([patch.unsqueeze(1) for patch in patches], dim=1)     # [b,3,c,h,w]

    return tensor_patches


class GDataset_original(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, data_dir, filename_path, transform, resize, frames, database_name):
        super(GDataset_original, self).__init__()

        dataInfo = pd.read_csv(filename_path)
        self.Image_names = dataInfo['Image_name'].tolist()
        self.score = dataInfo['MOS'].tolist()
        length = len(self.Image_names)
        seed = 20250611
        random.seed(seed)
        np.random.seed(seed)
        index_rd = np.random.permutation(length)
        train_source = index_rd[:int(length * 0.9)]
        val_source = index_rd[int(length * 0.9):]
        if database_name == 'train':
            self.Image_names = dataInfo.iloc[train_source]['Image_name'].tolist()
            self.score = dataInfo.iloc[train_source]['MOS'].tolist()
        elif database_name == 'val':
            self.Image_names = dataInfo.iloc[val_source]['Image_name'].tolist()
            self.score = dataInfo.iloc[val_source]['MOS'].tolist()
        self.resize = resize
        self.videos_dir = data_dir
        self.transform = transform
        self.length = len(self.Image_names)
        self.frames = frames
        self.dots = [(190, 431), (129, 207), (353, 185), (268, 310)]

    def __len__(self):
        return self.length

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
        Image_name = str(self.Image_names[idx]) + '.png'
        Image_score = torch.FloatTensor(np.array(float(self.score[idx])))
        frames_path = os.path.join(self.videos_dir, Image_name)

        image = Image.open(frames_path).convert('RGB')

        base_width, base_height = 600, 800
        ow, oh = image.size
        scale_w = ow / float(base_width)
        scale_h = oh / float(base_height)
        # dots = [(190, 431), (129, 207), (353, 185)]
        # dots = get_most_dense_regions()
        dots = self.dots
        adjusted_dots = [(int(x * scale_w), int(y * scale_h)) for x, y in dots]

        offsets = get_random_offsets(num_points=len(adjusted_dots), max_offset=50)
        adjusted_dots_with_offset = [(x + dx, y + dy) for (x, y), (dx, dy) in zip(adjusted_dots, offsets)]
        # print(adjusted_dots_with_offset)
        transformed_rs = torch.zeros([self.frames, 3, self.resize, self.resize])
        for i in range(self.frames):
            x, y = adjusted_dots_with_offset[i]
            cropped_image = functional_crop(self.transform(image), y, x, self.resize, self.resize)
            transformed_rs[i] = cropped_image

        # transformed_rs = self.extract_patches(self.transform(image), adjusted_dots_with_offset)
        # # print(transformed_rs.shape)

        resize_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            self.transform
        ])
        resized_image_tensor = resize_transform(image)

        return transformed_rs, resized_image_tensor, Image_score, Image_name


class ImageDataset(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, data_dir, filename_path, transform, resize, frames):
        super(ImageDataset, self).__init__()

        dataInfo = pd.read_csv(filename_path)
        self.Image_names = dataInfo['Image_name'].tolist()
        self.score = dataInfo['MOS'].tolist()
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
        Image_score = torch.FloatTensor(np.array(float(self.score[idx])))
        # print(f"正在读取视频 {Image_name_str}，分数 {Image_score}")
        frames_path = os.path.join(self.videos_dir, Image_name)
        image = Image.open(frames_path)
        resize_transform = transforms.Resize((self.resize, self.resize))
        resized_image_tensor = resize_transform(image)
        resized_image_tensor = self.transform(resized_image_tensor)  # 应用相同的变换（如归一化）
        if image.width > image.height:
            new_width = 600
            new_height = int(image.height * (new_width / image.width))
        else:
            new_height = 600
            new_width = int(image.width * (new_height / image.height))

        image_cropped = image.resize((new_width, new_height))

        # 转换为tensor
        image_tensor = self.transform(image_cropped)

        # 随机裁剪三个224×224的patch
        crop = transforms.RandomCrop((self.resize, self.resize))
        transformed_rs = torch.zeros([self.frames, 3, self.resize, self.resize])

        for i in range(self.frames):
            cropped_image = crop(image_tensor)
            transformed_rs[i] = cropped_image

        return transformed_rs, resized_image_tensor, Image_score, Image_name


class ImageDataset_VQC(data.Dataset):
    def __init__(self, data_dir, filename_path, transform, resize, frames, database_name):
        super(ImageDataset_VQC, self).__init__()

        dataInfo = pd.read_csv(filename_path)
        self.Image_names = dataInfo['Image_name'].tolist()
        self.score = dataInfo['MOS'].tolist()
        length = len(self.Image_names)
        seed = 20250611
        random.seed(seed)
        np.random.seed(seed)
        index_rd = np.random.permutation(length)
        train_source = index_rd[:int(length * 0.95)]
        val_source = index_rd[int(length * 0.95):]
        if database_name == 'train':
            self.Image_names = dataInfo.iloc[train_source]['Image_name'].tolist()
            self.score = dataInfo.iloc[train_source]['MOS'].tolist()
        elif database_name == 'val':
            self.Image_names = dataInfo.iloc[val_source]['Image_name'].tolist()
            self.score = dataInfo.iloc[val_source]['MOS'].tolist()
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
        Image_score = torch.FloatTensor(np.array(float(self.score[idx])))
        # print(f"正在读取视频 {Image_name}，分数 {Image_score}")
        frames_path = os.path.join(self.videos_dir, Image_name)
        image = Image.open(frames_path)

        resize_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])

        resized_image_tensor = resize_transform(image)

        resized_image_tensor = self.transform(resized_image_tensor)  # 应用相同的变换（如归一化）
        #
        # if image.width > image.height:
        #     new_width = 600
        #     new_height = int(image.height * (new_width / image.width))
        # else:
        #     new_height = 600
        #     new_width = int(image.width * (new_height / image.height))

        # image_cropped = image.resize((new_width, new_height))

        # 转换为tensor
        image_tensor = self.transform(image)

        # 随机裁剪三个224×224的patch
        crop = transforms.RandomCrop((self.resize, self.resize))

        transformed_rs = torch.zeros([self.frames, 3, self.resize, self.resize])

        for i in range(self.frames):
            cropped_image = crop(image_tensor)
            transformed_rs[i] = cropped_image

        return transformed_rs, resized_image_tensor, Image_score, Image_name


def extract_patches(image, dots, patch_size=(224, 224)):

    if isinstance(image, Image.Image):
        image = np.array(image)
        # image = np.asarray(image).astype('float32') / 255.0

    H, W, _ = image.shape
    ph, pw = patch_size

    base_width, base_height = 600, 800
    scale_w = W / float(base_width)
    scale_h = H / float(base_height)

    adjusted_dots = [(int(x * scale_w), int(y * scale_h)) for x, y in dots]

    offsets = get_random_offsets(num_points=len(adjusted_dots), max_offset=50)
    adjusted_dots_with_offset = [(x + dx, y + dy) for (x, y), (dx, dy) in zip(adjusted_dots, offsets)]

    patches = []
    for x, y in adjusted_dots_with_offset:
        x1 = max(0, min(x, W - pw))
        y1 = max(0, min(y, H - ph))
        x2 = x1 + pw
        y2 = y1 + ph

        patch = image[y1:y2, x1:x2]
        # patch = Image.fromarray(patch).resize(patch_size, Image.Resampling.LANCZOS)
        patches.append(patch)

    # 确保有3个patches，不足则重复最后一个patch
    while len(patches) < 3:
        patches.append(patches[-1])

    tensor_patches = [transforms.ToTensor()(patch).float() for patch in patches]

    return tensor_patches


class FIQADataset(data.Dataset):

    def __init__(self, data_dir, filename_path, transform, resize, frames, database_name):
        super(FIQADataset, self).__init__()

        dataInfo = pd.read_csv(filename_path)
        self.Image_names = dataInfo['Image_name'].tolist()
        self.score = dataInfo['MOS'].tolist()
        length = len(self.Image_names)
        seed = 20250629
        random.seed(seed)
        np.random.seed(seed)
        index_rd = np.random.permutation(length)
        train_source = index_rd[:int(length * 0.95)]
        val_source = index_rd[int(length * 0.95):]
        if database_name == 'train':
            self.Image_names = dataInfo.iloc[train_source]['Image_name'].tolist()
            self.score = dataInfo.iloc[train_source]['MOS'].tolist()
        elif database_name == 'val':
            self.Image_names = dataInfo.iloc[val_source]['Image_name'].tolist()
            self.score = dataInfo.iloc[val_source]['MOS'].tolist()
        self.resize = resize
        self.videos_dir = data_dir
        self.transform = transform
        self.length = len(self.Image_names)
        self.frames = frames
        self.dots = [(190, 431), (129, 207), (353, 185)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        image_name = str(self.Image_names[idx]) + '.png'
        # Image_score = torch.FloatTensor(np.array(float(self.score[idx])))
        Image_score = self.score[idx]

        frames_path = os.path.join(self.videos_dir, image_name)
        image = Image.open(frames_path).convert('RGB')

        transformed_rs = extract_patches(image, self.dots)

        resize_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            self.transform
        ])
        resized_image_tensor = resize_transform(image)
        transformed_rs.append(resized_image_tensor)
        data = torch.stack(transformed_rs, dim=0)

        return data, Image_score, image_name

