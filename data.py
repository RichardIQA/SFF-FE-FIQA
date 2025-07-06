import os
import random
import pandas as pd
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import crop as functional_crop


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
    C, H, W = image.shape
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

    tensor_patches = torch.cat([patch.unsqueeze(0) for patch in patches], dim=0)     # [b,3,c,h,w]

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
        image = self.transform(image)

        transformed_rs = extract_patches(image, self.dots)

        resize_transform = transforms.Resize((self.resize, self.resize), antialias=True)

        resized_image_tensor = resize_transform(image).unsqueeze(0)
        # print(resized_image_tensor.shape, transformed_rs.shape)

        data = torch.cat([transformed_rs, resized_image_tensor], dim=0)

        return data, Image_score, image_name



