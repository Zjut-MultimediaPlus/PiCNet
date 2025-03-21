import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os

def sob_img_fast(img):
    sobel_y = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32)  # Sobel operator for edge detection
    sobel_x = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32)

    img_padded = F.pad(img, (1, 1, 1, 1), mode='constant', value=0)
    img_sobel_x = F.conv2d(img_padded.unsqueeze(1), sobel_x.unsqueeze(0).unsqueeze(0))
    img_sobel_y = F.conv2d(img_padded.unsqueeze(1), sobel_y.unsqueeze(0).unsqueeze(0))
    img_s = torch.abs(img_sobel_x.squeeze(1)) + torch.abs(img_sobel_y.squeeze(1))  # Combine gradients

    return img_s

def log(data, a):  # loga(b)=ln(b)/ln(a)
    x_ones = torch.ones_like(data)
    a = x_ones * a
    data = torch.clamp(data, min=0)  # 把所有小于0的都变成0
    data = data + 1  # 相当于log（1+x）把所有数都挪到>0那边
    data = torch.log(data) / torch.log(a)
    return data

def default_loader(path):
    # print(path)
    raw_data = np.load(path, allow_pickle=True)
    tensor_data = torch.from_numpy(raw_data)  # np转化为tensor

    return tensor_data

class img_sobel(Dataset):
    def __init__(self, data_dir, transform=None, data_format='npy'):
        self.data_dir = data_dir
        self.data_name = []
        for npy_file in os.listdir(data_dir[0]):
            self.data_name.append(npy_file)  # 0-4999.npy
        self.transform = transform
        self.data_format = data_format

    def __getitem__(self, index):
        x_data_root = self.data_dir[0] + '/' + self.data_name[index]
        result = default_loader(x_data_root)
        result = result * 4783 / 100 * 12
        x = result[:6]  # [6,288,288]
        sobel_x = sob_img_fast(x)  # [6,288,288]
        x = torch.cat([x, sobel_x], dim=0)  # [12,288,288]
        y = result[6:]  # [12,288,288]
        sobel_y = sob_img_fast(y)  # [12,288,288]
        y = torch.cat([y, sobel_y], dim=0)  # [24,288,288]
        npy_name = self.data_name[index]

        return x, y, npy_name

    def __len__(self):
        return len(self.data_name)

