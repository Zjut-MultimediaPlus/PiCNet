import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

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


class KNMIDataset(Dataset):
    def __init__(self, data_dir, transform=None, data_format='npy'):
        """
        纸币分类任务的Dataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        # self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.data_dir = data_dir  # 传进来x,z两个地址
        self.data_name = []  # 存分片的文件名
        for npy_file in os.listdir(data_dir[0]):
            self.data_name.append(npy_file)  # 0-4999.npy
        # self.lens = len(default_loader(data_dir[0]).shape[0])
        # for i in self.data_dir:
        #     self.data.append(default_loader(i))
        # self.data = default_loader(self.data_dir)
        self.transform = transform
        self.data_format = data_format
        # for i in range(len(self.data)):
        #     self.data = self.data[i] * 4783 / 100 * 12  # 数值处理为mm/h的降雨量

    def __getitem__(self, index):  # 得到一组连续时间的降雨数据
        x_data_root = self.data_dir[0] + '/' + self.data_name[index]
        result = default_loader(x_data_root)
        result = result * 4783 / 100 * 12  # 数值处理为mm/h的降雨量
        # result = log(result, 7)
        # z部分：
        z_data_root = self.data_dir[1] + '/' + self.data_name[index][:-4]
        z_data_root = os.path.join(z_data_root, 'tests/1000/imgs/z.npy')
        z = np.load(z_data_root, allow_pickle=True)
        z = np.rot90(z, 3, axes=(1, 2))
        z = z.copy()
        z = torch.from_numpy(z)
        z = z * 4783 / 100 * 12  # 数值处理为mm/h的降雨量
        # z = log(z, 7)
        if self.transform is not None:
            result = self.transform(result)
            z = self.transform(z)

        x = result[:6]  # 前6帧是输入  # [6,288,288]0
        y = result[6:]  # 后12帧是输出  # [12,288,288]
        # z = y
        z = z[-12:]  # z是[12,288,288]
        # print(index)
        return x, y, z

    def __len__(self):
        return len(self.data_name)

class K_VDataset(Dataset):
    def __init__(self, data_dir, transform=None, data_format='npy'):

        #  x的地址是'/data/zhy/hhcode/data/KNMI/logs5/0/tests/1000/vorts/017.npy'
        self.data_dir = data_dir  # 传进来x,y两个地址 '/data/zhy/hhcode/data/KNMI/logs5'
        self.data_name = []  # 存分片的文件名
        for npy_file in os.listdir(data_dir[2]):
            self.data_name.append(npy_file)  # 用这个不小心新建别的文件夹的时候就会出错
        # for i in range(5000):
        #     self.data_name.append(str(i))
        self.transform = transform
        self.data_format = data_format
        self.data_max = 0.68848
        self.train_min = 0.0

    def __getitem__(self, index):
        x_data_root = self.data_dir[0] + '/' + self.data_name[index] + '.npy'
        result = default_loader(x_data_root)

        x = result[:6]  # 前6帧是输出 [6,288,288]
        y = result[6:]  # [12,288,288]
        img_vel_data_root = self.data_dir[2] + '/' + self.data_name[index]
        join_dir = 'tests/1000/vorts/img_vel.npy'
        img_vel_data_root = os.path.join(img_vel_data_root, join_dir)
        img_vel = default_loader(img_vel_data_root)
        vort_data_root = self.data_dir[3] + '/' + self.data_name[index]
        join_dir = 'tests/1000/vorts/017.npy'
        vort_data_root = os.path.join(vort_data_root, join_dir)
        vort = default_loader(vort_data_root)
        return x, y, img_vel, vort

    def __len__(self):
        return len(self.data_name)

class K_ZDataset(Dataset):
    def __init__(self, data_dir, transform=None, data_format='npy'):
        # self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.data_dir = data_dir  # 传进来x,z两个地址
        self.data_name = []  # 存分片的文件名
        for npy_file in os.listdir(data_dir[0]):
            self.data_name.append(npy_file)  # 0-4999.npy
        # self.lens = len(default_loader(data_dir[0]).shape[0])
        # for i in self.data_dir:
        #     self.data.append(default_loader(i))
        # self.data = default_loader(self.data_dir)
        self.transform = transform
        self.data_format = data_format
        # for i in range(len(self.data)):
        #     self.data = self.data[i] * 4783 / 100 * 12  # 数值处理为mm/h的降雨量

    def __getitem__(self, index):  # 得到一组连续时间的降雨数据
        x_data_root = self.data_dir[0] + '/' + self.data_name[index]
        result = default_loader(x_data_root)
        result = result * 4783 / 100 * 12  # 数值处理为mm/h的降雨量
        # result = log(result, 7)
        # z部分：
        z_data_root = self.data_dir[1] + '/' + self.data_name[index]
        # z_data_root = os.path.join(z_data_root, 'tests/1000/imgs/z.npy')
        # z = np.load(z_data_root, allow_pickle=True)
        z = default_loader(z_data_root)
        # z = np.rot90(z, 3, axes=(1, 2))
        # z = z.copy()
        # z = torch.from_numpy(z)
        z = z * 4783 / 100 * 12  # 数值处理为mm/h的降雨量
        # z = log(z, 7)
        if self.transform is not None:
            result = self.transform(result)
            z = self.transform(z)

        x = result[:6]  # 前6帧是输入  # [6,288,288]0
        y = result[6:]  # 后12帧是输出  # [12,288,288]
        # z = y
        z = z[-12:]  # z是[12,288,288]
        # print(index)
        npy_name = self.data_name[index]

        return x, y, z

    def __len__(self):
        return len(self.data_name)


class imgDataset6_12(Dataset):
    def __init__(self, data_dir, transform=None, data_format='npy'):
        # self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.data_dir = data_dir  # 传进来x,z两个地址
        self.data_name = []  # 存分片的文件名
        for npy_file in os.listdir(data_dir[0]):
            self.data_name.append(npy_file)  # 0-4999.npy
        self.transform = transform
        self.data_format = data_format

    def __getitem__(self, index):  # 得到一组连续时间的降雨数据
        x_data_root = self.data_dir[0] + '/' + self.data_name[index]
        result = default_loader(x_data_root)
        result = result * 4783 / 100 * 12  # 数值处理为mm/h的降雨量
        x = result[:6]  # 前6帧是输入  # [6,288,288]
        y = result[6:]  # 后12帧是输出  # [12,288,288]
        npy_name = self.data_name[index]

        return x, y, npy_name

    def __len__(self):
        return len(self.data_name)
# data_dir = '/data/zhy/hhcode/data/KNMI/train.npy'
# data_format = 'npy'
# train_dataset = KNMIDataset(data_dir, None, data_format)
# train_loader = DataLoader(dataset=train_dataset, batch_size=5, shuffle=False)
# for epoch in range(3):  # 训练所有!整套!数据 3 次
#     for step, (x, y) in enumerate(tqdm(train_loader)):
#         # 每一步 loader 释放一小批数据用来学习
#                                 #return:
#                                         #(tensor(x1,x2,x3,x4,x5),tensor(y1,y2,y3,y4,y5))
#                                         #(tensor(x6,x7,x8,x9,x10),tensor(y6,y7,y8,y9,y10))
#         # 假设这里就是你训练的地方...
#
#         # 打出来一些数据
#         print('Epoch: ', epoch, '| Step:', step)  # , '| batch x: ', batch_data.numpy())
# path = '/data/zhy/hhcode/data/KNMI/train//1106.npy'
# a = np.load(path)
# print(a)