import math
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import random

import torch.nn.functional as F
from scipy.signal import convolve2d

def gen_grid(batch_size, width, height, device):
    img_n_grid_x = width  # 4
    img_n_grid_y = height  # 4
    img_dx = 1./img_n_grid_y  # 1/4
    c_x, c_y = torch.meshgrid(torch.arange(img_n_grid_x), torch.arange(img_n_grid_y), indexing = "ij")
    img_x = img_dx * (torch.cat((c_x[..., None], c_y[..., None]), axis = 2) + 0.5).to(device) # grid center locations
    img_x = torch.unsqueeze(img_x, 0)
    img_x = img_x.repeat(batch_size, 1, 1, 1)
    return img_x

def show(a):
    plt.imshow(a)
    plt.show()

def to_np(a):
    return a.cpu().detach().numpy()

def get_data():
    root = '/data/zhy/hhcode/data/KNMI/KNMI.h5'
    f = h5py.File(root, mode='r')  # 整个数据包含train和test两个hdf5 dataset
    train = f['train']
    test = f['test']
    train_image = train['images']  # 大小(5734,18,288,288)
    test_image = test['images']  # 大小(1557,18,288,288)
    return train_image, test_image

def psi(a, scale=4):
    # a shape is [B, S, H, W]
    B, S, H, W = a.shape
    C = scale ** 2
    new_H = int(H // scale)  # 变小scale倍
    new_W = int(W // scale)
    a = np.reshape(a, (B, S, new_H, scale, new_W, scale))  # 64,64变成16,4,16,4
    a = np.transpose(a, (0, 1, 3, 5, 2, 4))  # B，S,4，4，16，16
    a = np.reshape(a, (B, S, C, new_H, new_W))  # B，S，16,16,16
    return a

def inverse(a, scale=4):
    B, S, C, new_H, new_W = a.shape
    H = int(new_H * scale)
    W = int(new_W * scale)
    a = np.reshape(a, (B, S, scale, scale, new_H, new_W))
    a = np.transpose(a, (0, 1, 4, 2, 5, 3))
    a = np.reshape(a, (B, S, H, W))
    return a

def get_mask(eta, shape, test=False):
    B, S, C, H, W = shape
    if test:
        return torch.zeros((B, int(S // 2), C, H, W))
    eta -= 0.00002
    if eta < 0:
        eta = 0
    mask = np.random.random_sample((B, int(S // 2), C, H, W))
    mask[mask < eta] = 0
    mask[mask > eta] = 1
    return eta, torch.tensor(mask, dtype=torch.float)

def data_2_rnn_mask(data, batch, batch_size, sequence, scale, eta, test=False):
    sequence = sequence[batch - batch_size:batch]
    result = []
    for i in sequence:
        tmp = data[i] * 4783 / 100 * 12
        result.append(torch.tensor(tmp, dtype=torch.float))
    result = torch.stack(result, dim=0)
    result = psi(result, scale=scale)

    B, S, C, H, W = result.shape

    if test:
        return result, torch.zeros((B, int(S // 2), C, H, W))
    eta -= 0.00002
    if eta < 0:
        eta = 0

    mask = np.random.random_sample((B, int(S // 2), C, H, W))
    mask[mask < eta] = 0
    mask[mask > eta] = 1

    return result, torch.tensor(mask, dtype=torch.float), eta


def data_2_rnn(data, batch, batch_size, sequence, scale):
    sequence = sequence[batch - batch_size:batch]
    result = []
    for i in sequence:
        tmp = data[i] * 4783 / 100 * 12
        result.append(torch.tensor(tmp, dtype=torch.float))
    result = torch.stack(result, dim=0)
    result = psi(result, scale=scale)
    return result

def data_2_cnn(data, batch, batch_size, sequence):
    sequence = sequence[batch - batch_size:batch] # 切sequence里barch_size长的一段
    result = []
    for i in sequence:
        tmp = data[i] * 4783 / 100 * 12  # 这个处理是为什么？数值处理
        result.append(torch.tensor(tmp, dtype=torch.float))  # data[i]的大小是18*288*288
    result = torch.stack(result, dim=0)  # 按照维度dim=0 拼一下 result最后是6*18*288*288
    x = result[:, :9]
    y = result[:, 9:]
    return x, y

def data_2_rainnet(data, batch, batch_size, sequence):
    sequence = sequence[batch - batch_size:batch]
    result = []
    for i in sequence:
        tmp = data[i] * 4783 / 100 * 12
        result.append(torch.tensor(tmp, dtype=torch.float))
    result = torch.stack(result, dim=0)
    x = result[:, :4]
    y = result[:, 4:5]
    return x, y

def data_2_rainnet_test(data, batch, batch_size, sequence):
    sequence = sequence[batch - batch_size:batch]
    result = []
    for i in sequence:
        tmp = data[i] * 4783 / 100 * 12
        result.append(torch.tensor(tmp, dtype=torch.float))
    result = torch.stack(result, dim=0)
    return result

def data_2_cnn2(data, batch, batch_size, sequence, scale=3):
    x, y = data_2_cnn(data, batch, batch_size, sequence)
    x2 = torch.zeros((x.shape[0], 1, x.shape[2] * 3, x.shape[3] * 3))
    y2 = torch.zeros((y.shape[0], 1, y.shape[2] * 3, y.shape[3] * 3))
    index = 0
    for i in range(0, x2.shape[2], x.shape[2]):
        for j in range(0, x2.shape[3], x.shape[3]):
            x2[:, 0, i:i+x.shape[2], j:j+x.shape[2]] = x[:, index]
            y2[:, 0, i:i+x.shape[2], j:j+x.shape[2]] = y[:, index]
            index += 1
    x2 = psi(x2, scale=scale)
    y2 = psi(y2, scale=scale)
    x2 = torch.squeeze(x2)
    y2 = torch.squeeze(y2)
    return x2, y2

def inverse_cnn2(x, y):
    x = torch.unsqueeze(x, dim=1)
    y = torch.unsqueeze(y, dim=1)
    x = to_np(x)
    y = to_np(y)
    x = inverse(x, scale=3)
    y = inverse(y, scale=3)

    x2 = np.zeros((x.shape[0], 9, 288, 288))
    y2 = np.zeros((y.shape[0], 9, 288, 288))

    index = 0

    for i in range(0, 864, 288):
        for j in range(0, 864, 288):
            x2[:, index] = x[:, 0, i:i+x2.shape[2], j:j+x2.shape[2]]
            y2[:, index] = y[:, 0, i:i+x2.shape[2], j:j+x2.shape[2]]
            index += 1
    return x2, y2



def _draw_color(t, flag, color):
    r = t[:, :, 0]
    g = t[:, :, 1]
    b = t[:, :, 2]
    r[flag] = color[0]
    g[flag] = color[1]
    b[flag] = color[2]
    return t

def draw_color_single(y):
    t = np.ones((y.shape[0], y.shape[1], 3)) * 255
    rain_1 = y >= 0.5
    rain_2 = y >= 2
    rain_3 = y >= 5
    rain_4 = y >= 10
    rain_5 = y >= 30
    _draw_color(t, rain_1, [156, 247, 144])
    _draw_color(t, rain_2, [55, 166, 0])
    _draw_color(t, rain_3, [103, 180, 248])
    _draw_color(t, rain_4, [0, 2, 254])
    _draw_color(t, rain_5, [250, 3, 240])
    t = t.astype(np.uint8)
    return t

def fundFlag(a, n, m):
    flag_1 = np.uint8(a >= n)
    flag_2 = np.uint8(a < m)
    flag_3 = flag_1 + flag_2
    return flag_3 == 2

def B_mse(a, b):
    mask = np.zeros(a.shape)
    mask[a < 2] = 1
    mask[fundFlag(a, 2, 5)] = 2
    mask[fundFlag(a, 5, 10)] = 5
    mask[fundFlag(a, 10, 30)] = 10
    mask[a > 30] = 30
    n = a.shape[0] * b.shape[0]
    mse = np.sum(mask * ((a - b) ** 2)) / n
    return mse

def B_mae(a, b):
    mask = np.zeros(a.shape)
    mask[a < 2] = 1
    mask[fundFlag(a, 2, 5)] = 2
    mask[fundFlag(a, 5, 10)] = 5
    mask[fundFlag(a, 10, 30)] = 10
    mask[a > 30] = 30
    n = a.shape[0] * b.shape[0]
    mae = np.sum(mask * np.abs(a - b)) / n
    return mae

def draw_color(data):
    B, C, H, W = data.shape
    result = torch.zoers((B, C, H, W, 3))
    for i in range(B):
        for j in range(C):
            result[B, C] = draw_color_single(data[B, C])
    return result

def tp(pre, gt):
    return np.sum(pre * gt)

def fn(pre, gt):
    a = pre + gt
    flag = (gt == 1) & (a == 1)
    return np.sum(flag)

def fp(pre, gt):
    a = pre + gt
    flag = (pre == 1) & (a == 1)
    return np.sum(flag)

def tn(pre, gt):
    a = pre + gt
    flag = a == 0
    return np.sum(flag)

def _csi(pre, gt):
    eps = 1e-9
    TP, FN, FP, TN = tp(pre, gt), fn(pre, gt), fp(pre, gt), tn(pre, gt)
    return TP / (TP + FN + FP + eps)


def _hss(pre, gt):
    eps = 1e-9
    TP, FN, FP, TN = tp(pre, gt), fn(pre, gt), fp(pre, gt), tn(pre, gt)
    a = TP * TN - FN * FP
    b = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN) + eps
    if a / b < 0:
        return 0
    return a / b

def csi(pred, gt):
    threshold = [0.5, 2, 5, 10, 30]
    result = []
    for i in threshold:
        a = np.zeros(pred.shape)
        b = np.zeros(gt.shape)
        a[pred >= i] = 1
        b[gt >= i] = 1
        result.append(_csi(a, b))
    return result

def hss(pred, gt):
    threshold = [0.5, 2, 5, 10, 30]
    result = []
    for i in threshold:
        a = np.zeros(pred.shape)
        b = np.zeros(gt.shape)
        a[pred >= i] = 1
        b[gt >= i] = 1
        result.append(_hss(a, b))
    return result

import torch
from torch import nn

class BMAEloss(nn.Module):
    def __init__(self):
        super(BMAEloss, self).__init__()

    def fundFlag(self, a, n, m):
        flag_1 = (a >= n).int()  # a里面大于等于n的全1，flag和a大小一样
        # print('flag_1:', flag_1)
        flag_2 = (a < m).int()  # a里面小于m的全1，flag和a大小一样
        flag_3 = flag_1 + flag_2  # 处于[n,m)之间的值加起来=2
        # print('flag_3:', flag_3)
        # print('flag_3 == 2:', flag_3 == 2)
        return flag_3 == 2  # 如果=2返回true，那么返回的是处于[n,m)之间的true，其他的false

    def forward(self, pred, y):
        mask = torch.zeros(y.shape).cuda()  # mask是和y一样大的全0tensor
        mask[y < 2] = 1  # 把小于2的地方赋值1
        mask[self.fundFlag(y, 2, 5)] = 2  # 2-5的赋值2
        mask[self.fundFlag(y, 5, 10)] = 5  # 5-10的赋值5
        mask[self.fundFlag(y, 10, 30)] = 10  # 10-30的赋值10
        mask[y > 30] = 30  # 大于30的赋值30
        return torch.sum(mask * torch.abs(y - pred))  # abs（y - pred）*mask



def pro_statistic(img, fill):
    res = (img.detach().cpu() * fill).sum()
    # print(img.shape)
    # print(fill.shape)
    # print(res)
    return res

def pre_convolve(img, fill):
    batch_size = img.shape[0]
    img_height = img.shape[1]
    img_width = img.shape[2]
    padding = (1, 1, 1, 1)
    padded_img = F.pad(img, padding, mode='constant', value=0)
    fill_height = fill.shape[0]
    fill_width = fill.shape[1]
    # fill = fill.unsqueeze(0)
    fill = fill.reshape(1, fill.shape[0], fill.shape[1])

    fill = fill.repeat(batch_size, axis=0)
    # print(fill.shape)

    img_pro_height = img_height
    img_pro_width = img_width
    # print(img_pro_width)
    # print(img.shape)
    pro_rgb = np.zeros((batch_size, img_height, img_width), dtype='uint8')

    for i in range(img_pro_height):
        for j in range(img_pro_width):
            pro_rgb[:, i, j] = pro_statistic(padded_img[:, i:i + fill_height, j:j + fill_width], fill)
    return pro_rgb

def convolve(motion, fill):
    # print(motion.shape)
    vx = motion[:, 0]
    vy = motion[:, 1]

    pro_vx = pre_convolve(vx, fill)
    pro_vy = pre_convolve(vy, fill)

    v_pro = np.dstack((pro_vx, pro_vy))

    return v_pro

def convolve1(motion, fill):
    # print(motion.shape)
    vx = motion[:, 0]
    vy = motion[:, 1]
    vx_cpu = vx.cpu()
    vy_cpu = vy.cpu()
    result_x = np.zeros_like(vx_cpu.detach().numpy())  # 创建一个空的输出张量
    for i in range(motion.shape[0]):
        result_x[i] = convolve2d(vx_cpu[i].detach().numpy(), fill, mode='same')
    result_y = np.zeros_like(vx_cpu.detach().numpy())  # 创建一个空的输出张量
    for i in range(motion.shape[0]):
        result_y[i] = convolve2d(vy_cpu[i].detach().numpy(), fill, mode='same')
    # pro_vx = pre_convolve(vx, fill)
    # pro_vy = pre_convolve(vy, fill)

    v_pro = np.dstack((result_x, result_y))

    return v_pro

def convolve2(motion, fill):
    # Assuming motion is a PyTorch tensor of shape [B, C, H, W] where C is the number of channels
    # Assuming fill is a numpy array of shape [3, 3]

    # Separate vx and vy channels
    vx = motion[:, 0]
    vy = motion[:, 1]

    # Convert fill numpy array to torch tensor
    fill_tensor = torch.tensor(fill, dtype=torch.float32)
    fill_tensor_expanded = fill_tensor.unsqueeze(0).unsqueeze(0).repeat(10, 1, 1, 1).cuda()

    # Perform convolution along the last two dimensions for vx and vy separately
    result_x = torch.nn.functional.conv2d(vx.unsqueeze(1), fill_tensor_expanded, padding=1, stride=1)
    result_x = result_x[:, 0]
    result_y = torch.nn.functional.conv2d(vy.unsqueeze(1), fill_tensor_expanded, padding=1, stride=1)
    result_y = result_y[:, 0]
    # Stack the results along the channel dimension
    v_pro = torch.stack((result_x, result_y), dim=1)

    return v_pro  # [b,2,288,288]

class Motionloss(nn.Module):
    def __init__(self):
        super(Motionloss, self).__init__()

    def fundFlag(self, a, n, m):
        flag_1 = (a >= n).int()  # a里面大于等于n的全1，flag和a大小一样
        flag_2 = (a < m).int()  # a里面小于m的全1，flag和a大小一样
        flag_3 = flag_1 + flag_2  # 处于[n,m)之间的值加起来=2
        return flag_3 == 2  # 如果=2返回true，那么返回的是处于[n,m)之间的true，其他的false


    def forward(self, v, y):
        mask = torch.zeros(y.shape).cuda()  # mask是和y一样大的全0tensor
        mask[y >= 23] = 0 # 把小于24的地方赋值 0
        y_min = y * mask + 1
        y_min = torch.sqrt(y_min)
        mask[y >= 23] = math.sqrt(24)  # 大于30的赋值30
        mask = mask + y_min
        sobel_y = np.array([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]])  # 索贝尔算子 边缘检测
        sobel_x = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])
        v1 = convolve2(v, sobel_y)
        v2 = convolve2(v, sobel_x)
        l1 = mask * (v1[:, 0] + v2[:, 1])
        l2 = mask * (v1[:, 0] + v2[:, 1])
        # l1_tensor = torch.from_numpy(l1)  # 将 numpy 数组转换为 PyTorch 张量
        # l2_tensor = torch.from_numpy(l2)

        # 在 PyTorch 中执行平方操作
        loss_motion = torch.sum(torch.square(l1) + torch.square(l2))
        # loss_motion = torch.sum(torch.square(l1)+torch.square(l2))

        return loss_motion  # abs（y - pred）*mask

class Wdisloss(nn.Module):
    def __init__(self):
        super(Wdisloss, self).__init__()

    def fundFlag(self, a, n, m):
        flag_1 = (a >= n).int()  # a里面大于等于n的全1，flag和a大小一样
        flag_2 = (a < m).int()  # a里面小于m的全1，flag和a大小一样
        flag_3 = flag_1 + flag_2  # 处于[n,m)之间的值加起来=2
        return flag_3 == 2  # 如果=2返回true，那么返回的是处于[n,m)之间的true，其他的false

    def forward(self, pred, y):
        mask = torch.zeros(y.shape).cuda()  # mask是和y一样大的全0tensor
        mask[y >= 23] = 0  # 把小于24的地方赋值 0
        y_min = y * mask + 1
        y_min = torch.sqrt(y_min)
        mask[y >= 23] = math.sqrt(24)  # 大于30的赋值30
        mask = mask + y_min
        return torch.sum(mask * torch.abs(y - pred))  # abs（y - pred）*mask


def MSE(y_hat, y):
    sub = y_hat - y
    return np.sum(sub * sub)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Random seed
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现
    #torch.backends.cudnn.enabled = False

def warp(input, flow, grid, mode="bilinear", padding_mode="zeros"):
    B, C, H, W = input.size()
    vgrid = grid + flow

    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(input, vgrid, padding_mode=padding_mode, mode=mode, align_corners=True)
    return output

def make_grid(input):
    B, C, H, W = input.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).cuda()
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).cuda()
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    return grid




