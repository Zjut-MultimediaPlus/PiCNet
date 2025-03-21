from Unet import *
from simulation_utils import *
import matplotlib.pyplot as plt

def fundFlag(a, n, m):
    flag_1 = np.uint8(a >= n)
    flag_2 = np.uint8(a < m)
    flag_3 = flag_1 + flag_2
    return flag_3 == 2

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

class BMAEloss(nn.Module):
    def __init__(self):
        super(BMAEloss, self).__init__()

    def fundFlag(self, a, n, m):
        flag_1 = (a >= n).int()
        flag_2 = (a < m).int()
        flag_3 = flag_1 + flag_2
        return flag_3 == 2

    def forward(self, pred, y):
        mask = torch.zeros(y.shape).cuda()
        mask[y < 2] = 1
        mask[self.fundFlag(y, 2, 5)] = 2
        mask[self.fundFlag(y, 5, 10)] = 5
        mask[self.fundFlag(y, 10, 30)] = 10
        mask[y > 30] = 30
        return torch.sum(mask * torch.abs(y - pred))

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

def to_np(a):
    return a.cpu().detach().numpy()

def show(a):
    plt.imshow(a)
    plt.show()

def _draw_color(t, flag, color):
    r = t[:, :, 0]
    g = t[:, :, 1]
    b = t[:, :, 2]
    # color = color.cpu().numpy()
    # print(color.is_cuda(), r.is_cuda())
    r[flag] = color[0]
    g[flag] = color[1]
    b[flag] = color[2]
    return t

def draw_color_single(y):
    t = np.ones((y.shape[0], y.shape[1], 3)) * 255
    tt1 = []
    index = 0.5
    for i in range(30):
        tt1.append(index)
        index += 1
    color = [[28, 230, 180], [39, 238, 164], [58, 245, 143], [74, 248, 128], [97, 252, 108],
             [121, 254, 89], [143, 255, 73], [159, 253, 63], [173, 251, 56], [190, 244, 52],
             [203, 237, 52], [215, 229, 53], [227, 219, 56], [238, 207, 58], [246, 195, 58],
             [251, 184, 56], [254, 168, 51], [254, 153, 44], [253, 138, 38], [249, 120, 30],
             [244, 103, 23], [239, 88, 17], [231, 73, 12], [221, 61, 8], [212, 51, 5],
             [202, 42, 4], [188, 32, 2], [172, 23, 1], [158, 16, 1], [142, 10, 1]]

    for i in range(30):
        rain = y >= tt1[i]
        _draw_color(t, rain, color[i])
    t = t.astype(np.uint8)
    return t

class z_max(nn.Module):
    def __init__(self, in_channels=12, n_hidden_1=64, n_hidden_2=12):
        super(z_max, self).__init__()
        self.mpool = nn.AdaptiveMaxPool2d(1)
        self.apool = nn.AdaptiveAvgPool2d(1)
        self.layer1 = nn.Linear(in_channels, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)

    def forward(self, x):
        x = self.mpool(x) #  b,12,288,288--> b,12,1,1
        x = torch.squeeze(x) # b,12,1,1 --> b,12
        x = F.relu(self.layer1(x))  # b,12 --> b,64
        x = self.layer2(x)  # b,64 --> b,64
        return x


class z_avg(nn.Module):
    def __init__(self, in_channels=12, n_hidden_1=64, n_hidden_2=12):
        super(z_avg, self).__init__()
        self.mpool = nn.AdaptiveMaxPool2d(1)
        self.apool = nn.AdaptiveAvgPool2d(1)
        self.layer1 = nn.Linear(in_channels, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)

    def forward(self, x):
        x = self.apool(x)
        x = torch.squeeze(x)  # b,12,1,1 --> b,12
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x

class x_max(nn.Module):
    def __init__(self, in_channels=6, n_hidden_1=64, n_hidden_2=12):
        super(x_max, self).__init__()
        self.mpool = nn.AdaptiveMaxPool2d(1)
        self.apool = nn.AdaptiveAvgPool2d(1)
        self.layer1 = nn.Linear(in_channels, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)

    def forward(self, x):
        x = self.mpool(x)
        x = torch.squeeze(x)  # b,12,1,1 --> b,12
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

class x_avg(nn.Module):
    def __init__(self, in_channels=6, n_hidden_1=64, n_hidden_2=12):
        super(x_avg, self).__init__()
        self.mpool = nn.AdaptiveMaxPool2d(1)
        self.apool = nn.AdaptiveAvgPool2d(1)
        self.layer1 = nn.Linear(in_channels, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)

    def forward(self, x):
        x = self.mpool(x)
        x = torch.squeeze(x)  # b,12,1,1 --> b,12
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x



def basic_sigmoid(x):
    s = 1 / (1 + torch.exp(-x))
    return s

class TFC(nn.Module):
    def __init__(self, z_in_channels=12, z_out_channels=12, hidden_channels=128, x_in_channels=6, x_out_channels=12):
        super(TFC, self).__init__()
        height, width = 288, 288
        self.z_max = z_max(z_in_channels, hidden_channels, z_out_channels)
        self.z_avg = z_avg(z_in_channels, hidden_channels, z_out_channels)
        self.x_max = x_max(x_in_channels, hidden_channels, x_out_channels)
        self.x_avg = x_avg(x_in_channels, hidden_channels, x_out_channels)
        self.layer1 = nn.Linear(12, 12)  # 全连接
        self.sigmoid = basic_sigmoid
        self.conv = nn.Sequential(
            nn.Conv2d(x_in_channels, x_out_channels, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([z_out_channels, height, width])
        )


    def forward(self, z, x):
        z1 = self.z_avg(z) + self.z_max(z)
        b, c = z.shape[0], z.shape[1]
        z1 = self.layer1(z1).view(b, c, 1, 1)
        x1 = self.conv(x)
        w1 = self.sigmoid(x1)
        z2 = z1 * w1 + x1

        return z2



def cat(x, z):
    data = []
    for j in range(x.shape[0]):
        lst = []
        for i in range(x.shape[1]):
            lst.append(x[j][i])
        for i in range(z.shape[1]):
            lst.append(z[j][i])
        data.append(torch.stack(lst, 0))
    data = torch.stack(data, 0)
    return data

class Net(nn.Module):
    def __init__(self, loga, in_channel=30, out_channel=12):
        super(Net, self).__init__()
        self.loga = loga
        self.TFC = TFC(z_in_channels=12,z_out_channels=12,hidden_channels=256,x_in_channels=6,x_out_channels=12)
        self.inc = DoubleConv(in_channel, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, out_channel)

    def forward(self, z, x):
        z1 = self.TFC(z, x)
        if z1.shape[0] == 12:
            z1 = torch.unsqueeze(z1, 0)
            z1 = z1.expand(1, z1.shape[1], 288, 288)
        else:
            z1 = z1.expand(z1.shape[0], z1.shape[1], 288, 288)
        x = cat(x, z1)
        x = cat(x, z)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        out = out + z
        if self.loga != 1:
            out = torch.pow(self.loga, out) - 1

        return out

