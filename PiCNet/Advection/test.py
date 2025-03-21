from torch.utils.data import DataLoader
from AS import AD
from dataloder import *
from tqdm import tqdm
from simulation_utils import *
import sys
sys.path.append('..')
from tool import *
from skimage import measure
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# -----------------------------------------------
cuda_idx = 0
file_idx = 1
device = torch.device('cuda')


# -----------------------------------------------
x_test_data_dir= '/data/test'
img_test_data_dir = '/data/test'
test_data_dir = [x_test_data_dir, img_test_data_dir]
data_format = 'npy'
test_dataset = img_sobel(test_data_dir, None, data_format)

epoch_size, batch_size = 50, 16
in_channel = 6
out_channel = 12
setup_seed(0)
net = AD(out_channel, out_channel).cuda()
min_test_loss, out_count = 1e10, 0
min_mae = 1e10
relu = nn.ReLU()


net.load_state_dict(torch.load('12-17model0_0_100.pt', map_location=torch.device('cuda:0')))
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
MSE_criterion = nn.MSELoss(reduction='sum')
# MSE_criterion = BMAEloss().cuda()
mae_loss = torch.nn.L1Loss(reduction='sum')

# -----------------------------------------------
for step, (x, y, npy_name) in enumerate(test_dataloader):
    x = x.cuda()  # b*6*288*288
    y = y.cuda()
    print(npy_name)
    start_time = time.time()
    s, y_hat = net(x)
    y_hat = y_hat.view(x.shape[0], out_channel, 288, 288, 2)
    end_time = time.time()
    tmp = x[:, :6]
    for i in range(12):
        t = tmp[:, -1]
        if x.shape[0] != batch_size:
            img_x = gen_grid(x.shape[0], 288, 288, device)
        else:
            img_x = gen_grid(batch_size, 288, 288, device)
        t = t.unsqueeze(-1)
        imgs = t.repeat(1, 1, 1, 3)
        u = y_hat[:, i]
        u = torch.stack((u[:, :, :, 1], u[:, :, :, 0] * -1), dim=-1)
        new_img = advect_quantity_batched(imgs, u, img_x, dt=0.01, boundary=None).cuda()
        new_img = new_img[:, :, :, 0]
        tmp_s = s[:, i]
        new_img = new_img.squeeze() + tmp_s
        new_img = relu(new_img)
        if x.shape[0] > 1:
            new_img = new_img.unsqueeze(1)
        else:
            new_img = new_img.unsqueeze(0)
        tmp = torch.cat([tmp, new_img], dim=1)

        del imgs, u, img_x, new_img


    loss = MSE_criterion(tmp[:, 6:], y[:, :12]) / (12 * 288 * 288)
    loss_num = loss.detach().cpu().numpy()
    mse = loss_num / batch_size
    loss1 = mae_loss(tmp[:, 6:], y[:, :12]) / (12 * 288 * 288)
    loss1 = loss1.detach().cpu().numpy()
    mae = loss1 / batch_size

    index = 0
    y = to_np(y)
    y_hat = to_np(tmp[:, 6:])
    print(np.mean(y),np.mean(y_hat))
    print(np.max(y), np.max(y_hat))
    print(np.min(y), np.min(y_hat))
    print(np.sum(y_hat<0))
    for i in range(12):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(y[index][i]), axs[0].set_title('y')
        axs[1].imshow(y_hat[index][i], vmin=0), axs[1].set_title('y-hat')
        plt.savefig('{}.png'.format(i + 1))
        plt.show()

    print('MSE:', mse, 'MAE:', mae)
    if step == 0:
        break

net.eval()
pbar = tqdm(test_dataloader)

test_l_sum, n = 0.0, 0
test_l_sum1 = 0.0
CSI, HSS, mse1, mae1 = [], [], [], []
for i in range(5):
    CSI.append([])
    HSS.append([])
for step, (x, y, npy_name) in enumerate(pbar):
    x = x.cuda()
    y = y.cuda()
    s, y_hat = net(x)
    y_hat = y_hat.view(x.shape[0], out_channel, 288, 288, 2)
    tmp = x[:, :6]
    for i in range(12):
        t = tmp[:, -1]
        if x.shape[0] != batch_size:
            img_x = gen_grid(x.shape[0], 288, 288, device)
        else:
            img_x = gen_grid(batch_size, 288, 288, device)
        t = t.unsqueeze(-1)
        imgs = t.repeat(1, 1, 1, 3).cuda()
        u = y_hat[:, i]
        u = torch.stack((u[:, :, :, 1], u[:, :, :, 0] * -1), dim=-1).cuda()
        new_img = advect_quantity_batched(imgs, u, img_x, dt=0.01, boundary=None).cuda()
        new_img = new_img[:, :, :, 0]
        tmp_s = s[:, i]
        new_img = new_img.squeeze() + tmp_s
        new_img = relu(new_img)
        if x.shape[0] > 1:
            new_img = new_img.unsqueeze(1)
        else:
            new_img = new_img.unsqueeze(0)
        tmp = torch.cat([tmp, new_img], dim=1)

        del imgs, u, img_x, new_img


    loss = MSE_criterion(tmp[:, 6:], y[:, :12]) / (12 * 288 * 288)
    loss_num = loss.detach().cpu().numpy()
    pbar.set_description('Test MSE Loss: ' + str(loss_num / batch_size))
    test_l_sum += loss_num
    n += batch_size
    loss1 = mae_loss(tmp[:, 6:], y[:, :12]) / (12 * 288 * 288)
    loss1 = loss1.detach().cpu().numpy()
    test_l_sum1 += loss1

    img_gt = to_np(y)
    new_img = to_np(tmp[:, 6:])
    for i in range(x.shape[0]):
        for j in range(12):
            a, b = img_gt[i, j], new_img[i, j]
            csi_result = csi(a, b)
            hss_result = hss(a, b)
            for t in range(5):
                CSI[t].append(csi_result[t])
                HSS[t].append(hss_result[t])
test_loss = test_l_sum / n
mae = test_l_sum1 / n

for i in range(5):
    CSI[i] = np.array(CSI[i]).mean()
    HSS[i] = np.array(HSS[i]).mean()


threshold = [0.5, 2, 5, 10, 30]
print('CSI:')
for i in range(5):
    print('r >=', threshold[i], ':', CSI[i], end=' ')
print()
print('HSS:')
for i in range(5):
    print('r >=', threshold[i], ':', HSS[i], end=' ')
print()

print('MSE:', test_loss, 'MAE:', mae)

