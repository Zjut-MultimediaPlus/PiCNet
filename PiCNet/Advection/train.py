from torch import nn, optim
import matplotlib
from dataloder import *
from AS import AD
from tqdm import tqdm
import sys
from tool import *
from simulation_utils import *
sys.path.append('..')
# -----------------------------------------------
cuda_idx = 0
file_idx = 1
device = torch.device('cuda:' + str(cuda_idx))
torch.cuda.set_device(device)
# -----------------------------------------------
x_train_data_dir, x_val_data_dir = '/data/train', '/data/val'
y_train_data_dir, y_val_data_dir ='/data/train', '/data/val'
train_data_dir = [x_train_data_dir, y_train_data_dir]
val_data_dir = [x_val_data_dir, y_val_data_dir]
data_format = 'npy'
train_dataset = img_sobel(train_data_dir, None, data_format)
val_dataset = img_sobel(val_data_dir, None, data_format)

epoch_size, batch_size = 100, 16
in_channel = 6
out_channel = 12
setup_seed(0)  # 固定
net = AD(out_channel, out_channel).cuda()
min_test_loss, out_count = 1e10, 0
min_mae = 1e10
relu = nn.ReLU()

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
# -----------------------------------------------
opt = optim.Adam(net.parameters(), lr=1e-3)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=5, verbose=True)
MSE_criterion = BMAEloss().cuda()

def MSE(y_hat, y):
    sub = y_hat - y
    return np.sum(sub * sub)

# -----------------------------------------------
train_epoch_loss_array = [0]
hid_loss_array, img_loss_array = [0], [0]
val_epoch_loss_array = [0]
# csi_array = [[0],[0],[0],[0],[0]]
# hss_array = [[0],[0],[0],[0],[0]]

for epoch in range(1, epoch_size + 1):
    if file_idx == -1:
        f = open('12-12log_' + str(cuda_idx) + '.txt', 'a+')
    else:
        f = open('12-12log_' + str(file_idx) + '.txt', 'a+')
    train_l_sum, test_l_sum, n = 0.0, 0.0, 0
    hid_l_sum, img_l_sum = 0.0, 0.0
    net.train()
    pbar = tqdm(train_dataloader)
    for step, (x, y, npy_name) in enumerate(pbar):
        x = x.cuda()
        y = y.cuda()
        s, y_hat = net(x)  # [b,24,288,288]
        y_hat = y_hat.view(x.shape[0], out_channel, 288, 288, 2)

        tmp = x[:, :6]
        loss1 = 0.0
        for i in range(12):
            t = tmp[:, -1]
            if x.shape[0] == 1:
                t = t.unsqueeze(0)
            if x.shape[0] != batch_size:
                img_x = gen_grid(x.shape[0], 288, 288, device)
            else:
                img_x = gen_grid(batch_size, 288, 288, device)
            t = t.unsqueeze(-1)
            imgs = t.repeat(1, 1, 1, 3)

            u = y_hat[:, i]
            u = torch.stack((u[:, :, :, 1], u[:, :, :, 0] * -1), dim=-1)
            new_img = advect_quantity_batched(imgs, u, img_x, dt=0.01, boundary=None).cuda()
            new_img = new_img[:, :, :, 0]  # (b,288,288)
            tmp_s = s[:, i]
            new_img = new_img.squeeze() + tmp_s
            new_img = relu(new_img)
            if x.shape[0] > 1:
                new_img = new_img.unsqueeze(1)
            else:
                new_img = new_img.unsqueeze(0)
                new_img = new_img.unsqueeze(0)
            tmp = torch.cat([tmp, new_img], dim=1)

        img_loss = MSE_criterion(tmp[:, 6:], y[:, :12]) / (out_channel * 288 * 288)
        loss = img_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_num = loss.detach().cpu().numpy()
        pbar.set_description('Train MSE Loss: ' + str(loss_num / batch_size))
        train_l_sum += loss_num
        img_l_sum += img_loss.detach().cpu().numpy()
        n += batch_size
    train_loss = train_l_sum / n
    train_epoch_loss_array.append(train_loss)
    n = 0
    net.eval()

    with torch.no_grad():
        CSI, HSS, mse, mae = [], [], [], []
        for i in range(5):
            CSI.append([])
            HSS.append([])

        pbar = tqdm(val_dataloader)
        for step, (x, y, npy_name) in enumerate(pbar):
            x = x.cuda()
            y = y.cuda()
            s, y_hat = net(x)
            y_hat = y_hat.view(x.shape[0], out_channel, 288, 288, 2)
            tmp = x[:, :6]
            loss1 = 0.0
            for i in range(12):
                t = tmp[:, -1]
                if x.shape[0] == 1:
                    t = t.unsqueeze(0)
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
                    new_img = new_img.unsqueeze(0)
                tmp = torch.cat([tmp, new_img], dim=1)

            img_loss = MSE_criterion(tmp[:, 6:], y[:, :12]) / (out_channel * 288 * 288)
            loss = img_loss
            loss_num = loss.detach().cpu().numpy()
            test_l_sum += loss_num
            pbar.set_description('Test MSE Loss: ' + str(loss_num / batch_size))
            n += batch_size

            img_gt = to_np(y[:, :12])
            new_img = to_np(tmp[:, 6:])
            for i in range(x.shape[0]):
                for j in range(12):
                    a, b = img_gt[i, j], new_img[i, j]
                    mse.append(B_mse(a, b))
                    mae.append(B_mae(a, b))

        mse = np.array(mse).mean()
        mae = np.array(mae).mean()

        f.write('Iter: ' + str(epoch) + ' ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n')
        print('Iter:', epoch, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        f.write('MSE: ' + str(mse) + ' MAE:' + str(mae) + '\n')
        print('MSE:', mse, 'MAE:', mae)

        val_loss = test_l_sum / n
        val_epoch_loss_array.append(val_loss)
        n = 0
        lr_scheduler.step(mae)
        if epoch % 10 == 0:
            torch.save(net.state_dict(), '12-12model' + str(cuda_idx) + '_' + str(file_idx) + '_' + str(epoch) + '.pt')

    f.write('Train loss: ' + str(train_loss) + ' Val loss: ' + str(val_loss) + '\n')
    seg_line = '=======================================================================' + '\n'
    f.write(seg_line)
    f.close()


matplotlib.use('Agg')
fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.plot(np.arange(1, epoch_size + 1), train_epoch_loss_array[1:], 'r')
ax1.set_xlabel('epochs')
ax1.set_ylabel('train loss')
ax1.set_title('Size Esimation Traing loss vs. Training Epoch')
# plt.show()
plt.savefig("train loss.png")
plt.close()

matplotlib.use('Agg')
fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.plot(np.arange(1, epoch_size + 1), val_epoch_loss_array[1:], 'r')
ax2.set_xlabel('epochs')
ax2.set_ylabel('val loss')
ax2.set_title('Size Esimation Traing loss vs. Training Epoch')
# plt.show()
plt.savefig("val loss.png")
plt.close()

matplotlib.use('Agg')
fig3, ax3 = plt.subplots(figsize=(12, 8))
ax3.plot(np.arange(1, epoch_size + 1), hid_loss_array[1:], 'r')
ax3.set_xlabel('epochs')
ax3.set_ylabel('hid loss')
ax3.set_title('Size Esimation Traing loss vs. Training Epoch')
# plt.show()
plt.savefig("hid loss.png")
plt.close()

matplotlib.use('Agg')
fig4, ax4 = plt.subplots(figsize=(12, 8))
ax4.plot(np.arange(1, epoch_size + 1), img_loss_array[1:], 'r')
ax4.set_xlabel('epochs')
ax4.set_ylabel('img loss')
ax4.set_title('Size Esimation Traing loss vs. Training Epoch')
# plt.show()
plt.savefig("img loss.png")
plt.close()
