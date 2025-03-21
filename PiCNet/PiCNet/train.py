from dataloder import *
from torch import optim
import matplotlib
from _2PNet import *
from tqdm import tqdm
import time

# -----------------------------------------------

cuda_idx = 0
file_idx = 3
device = torch.device('cuda:' + str(cuda_idx))
torch.cuda.set_device(device)

# -----------------------------------------------
x_train_data_dir, x_val_data_dir = '/root/data/train', '/root/data/val'
z_train_data_dir, z_val_data_dir ='/root/data/6-12img-BAE/train' , '/root/data/6-12img-BAE/val'

train_data_dir = [x_train_data_dir, z_train_data_dir]
val_data_dir = [x_val_data_dir, z_val_data_dir]
data_format = 'npy'
train_dataset = K_ZDataset(train_data_dir, None, data_format)
val_dataset = K_ZDataset(val_data_dir, None, data_format)


epoch_size, batch_size = 100, 16
in_channel = 30
out_channel = 12
min_test_loss, out_count = 1e10, 0
min_mae = 1e10
loga = 7

net = Net(loga, in_channel, out_channel).cuda()
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
# -----------------------------------------------

opt = optim.Adam(net.parameters(), lr=1e-4)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=10, verbose=True)
MSE_criterion = BMAEloss().cuda()

relu = nn.ReLU()
# -----------------------------------------------
train_epoch_loss_array = [0]
csi_array = [[0],[0],[0],[0],[0]]
hss_array = [[0],[0],[0],[0],[0]]


for epoch in range(1, epoch_size + 1):
    if file_idx == -1:
        f = open('nos_log_' + str(cuda_idx) + '.txt', 'a+')
    else:
        f = open('nos_log_' + str(file_idx) + '.txt', 'a+')
    train_l_sum, test_l_sum, n = 0.0, 0.0, 0
    net.train()

    pbar = tqdm(train_dataloader)

    for step, (x, y, z) in enumerate(pbar):
        z = z.cuda()
        x = x.cuda()
        y = y.cuda()
        z = log(z, loga)
        x = log(x, loga)
        y_hat = net(z, x)
        y_hat = relu(y_hat)
        loss = MSE_criterion(y_hat, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_num = loss.detach().cpu().numpy()
        pbar.set_description('Train MSE Loss: ' + str(loss_num / batch_size))
        train_l_sum += loss_num
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
        for step, (x, y, z) in enumerate(pbar):
            z = z.cuda()
            x = x.cuda()
            z = log(z, loga)
            x = log(x, loga)
            y = y.cuda()
            y_hat = net(z, x)
            y_hat = relu(y_hat)

            loss = MSE_criterion(y_hat, y)
            loss_num = loss.detach().cpu().numpy()
            test_l_sum += loss_num
            pbar.set_description('Test MSE Loss: ' + str(loss_num / batch_size))
            n += batch_size

            y = to_np(y)
            y_hat = to_np(y_hat)
            for i in range(y.shape[0]):
                for j in range(12):
                    a, b = y[i, j], y_hat[i, j]
                    mse.append(B_mse(a, b))
                    mae.append(B_mae(a, b))
                    csi_result = csi(a, b)
                    hss_result = hss(a, b)
                    for t in range(5):
                        CSI[t].append(csi_result[t])
                        HSS[t].append(hss_result[t])

        for i in range(5):
            CSI[i] = np.array(CSI[i]).mean()
            HSS[i] = np.array(HSS[i]).mean()
        mse = np.array(mse).mean()
        mae = np.array(mae).mean()

        f.write('Iter: ' + str(epoch) + ' ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n')
        print('Iter:', epoch, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        threshold = [0.5, 2, 5, 10, 30]
        f.write('CSI: ')
        print('CSI: ')
        for i in range(5):
            f.write('r >= ' + str(threshold[i]) + ' : ' + str(CSI[i]) + ' ')
            csi_array[i].append(CSI[i])
            print('r >=', threshold[i], ':', CSI[i], end=' ')

        f.write('\n')
        print()
        f.write('HSS: ')
        print('HSS:')
        for i in range(5):
            f.write('r >= ' + str(threshold[i]) + ' : ' + str(HSS[i]) + ' ')
            hss_array[i].append(HSS[i])
            print('r >=', threshold[i], ':', HSS[i], end=' ')
        f.write('\n')
        print()

        f.write('MSE: ' + str(mse) + ' MAE:' + str(mae) + '\n')
        print('MSE:', mse, 'MAE:', mae)

        test_loss = test_l_sum / n
        lr_scheduler.step(mae)
        if epoch % 10 == 0 and epoch > 40:
            torch.save(net.state_dict(), 'nos_model' + str(cuda_idx) + '_' + str(file_idx) + '_' + str(epoch) + '.pt')

    f.write('Train loss: ' + str(train_loss) + ' Test loss: ' + str(test_loss) + '\n')
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
plt.savefig("train loss-vel.png")
plt.close()

# fig1, ax1 = plt.subplots(figsize=(12, 8))
plt.plot(np.arange(1, epoch_size + 1), csi_array[0][1:], 'y', label='0.5-2')
plt.plot(np.arange(1, epoch_size + 1), csi_array[1][1:], 'r', label='2-5')
plt.plot(np.arange(1, epoch_size + 1), csi_array[2][1:], color='orangered', label='5-10')
plt.plot(np.arange(1, epoch_size + 1), csi_array[3][1:], color='blueviolet', label='10-30')
plt.plot(np.arange(1, epoch_size + 1), csi_array[4][1:], color='green', label='>=30')
plt.legend()  # 显示图例
plt.xlabel('epochs')
plt.ylabel('CSI')
# plt.show()
plt.savefig("CSI-vel.png")
plt.close()

plt.plot(np.arange(1, epoch_size + 1), hss_array[0][1:], 'y', label='0.5-2')
plt.plot(np.arange(1, epoch_size + 1), hss_array[1][1:], 'r', label='2-5')
plt.plot(np.arange(1, epoch_size + 1), hss_array[2][1:], color='orangered', label='5-10')
plt.plot(np.arange(1, epoch_size + 1), hss_array[3][1:], color='blueviolet', label='10-30')
plt.plot(np.arange(1, epoch_size + 1), hss_array[4][1:], color='green', label='>=30')
plt.legend()  # 显示图例
plt.xlabel('epochs')
plt.ylabel('HSS')
# plt.show()
plt.savefig("HSS-vel.png")


