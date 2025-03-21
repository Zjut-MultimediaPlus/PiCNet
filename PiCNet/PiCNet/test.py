from dataloder import *
from _2PNet import *
from tqdm import tqdm
import sys
sys.path.append('..')
import time
import matplotlib
matplotlib.use('Agg')

# -----------------------------------------------
cuda_idx = 0
file_idx = 3
device = torch.device('cuda:' + str(cuda_idx))
torch.cuda.set_device(device)
x_test_data_dir = '/root/data/test'
z_test_data_dir ='/opt/data/private/data/KNMI/img_vel/6-12img-BAE/test'

data_format = 'npy'
test_data_dir = [x_test_data_dir, z_test_data_dir]
# -----------------------------------------------
test_dataset = K_ZDataset(test_data_dir, None, data_format)
epoch_size, batch_size = 200, 8
in_channel = 30
out_channel = 12
loga = 7
net = Net(loga, in_channel, out_channel).cuda()
min_test_loss, out_count = 1e10, 0
min_mae = 1e10
# MSE_criterion = BMAEloss().cuda()
MSE_criterion = nn.MSELoss()
net.load_state_dict(torch.load('11model0_1_100.pt'))
relu = nn.ReLU()
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
# -----------------------------------------------

npy_name = '1450.npy'
for step, (x, y, z) in enumerate(test_dataloader):
    # print(npy_name)
    z = default_loader(z_test_data_dir+'/'+npy_name)
    data = default_loader(x_test_data_dir+'/'+npy_name)
    x = data[:6] * 4783 / 100 * 12
    y = data[6:] * 4783 / 100 * 12
    z = z * 4783 / 100 * 12
    x = np.expand_dims(x, axis=0)  # [1, t, h, w]，
    y = np.expand_dims(y, axis=0)  # [1, t, h, w]，
    z = np.expand_dims(z, axis=0)  # [1, t, h, w]，
    z = torch.tensor(z)
    x = torch.tensor(x)
    y = torch.tensor(y)
    z = z.cuda()
    x = x.cuda()  # b*6*288*288
    y = y.cuda()  # b*12*288*288
    z = log(z, loga)
    x = log(x, loga)
    start_time = time.time()
    y_hat = net(z, x)
    y_hat = relu(y_hat)
    end_time = time.time()
    y = to_np(y)
    y_hat = to_np(y_hat)
    z = to_np(z)
    np.save("PICNET_result.npy", y_hat[0,3,:,:])
    index = 0



    for i in range(12):
        show(draw_color_single(y[index, i]))
        t = y_hat[index, i]
        t2 = y[index, i]
        t[t2 > 30] = 30
        show(draw_color_single(t))
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        axs[0].imshow(y[index][i]), axs[0].set_title('y')
        axs[1].imshow(y_hat[index][i]), axs[1].set_title('y-hat')
        axs[2].imshow(z[index][i]), axs[2].set_title('z')
        plt.show()

    print(y_hat.shape)
    print(end_time - start_time)
    print('FPS:', 1 / (end_time - start_time) * (9 * batch_size))

    mse, mae = [], []
    for i in range(y.shape[0]):
        for j in range(12):
            a, b = y[i, j], y_hat[i, j]
            mse.append(B_mse(a, b))
            mae.append(B_mae(a, b))
            csi_result = csi(a, b)
            hss_result = hss(a, b)
    print(mse)
    print(mae)
    mse = np.array(mse).mean()
    mae = np.array(mae).mean()

    print('MSE:', mse, 'MAE:', mae)
    if step == 0:
        break

net.eval()
pbar = tqdm(test_dataloader)
CSI, HSS, mse, mae = [], [], [], []
p_value_ls = []

for i in range(5):
    CSI.append([])
    HSS.append([])
    GSS.append([])
for step, (x, y, z) in enumerate(pbar):
    z = z.cuda()
    x = x.cuda()
    y = y.cuda()
    z = log(z, loga)
    x = log(x, loga)
    y_hat = net(z, x)
    y_hat = relu(y_hat)

    loss = MSE_criterion(y_hat, y)
    loss_num = loss.detach().cpu().numpy()
    pbar.set_description('Test MSE Loss: ' + str(loss_num / batch_size))


for i in range(5):
    CSI[i] = np.array(CSI[i]).mean()
    HSS[i] = np.array(HSS[i]).mean()
mse = np.array(mse).mean()
mae = np.array(mae).mean()

threshold = [0.5, 2, 5, 10, 30]
print('CSI:')
for i in range(5):
    print('r >=', threshold[i], ':', CSI[i], end=' ')
print()
print('HSS:')
for i in range(5):
    print('r >=', threshold[i], ':', HSS[i], end=' ')
print()


print('MSE:', mse, 'MAE:', mae)
print(CSI)
print(HSS)
# print(GSS)