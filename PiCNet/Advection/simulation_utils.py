import torch
torch.manual_seed(123)
import numpy as np
import os
from learning_utils import calc_grad
import math
import time
from torch import vmap
import torch.nn.functional as F

# device = torch.device("cuda:2")
# real = torch.float32

# generate grid coordinates
def gen_grid(batch_size, width, height, device):
    img_n_grid_x = width  # 4
    img_n_grid_y = height  # 4
    img_dx = 1./img_n_grid_y  # 1/4
    c_x, c_y = torch.meshgrid(torch.arange(img_n_grid_x), torch.arange(img_n_grid_y), indexing = "ij")
    # 得到4*4长度的坐标c_x,c_y，目前是全0
    img_x = img_dx * (torch.cat((c_x[..., None], c_y[..., None]), axis = 2) + 0.5).to(device) # grid center locations
    # c_x[..., None]增加了1维到4*4*1，cat到4*4*2，数值上变成0.5/4
    img_x = torch.unsqueeze(img_x, 0)
    img_x = img_x.repeat(batch_size, 1, 1, 1)
    return img_x


def RK1(pos, u, dt):
    return pos + dt * u

def RK2(pos, u, dt):
    p_mid = pos + 0.5 * dt * u
    return pos + dt * sample_grid_batched(u, p_mid)

def RK3(pos, u, dt):
    u1 = u
    # print('pos shape:', pos.shape)
    # print('u1 shape:', u1.shape)
    p1 = pos + 0.5 * dt * u1
    u2 = sample_grid_batched(u, p1)
    p2 = pos + 0.75 * dt * u2
    u3 = sample_grid_batched(u, p2)
    return pos + dt * (2/9 * u1 + 1/3 * u2 + 4/9 * u3)

def advect_quantity_batched(quantity, u, x, dt, boundary):
    return advect_quantity_batched_BFECC(quantity, u, x, dt, boundary)

# pos: [num_queries, 2]
# if a backtraced position is out-of-bound, project it to the interior
def project_to_inside(pos, boundary):
    if boundary is None: # if no boundary then do nothing
        return pos
    sdf, sdf_normal, _ = boundary
    W, H = sdf.shape
    dx = 1./H
    pos_grid = (pos / dx).floor().long()
    pos_grid_x = pos_grid[...,0]
    pos_grid_y = pos_grid[...,1]
    pos_grid_x = torch.clamp(pos_grid_x, 0, W-1)
    pos_grid_y = torch.clamp(pos_grid_y, 0, H-1)
    sd_at_pos = sdf[pos_grid_x, pos_grid_y][...,None] # [num_queries, 1]
    sd_normal_at_pos = sdf_normal[pos_grid_x, pos_grid_y] # [num_queries, 2]
    OUT = (sd_at_pos >= -boundary[2]).squeeze(-1) # [num_queries]
    OUT_pos = pos[OUT] #[num_out_queries, 2]
    OUT_pos_fixed = OUT_pos - (sd_at_pos[OUT]+boundary[2]) * dx * sd_normal_at_pos[OUT] # remember to multiply by dx
    pos[OUT] = OUT_pos_fixed
    return pos


def index_take_2D(source, index_x, index_y):
    W, H, Channel = source.shape
    W_, H_ = index_x.shape
    index_flattened_x = index_x.flatten()
    index_flattened_y = index_y.flatten()
    sampled = source[index_flattened_x, index_flattened_y].view((W_, H_, Channel))
    return sampled

index_take_batched = vmap(index_take_2D)

# clipping used for MacCormack and BFECC
def MacCormack_clip(advected_quantity, quantity, u, x, dt, boundary):
    batch, W, H, _ = u.shape
    prev_pos = RK3(x, u, -1. * dt) # [batch, W, H, 2]
    prev_pos = project_to_inside(prev_pos.view((-1, 2)), boundary).view(prev_pos.shape)
    dx = 1./H
    pos_grid = (prev_pos / dx - 0.5).floor().long()
    pos_grid_x = torch.clamp(pos_grid[..., 0], 0, W-2)
    pos_grid_y = torch.clamp(pos_grid[..., 1], 0, H-2)
    pos_grid_x_plus = pos_grid_x + 1
    pos_grid_y_plus = pos_grid_y + 1
    # print('pos_grid_x shape:', pos_grid_x.shape)
    # print('pos_grid_x_plus shape:', pos_grid_x_plus.shape)
    # print('pos_grid_y shape:', pos_grid_y.shape)
    # print('pos_grid_y_plus shape:', pos_grid_y_plus.shape)
    BL = index_take_batched(quantity, pos_grid_x, pos_grid_y)
    # print('quantity shape:', quantity.shape)
    # print('BL shape:', BL.shape)
    BR = index_take_batched(quantity, pos_grid_x_plus, pos_grid_y)
    TR = index_take_batched(quantity, pos_grid_x_plus, pos_grid_y_plus)
    TL = index_take_batched(quantity, pos_grid_x, pos_grid_y_plus)
    stacked = torch.stack((BL, BR, TR, TL), dim = 0)
    maxed = torch.max(stacked, dim = 0).values # [batch, W, H, 3]
    mined = torch.min(stacked, dim = 0).values # [batch, W, H, 3]
    _advected_quantity = torch.clamp(advected_quantity, mined, maxed)
    return _advected_quantity

# SL
def advect_quantity_batched_SL(quantity, u, x, dt, boundary):
    prev_pos = RK3(x, u, -1. * dt) # [batch, W, H, 2]
    prev_pos = project_to_inside(prev_pos.view((-1, 2)), boundary).view(prev_pos.shape)
    # print('prev_pos shape:', prev_pos.shape)
    # print('quantity shape:', quantity.shape)
    new_quantity = sample_grid_batched(quantity, prev_pos)
    return new_quantity

# BFECC Back and Forth Error Compensation and Correction来回误差补偿和校正
def advect_quantity_batched_BFECC(quantity, u, x, dt, boundary):
    quantity1 = advect_quantity_batched_SL(quantity, u, x, dt, boundary)  # 半拉格朗日积分
    quantity2 = advect_quantity_batched_SL(quantity1, u, x, -1.*dt, boundary)
    new_quantity = advect_quantity_batched_SL(quantity + 0.5 * (quantity-quantity2), u, x, dt, boundary)
    new_quantity = MacCormack_clip(new_quantity, quantity, u, x, dt, boundary)
    return new_quantity

# MacCormack 一种解可压缩流体流动问题的二步二阶差分格式
def advect_quantity_batched_MacCormack(quantity, u, x, dt, boundary):
    quantity1 = advect_quantity_batched_SL(quantity, u, x, dt, boundary)
    quantity2 = advect_quantity_batched_SL(quantity1, u, x, -1.*dt, boundary)
    new_quantity = quantity1 + 0.5 * (quantity - quantity2)
    new_quantity = MacCormack_clip(new_quantity, quantity, u, x, dt, boundary)
    return new_quantity

# data = [batch, X, Y, n_channel]
# pos = [batch, X, Y, 2]
def sample_grid_batched(data, pos):
    data_ = data.permute([0, 3, 2, 1])
    pos_ = pos.clone().permute([0, 2, 1, 3])
    pos_ = (pos_ - 0.5) * 2
    F_sample_grid = F.grid_sample(data_, pos_, padding_mode = 'border', align_corners = False, mode = "bilinear")
    F_sample_grid = F_sample_grid.permute([0, 3, 2, 1])
    return F_sample_grid

def sample_grid_batched1(data, pos):
    if data.shape[2] == 288 and data.shape[3] == 288:
        data_ = data
    else:
        data_ = data.permute([0, 3, 2, 1])
    pos_ = pos.clone().permute([0, 2, 1, 3])
    pos_ = (pos_ - 0.5) * 2
    F_sample_grid = F.grid_sample(data_, pos_, padding_mode = 'border', align_corners = False, mode = "bilinear")
    F_sample_grid = F_sample_grid.permute([0, 3, 2, 1])
    return F_sample_grid

# pos: [num_query, 2] or [batch, num_query, 2]
# vel: [batch, num_query, 2]
# mode: 0 for image, 1 for vort
def boundary_treatment(pos, vel, boundary, mode = 0):
    vel_after = vel.clone()
    batch, num_query, _ = vel.shape
    sdf = boundary[0] # [W, H]
    sdf_normal = boundary[1]
    if mode == 0:
        score = torch.clamp((sdf / -15.), min = 0.).flatten()
        inside_band = (score < 1.).squeeze(-1).flatten()
        score = score[None, ..., None]
        vel_after[:, inside_band, :] = score[:, inside_band, :] * vel[:, inside_band, :]
    else:
        W, H = sdf.shape
        dx = 1./H
        pos_grid = (pos / dx).floor().long()
        pos_grid_x = pos_grid[...,0]
        pos_grid_y = pos_grid[...,1]
        pos_grid_x = torch.clamp(pos_grid_x, 0, W-1)
        pos_grid_y = torch.clamp(pos_grid_y, 0, H-1)
        sd = sdf[pos_grid_x, pos_grid_y][...,None]
        sd_normal = sdf_normal[pos_grid_x, pos_grid_y]
        score = torch.clamp((sd / -75.), min = 0.)
        inside_band = (score < 1.).squeeze(-1)
        vel_normal = torch.einsum('bij,bij->bi', vel, sd_normal)[...,None] * sd_normal
        vel_tang = vel - vel_normal
        tang_at_boundary = 0.33
        vel_after[inside_band] = ((1.-tang_at_boundary) * score[inside_band] + tang_at_boundary) * vel_tang[inside_band] + score[inside_band] * vel_normal[inside_band]

    return vel_after

# simulate a single step
def simulate_step(img, img_x, vorts_pos, vorts_w, vorts_size, vel_func, dt, boundary):
    batch_size = vorts_pos.shape[0]   # vorts_pos（batch_size,12,2)
    img_x_flattened = img_x.view(-1, 2)  # （288*288，2）img的网格坐标
    if boundary is None:
        img_vel_flattened = vel_func(vorts_size, vorts_w, vorts_pos, img_x_flattened)  # [batch, num_queries, num_vorts, 2]
        img_vel = img_vel_flattened.view((batch_size, img_x.shape[0], img_x.shape[1], -1))  # [batch, h, w, 2*num_vorts]
        new_img = torch.clip(advect_quantity_batched(img, img_vel, img_x, dt, boundary), 0., 1.)
        vorts_vel = vel_func(vorts_size, vorts_w, vorts_pos, vorts_pos)
        new_vorts_pos = RK1(vorts_pos, vorts_vel, dt)  # vorts_pos+V*T
    else:
        OUT = (boundary[0]>=-boundary[2])
        IN = ~OUT
        img_x_flattened = img_x.view(-1, 2)
        IN_flattened = IN.expand(img_x.shape[:-1]).flatten()
        img_vel_flattened = torch.zeros(batch_size, *img_x_flattened.shape).to(device)
        # only the velocity of the IN part will be computed, the rest will be left as 0
        img_vel_flattened[:, IN_flattened] = vel_func(vorts_size, vorts_w, vorts_pos, img_x_flattened[IN_flattened])
        img_vel_flattened = boundary_treatment(img_x_flattened, img_vel_flattened, boundary, mode = 0)
        img_vel = img_vel_flattened.view((batch_size, img_x.shape[0], img_x.shape[1], -1))
        new_img = torch.clip(advect_quantity_batched(img, img_vel, img_x, dt, boundary), 0., 1.)
        new_img[:, OUT] = img[:, OUT] # the image of the OUT part will be left unchanged
        vorts_vel = vel_func(vorts_size, vorts_w, vorts_pos, vorts_pos)
        vorts_vel = boundary_treatment(vorts_pos, vorts_vel, boundary, mode = 1)
        new_vorts_pos = RK1(vorts_pos, vorts_vel, dt)

    return new_img, new_vorts_pos, img_vel, vorts_vel

def simulate_step1(img, img_x, vorts_pos, vorts_w, vorts_size, max_scale, vel_func, dt, boundary):
    batch_size = vorts_pos.shape[0]   # vorts_pos（6,9,2)
    img_x_flattened = img_x.view(-1, 2)  # （288*288，2）img的网格坐标
    if boundary is None:
        img_vel_flattened = vel_func(vorts_size, vorts_w, vorts_pos, img_x_flattened).to(device)  # [batch, num_queries, num_vorts, 2]
        img_vel = img_vel_flattened.view((batch_size, img_x.shape[0], img_x.shape[1], -1))  # [batch, h, w, 2*num_vorts]
        # new_img = torch.clip(advect_quantity_batched(img, img_vel, img_x, dt, boundary), 0., 1.)
        # print('img max:', img.max())
        new_img = advect_quantity_batched(img, img_vel, img_x, dt, boundary)
        new_img = torch.clip(new_img, 0., float(max_scale))
        vorts_vel = vel_func(vorts_size, vorts_w, vorts_pos, vorts_pos)
        new_vorts_pos = RK1(vorts_pos, vorts_vel, dt)  # vorts_pos+V*T
    else:
        OUT = (boundary[0]>=-boundary[2])
        IN = ~OUT
        img_x_flattened = img_x.view(-1, 2)
        IN_flattened = IN.expand(img_x.shape[:-1]).flatten()
        img_vel_flattened = torch.zeros(batch_size, *img_x_flattened.shape).to(device)
        # only the velocity of the IN part will be computed, the rest will be left as 0
        img_vel_flattened[:, IN_flattened] = vel_func(vorts_size, vorts_w, vorts_pos, img_x_flattened[IN_flattened])
        img_vel_flattened = boundary_treatment(img_x_flattened, img_vel_flattened, boundary, mode = 0)
        img_vel = img_vel_flattened.view((batch_size, img_x.shape[0], img_x.shape[1], -1))
        new_img = torch.clip(advect_quantity_batched(img, img_vel, img_x, dt, boundary), 0., 1.)
        new_img[:, OUT] = img[:, OUT] # the image of the OUT part will be left unchanged
        vorts_vel = vel_func(vorts_size, vorts_w, vorts_pos, vorts_pos)
        vorts_vel = boundary_treatment(vorts_pos, vorts_vel, boundary, mode = 1)
        new_vorts_pos = RK1(vorts_pos, vorts_vel, dt)

    return new_img, new_vorts_pos, img_vel, vorts_vel

# simulate in batches
# img: the initial image
# img_x: the grid coordinates (meshgrid)
# vorts_pos: init vortex positions
# vorts_w: vorticity
# vorts_size: size
# num_steps: how many steps to simulate
# vel_func: how to compute velocity from vorticity
def simulate(img, img_x, vorts_pos, vorts_w, vorts_size, num_steps, vel_func, boundary = None, dt = 0.01):
    imgs = []
    vorts_poss = []
    img_vels = []
    vorts_vels = []
    for i in range(num_steps):  # num_steps=num_sim=2
        img, vorts_pos, img_vel, vorts_vel = simulate_step(img, img_x, vorts_pos, vorts_w, vorts_size, vel_func, dt, boundary = boundary)
        imgs.append(img.clone())
        vorts_poss.append(vorts_pos.clone())
        img_vels.append(img_vel)
        vorts_vels.append(vorts_vel)

    return imgs, vorts_poss, img_vels, vorts_vels

def simulate1(img, img_x, vorts_pos, vorts_w, vorts_size, num_steps, max_scale, vel_func, boundary = None, dt = 0.01):
    imgs = []
    vorts_poss = []
    img_vels = []
    vorts_vels = []
    for i in range(num_steps):  # num_steps=num_sim
        img, vorts_pos, img_vel, vorts_vel = simulate_step1(img, img_x, vorts_pos, vorts_w, vorts_size, max_scale, vel_func, dt, boundary = boundary)
        imgs.append(img.clone())
        vorts_poss.append(vorts_pos.clone())
        img_vels.append(img_vel)
        vorts_vels.append(vorts_vel)

    return imgs, vorts_poss, img_vels, vorts_vels


def simulate2(img, img_x, x_max, vort_max, vort_min, img_vort_net, vort_vel_net, img_vort_net_load, vort_vel_net_load, dt = 0.01):
    img_vort_net.load_state_dict(torch.load(img_vort_net_load))
    vort_vel_net.load_state_dict(torch.load(vort_vel_net_load))
    vort = img_vort_net(img)
    vort = vort * x_max  # 反归一化
    vort = (vort - vort_min) / (vort_max - vort_min)  # 归一化
    vel = vort_vel_net(vort)
    vel = vel * (vort_max - vort_min) + vort_min  # 反归一化

    # batchsize > 1的时候：
    if img.shape[0] > 1:
        t = img[:, -1:].squeeze()
        t = t.unsqueeze(-1)
        imgs = t.repeat(1, 1, 1, 3)
    # batchsize = 1的时候：
    else:
        t = img[:, -1:].squeeze()
        t = t.unsqueeze(0)
        t = t.unsqueeze(-1)
        imgs = t.repeat(1, 1, 1, 3)
    new_img = advect_quantity_batched(imgs, vel[:, -1], img_x, dt, boundary=None).cuda()

    return new_img

def simulate3(img, img_x, x_max, vort_max, vort_min, img_vort_net, vort_vel_net, dt = 0.01):
    # img_vort_net.load_state_dict(torch.load('model1_1_100.pt'))
    # vort_vel_net.load_state_dict(torch.load('model1_0_200.pt'))
    vort = img_vort_net(img)
    vort = vort * x_max  # 反归一化
    vort = (vort - vort_min) / (vort_max - vort_min)  # 归一化
    vel = vort_vel_net(vort)
    vel = vel * (vort_max - vort_min) + vort_min  # 反归一化

    # batchsize > 1的时候：
    # t = x[:, -1:].squeeze()
    # t = t.unsqueeze(-1)
    # imgs = t.repeat(1, 1, 1, 3)

    # batchsize = 1的时候：
    x = img
    for i in range(12):

        t = x[:, -1:].squeeze()
        # print(t.shape)
        # t = t.unsqueeze(0)
        t = t.unsqueeze(-1)
        # print(t.shape)
        imgs = t.repeat(1, 1, 1, 3)

        new_img = advect_quantity_batched(imgs, vel[:, 5 + i], img_x, dt, boundary=None).cuda()

        # print('new_img shape:', new_img.shape)

        new_img = new_img[:, :, :, 0]
        new_img = new_img.squeeze()
        # batchsize>1：
        if x.shape[0] > 1:
            new_img = new_img.unsqueeze(1)
        # batchsize=1：
        else:
            new_img = new_img.unsqueeze(0)
            new_img = new_img.unsqueeze(0)
        # print(new_img.shape)
        img = torch.cat([img, new_img], dim=1)
        x = img[:, -6:]

    return img