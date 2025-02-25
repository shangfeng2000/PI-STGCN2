import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx


def ade(predAll,targetAll,count_):
    predAll=np.array(predAll)
    targetAll=np.array(targetAll)
    All = len(predAll)
    if (All == 0):
        return 0
    sum_all = 0 
    for s in range(All):
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1) #交换数组的轴
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            for t in range(T):
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N*T)
    #T是时间长度,N是数据量，ALL是人数量
    return sum_all/All

def compute_ADE(pred_arr, gt_arr):
    ade = 0.0
    ade2 = 0.0
    for pred, gt in zip(pred_arr, gt_arr):
        diff = pred - np.expand_dims(gt, axis=0)        # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)            # samples x frames
        ade2_inside = 0
        for f in range(dist.shape[1]):
            min_sample = np.min(dist[:,f])
            ade2_inside+= min_sample
        ade2_inside = ade2_inside/dist.shape[1]
        dist = dist.mean(axis=0)                      # samples
        ade += dist.mean(axis=0)                         # (1, )
        ade2 += ade2_inside
    ade /= len(pred_arr)
    ade2 /= len(pred_arr)
    return ade

def fde(predAll,targetAll,count_):
    All = len(predAll)
    sum_all = 0 
    for s in range(All):
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            for t in range(T-1,T):
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N)

    return sum_all/All

def compute_FDE(pred_arr, gt_arr):
    fde = 0.0
    for pred, gt in zip(pred_arr, gt_arr):
        diff = pred - np.expand_dims(gt, axis=0)        # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)            # samples x frames
        dist = dist[..., -1]                            # samples
        fde += dist.mean(axis=0)                         # (1, )
    fde /= len(pred_arr)
    return fde
def seq_to_nodes(seq_):
    max_nodes = seq_.shape[1] #number of pedestrians in the graph
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]
    
    V = np.zeros((seq_len,max_nodes,2))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_[h]
            
    return V.squeeze()

def position_to_velocity(V_obs,V_pred,Velocity_obs,T_step):
    V = np.ones_like(V_pred)
    previous_position = V_obs[-1, :, :]
    previous_velocity = Velocity_obs[-1,:,:]
    for i in range(0,V.shape[0]):
        next_position = V_pred[i,:,:]
        next_velocity = 2*(next_position-previous_position)/T_step-previous_velocity
        V[i,:,:] = next_velocity
        previous_position = next_position
        previous_velocity = next_velocity
    return V.squeeze()

def nodes_rel_to_nodes_abs(nodes,init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s,ped,:] = np.sum(nodes[:s+1,ped,:],axis=0) + init_node[ped,:]

    return nodes_.squeeze()

def closer_to_zero(current,new_v):
    dec =  min([(abs(current),current),(abs(new_v),new_v)])[1]
    if dec != current:
        return True
    else: 
        return False
        
def bivariate_loss(V_pred,V_trgt):
    #mux, muy, sx, sy, corr
    #assert V_pred.shape == V_trgt.shape
    normx = V_trgt[:,:,0]- V_pred[:,:,0]
    normy = V_trgt[:,:,1]- V_pred[:,:,1]

    sx = torch.exp(V_pred[:,:,2]) #sx
    sy = torch.exp(V_pred[:,:,3]) #sy
    corr = torch.tanh(V_pred[:,:,4]) #corr
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon)) #用于将张量 result 中的元素限制在一个指定的范围内
    result = torch.mean(result)
    nan = float('nan')
    if math.isnan(result):
        print("ICDEc测试1：",denom.size(),negRho.size())
        torch.save(V_pred, 'H:/GRK/ICDE/V_pred.pt')
        torch.save(V_trgt, 'H:/GRK/ICDE/V_trgt.pt')
    return result

def construct_loss(V_pred,V_trgt):
    V_pred_mean = V_pred[:, :, 0:2]
    diff = V_pred_mean - V_trgt  # 形状为 [T, N, 2]
    sq_loss = diff.pow(2)  # 对每个元素进行平方，形状为 [T, N, 2]
    sq_loss = sq_loss.sum(dim=2)  # 在第3维（x和y）上求和，得到形状 [T, N]
    con_loss = sq_loss[-1,:]  # 求和得到标量
    con_loss = con_loss.sum()  # 求均值
    return con_loss


def velocity_bound_loss(Velocity_pred,Velocity_trgt):
    #V_pre_x = torch.log((Velocity_pred[:,:,0]-Velocity_trgt[:,:,0])**2)
    #V_pre_y = torch.log((Velocity_pred[:,:,1]-Velocity_trgt[:,:,1])**2)
    V_pre_x = (Velocity_pred[:, :, 0] - Velocity_trgt[:, :, 0]) ** 2
    V_pre_y = (Velocity_pred[:, :, 1] - Velocity_trgt[:, :, 1]) ** 2
    V_pre_loss = V_pre_x+V_pre_y
    V_pre_loss = torch.mean(V_pre_loss)
    return V_pre_loss

def velocity_physic_loss(V_pred,Velocity_pred,Accelarete_pred):
    #遍历时间序列维度
    result = torch.zeros(Accelarete_pred.shape[1],Accelarete_pred.shape[2]).cuda()
    for t in range(Accelarete_pred.shape[0]):
        current_p = V_pred[t, :, 0:2]
        next_p = V_pred[t + 1, :, 0:2]
        current_v = Velocity_pred[t, :, 0:2]
        next_v = Velocity_pred[t + 1, :, 0:2]
        current_a = Accelarete_pred[t, :, :]
        result += (next_v**2-current_v**2-2*current_a*(next_p-current_p))**2
    result = torch.mean(result)
    return result


def poly_physic_loss_new(V_pred, Velocity_pred, V_poly, delta_t=0.4):
    # V_pred [T, N, h] 预测的行人轨迹 (x, y 坐标)
    # Velocity_pred [T, N, h] 预测的速度 (x, y) 可用于扩展
    # V_poly [N, h, order] 运动方程的多项式系数
    # delta_t 时间间隔 (通常为 0.4)
    #print("KDD测试：",V_pred.shape,V_poly.shape)
    T, N, h = V_pred.shape
    order = V_poly.shape[2]  # 多项式的阶数
    # 存储所有相邻时间间隔点的 MSE
    total_loss = 0.0
    num_pairs = 0  # 用于计算平均损失
    # 遍历所有相邻的时间步对 (t, t+1)
    for t in range(T - 1):  # 0 到 T-2 之间
        # 获取实际位置 (t 和 t+1) 的 x 和 y 坐标
        actual_positions_t = V_pred[t, :, :2]  # (N, 2)
        actual_positions_t1 = V_pred[t + 1, :, :2]  # (N, 2)
        # 计算基于多项式的预测位置
        predicted_positions_t = torch.zeros(N, 2).cuda()  # (N, 2)
        predicted_positions_t1 = torch.zeros(N, 2).cuda()  # (N, 2)
        # 计算时间点
        t_current = t * delta_t  # 当前时刻的时间
        t_next = (t + 1) * delta_t  # 下一时刻的时间
        t_current_powers = torch.pow(t_current,
                                     torch.arange(1, order + 1, dtype=torch.float32).unsqueeze(0)).cuda()  # (2, order)
        t_next_powers = torch.pow(t_next, torch.arange(1, order + 1, dtype=torch.float32).unsqueeze(0)).cuda()  # (2, order)
        x_poly = V_poly[:, 0, :]  # (N, order)，x方向的多项式系数
        y_poly = V_poly[:, 1, :]  # (N, order)，y方向的多项式系数
        x_pred_t = torch.matmul(torch.unsqueeze(t_current_powers[0, :], 0), x_poly.T).squeeze()  # (N,)
        y_pred_t = torch.matmul(torch.unsqueeze(t_current_powers[0, :], 0), y_poly.T).squeeze()  # (N,)
        x_pred_t1 = torch.matmul(torch.unsqueeze(t_next_powers[0, :], 0), x_poly.T).squeeze()  # (N,)
        y_pred_t1 = torch.matmul(torch.unsqueeze(t_next_powers[0, :], 0), y_poly.T).squeeze()  # (N,)
        # 更新预测位置
        predicted_positions_t[:, 0] = x_pred_t
        predicted_positions_t[:, 1] = y_pred_t
        predicted_positions_t1[:, 0] = x_pred_t1
        predicted_positions_t1[:, 1] = y_pred_t1
        predicted_positions_t1[:, 0] -= predicted_positions_t[:, 0]
        predicted_positions_t1[:, 1] -= predicted_positions_t[:, 1]
        # 计算实际距离
        physical_distances = torch.norm(actual_positions_t1 - predicted_positions_t1, dim=1)  # (N,)
        # 计算损失（均方误差）
        loss = torch.mean(physical_distances)
        # 累加损失
        total_loss += loss
        num_pairs += 1
    # 计算平均损失
    avg_loss = total_loss / num_pairs
    return avg_loss


def poly_physic_loss(V_pred,Velocity_pred,V_poly,delta_t=0.4):
    #遍历时间序列维度
    result = torch.zeros(V_poly.shape[1],2).cuda()
    order = V_poly.shape[2]/2
    for t in range(V_poly.shape[0]):
        current_p = V_pred[t, :, 0:2]
        next_p = V_pred[t + 1, :, 0:2]
        pinn_dist_x = V_poly[t, :,0]*(delta_t**4)+V_poly[t, :,1]*(delta_t**3)+V_poly[t, :,2]*(delta_t**2)+V_poly[t, :,3]*(delta_t)+V_poly[t, :,4]
        pinn_dist_y = V_poly[t, :, 5] * (delta_t ** 4) + V_poly[t, :, 6] * (delta_t ** 3) + V_poly[t, :, 7] * (delta_t ** 2) + V_poly[t, :, 8] * (delta_t) + V_poly[t, :, 9]
        pinn_dist = torch.stack((pinn_dist_x, pinn_dist_y), dim=-1)
        result +=(next_p-pinn_dist)**2
    result = torch.mean(result)
    return result
