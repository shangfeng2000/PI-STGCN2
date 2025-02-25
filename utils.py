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
from tqdm import tqdm
import time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.list = list()

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.list.append(val)
# 核函数计算
def anorm(p1,p2):
    NORM = math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
    if NORM ==0:
        return 0
    return 1/(NORM)

#ICDE修改邻接矩阵计算
def anorm2(p1,p2):
    NORM = math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    if NORM ==0:
        return 0
    return NORM


def cross_product(v1, v2):
    """计算二维向量的叉积"""
    return v1[0] * v2[1] - v1[1] * v2[0]


def ray_intersection(P1, P2, Q1, Q2):
    """计算两条射线的交点，如果相交则返回交点坐标，否则返回0"""
    # 计算向量P2 - P1 和 Q2 - Q1
    P2_P1 = np.array([P2[0] - P1[0], P2[1] - P1[1]])
    Q2_Q1 = np.array([Q2[0] - Q1[0], Q2[1] - Q1[1]])

    # 计算矩阵的行列式，判断是否平行
    denom = cross_product(P2_P1, Q2_Q1)
    if denom == 0:
        return 0  # 两条射线平行或共线，不相交

    # 计算交点参数 t1 和 t2
    Q1_P1 = np.array([Q1[0] - P1[0], Q1[1] - P1[1]])

    t1 = cross_product(Q1_P1, Q2_Q1) / denom
    t2 = cross_product(Q1_P1, P2_P1) / denom

    # 如果 t1 >= 0 且 t2 >= 0，射线相交
    if t1 >= 0 and t2 >= 0:
        # 计算交点坐标
        intersection = P1 + t1 * P2_P1
        return tuple(intersection)
    else:
        return 0  # 不相交

def seq_to_graph(seq_,seq_rel,norm_lap_matr = True):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    
    V = np.zeros((seq_len,max_nodes,2))
    A = np.zeros((seq_len,max_nodes,max_nodes))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        step_rel = seq_rel[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_rel[h]
            A[s,h,h] = 1
            for k in range(h+1,len(step_)):
                l2_norm = anorm(step_rel[h],step_rel[k])
                A[s,h,k] = l2_norm
                A[s,k,h] = l2_norm
        if norm_lap_matr: 
            G = nx.from_numpy_array(A[s,:,:])
            A[s,:,:] = nx.normalized_laplacian_matrix(G).toarray() # 标准化拉普拉斯矩阵
    # 邻接矩阵和顶点特征
    return torch.from_numpy(V).type(torch.float),\
           torch.from_numpy(A).type(torch.float)


def seq_to_graph2(seq_, seq_rel, norm_lap_matr=True):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    V = np.zeros((seq_len, max_nodes, 2))
    A = np.zeros((seq_len, max_nodes, max_nodes))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        # 先获取两个相邻的数据点
        if (s == 0):
            cur_step_ = seq_[:, :, s + 1]
            pre_step_ = seq_[:, :, s]
        else:
            cur_step_ = seq_[:, :, s]
            pre_step_ = seq_[:, :, s - 1]
        for h in range(len(step_)):
            V[s, h, :] = step_rel[h]
            A[s, h, h] = 1
            for k in range(h + 1, len(step_)):

                # 计算射线P1,P2
                P1 = cur_step_[h]
                P2 = pre_step_[h]
                Q1 = cur_step_[k]
                Q2 = pre_step_[k]
                Intera = ray_intersection(np.array(P1), np.array(P2), np.array(Q1), np.array(Q2))
                if Intera == 0:
                    A[s, h, k] = 0
                    A[s, k, h] = 0
                else:
                    dhk = anorm2(cur_step_[h], cur_step_[k])
                    dhc = anorm2(cur_step_[h], Intera)
                    dkc = anorm2(cur_step_[h], Intera)

                    A[s, h, k] = 1 / (dhk * (dhc + dkc))
                    A[s, k, h] = 1 / (dhk * (dhc + dkc))
                    # 需要修改考虑两个人之间的碰撞风险
        if norm_lap_matr:
            G = nx.from_numpy_array(A[s, :, :])
            A[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()  # 标准化拉普拉斯矩阵
    # 邻接矩阵和顶点特征
    return torch.from_numpy(V).type(torch.float), \
        torch.from_numpy(A).type(torch.float)


def seq_to_feature(seq_):
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    V = np.zeros((seq_len, max_nodes, 2))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_[h]
    return torch.from_numpy(V).type(torch.float)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 4, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 4, full=True)[1]
    #print(res_x + res_y)
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0
def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002,
        min_ped=1, delim='\t',norm_lap_matr = True):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files 分隔符
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []

        frame_id = []
        valid_ped_list = []
        for path in all_files:
            print(path)
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))  #可构建序列数量

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0) # 列拼接

                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])#获取当前时间序列中的人数
                self.max_peds_in_frame = max(self.max_peds_in_frame,len(peds_in_curr_seq))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len)) # 这里的len(peds_in_curr_seq)是筛选后的人的数量
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                seq_total_peds = 0
                _non_linear_ped = []
                considered_ped_id = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    seq_total_peds += 1
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq_origin = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq_origin[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq_origin[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue

                    curr_ped_seq_abs = np.transpose(curr_ped_seq_origin[:, 2:4])  #[2,20]
                    curr_ped_seq = np.zeros(curr_ped_seq_abs.shape)
                    curr_ped_seq[:,1:] = curr_ped_seq_abs[:,1:] - curr_ped_seq_abs[:, 0].reshape(2, 1)
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    if (poly_fit(curr_ped_seq, pred_len, threshold)>=0):

                        curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq_abs
                        curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                        # Linear vs Non-Linear Trajectory
                        _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                        curr_loss_mask[_idx, pad_front:pad_end] = 1
                        num_peds_considered += 1
                        # 需要创建一个数组，用来存储每个序列中对应的人的id
                        considered_ped_id.append(ped_id)
                #print(num_peds_considered,seq_total_peds)
                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped # 每个序列中选的人的移动是否线性
                    num_peds_in_seq.append(num_peds_considered) # 每个序列中选的人数（数量）
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered]) # 每个序列筛选的人的位置帧
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered]) # 每个序列筛选人的相对位移帧

                    frame_id.append(frames[idx + obs_len])  # threshold frame
                    valid_ped_list.append(considered_ped_id)
        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        frame_idx = np.asarray(frame_id)
        valid_ped = np.concatenate(valid_ped_list, axis=0)  # maybe problematic
        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)

        self.valid_ped = torch.from_numpy(valid_ped).type(torch.float)
        self.frame_idx = torch.from_numpy(frame_idx).type(torch.float)

        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist() #采用一维累加标记每个序列的开始位置
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        #Convert to Graphs 
        self.v_obs = [] 
        self.A_obs = [] 
        self.v_pred = [] 
        self.A_pred = []
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end)) # 创建了一个进度条
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)

            start, end = self.seq_start_end[ss]
            v_,a_ = seq_to_graph(self.obs_traj[start:end,:],self.obs_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_,a_=seq_to_graph(self.pred_traj[start:end,:],self.pred_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index],self.valid_ped[start:end],self.frame_idx[index]

        ]
        return out
