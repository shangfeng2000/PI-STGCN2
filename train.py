
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
from torch.utils.data import TensorDataset
from numpy import linalg as LA
import networkx as nx

from utils import *
from metrics import *
import pickle
import argparse
from torch import distributed as dist
from torch import autograd
import torch.optim.lr_scheduler as lr_scheduler
from model import *

def reduce_mean(tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def graph_loss(V_pred ,V_target):
    return bivariate_loss(V_pred ,V_target)

# 损失函数计算
def graph_pinn_loss(V_trgt, V_pred,Velocity_pred, Accelarete_pred, lambda1=1e-1, lambda2=1e-1):
    graph_loss = bivariate_loss(V_pred, V_trgt)
    pinn_physic_loss = velocity_physic_loss(V_pred,Velocity_pred, Accelarete_pred)
    loss = graph_loss +lambda2 *pinn_physic_loss
    return loss,graph_loss,pinn_physic_loss

def graph_pinn_loss_any_order(V_trgt, V_pred,Velocity_pred, Accelarete_pred, lambda1=1e-1, lambda2=1e-1):
    graph_loss = bivariate_loss(V_pred, V_trgt)+lambda1*construct_loss(V_pred, V_trgt)
    pinn_physic_loss = poly_physic_loss_new(V_pred,Velocity_pred, Accelarete_pred)  #测试用V_trgt好还是V_pred好
    loss = graph_loss +lambda2 *pinn_physic_loss
    return loss,graph_loss,pinn_physic_loss

def train(args, epoch ,model, loader_train, optimizer, metrics):
    model.train()
    loss_batch = 0
    loss_g_batch = 0
    loss_p_batch = 0

    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_train)
    turn_point =int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1

    for cnt, batch in enumerate(loader_train):
        batch_count += 1

        # Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
            loss_mask, V_obs, A_obs, V_tr, A_tr, valid_ped, frame_idx = batch

        optimizer.zero_grad()
        # Forward
        # V_obs = batch,seq,node,feat
        # V_obs_tmp = batch,feat,seq,node
        V_obs_abs = obs_traj.permute(0,2,3,1)
        V_obs_abs = V_obs_abs.contiguous()
        V_pred_abs = pred_traj_gt.permute(0,3,1,2)
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        V_pred, Velocity_pred, Accelerate_pred, _ = model(V_obs_tmp, A_obs.squeeze(),V_obs_abs)

        V_pred = V_pred.permute(0, 2, 3, 1)    #[1,T,N,2]
        Velocity_pred = Velocity_pred.permute(0, 2, 3, 1)
        #Accelerate_pred = Accelerate_pred.permute(0, 2, 3, 1)

        V_tr = V_tr.squeeze()  # tr的维度没有发生变化
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()
        Velocity_pred = Velocity_pred.squeeze()
        Accelerate_pred = Accelerate_pred.squeeze()

        if batch_count % args.batch_size != 0 and cnt != turn_point:
            increase_rate = 1.01
            max_lambda_value =5
            construct_loss_lambda =args.construct_loss_lambda * (increase_rate ** epoch)
                # 如果已达到最大值，保持不变
            if construct_loss_lambda > max_lambda_value:
                construct_loss_lambda = max_lambda_value
            #construct_loss_lambda = args.construct_loss_lambda * (alpha_decay_rate ** epoch)
            if (args.is_any_order == False):
                l, l_g, l_p = graph_pinn_loss(V_tr, V_pred, Velocity_pred, Accelerate_pred, lambda1=construct_loss_lambda, lambda2=args.physical_loss_lambda)
            else:
                l, l_g, l_p = graph_pinn_loss_any_order(V_tr, V_pred, Velocity_pred, Accelerate_pred, lambda1=construct_loss_lambda, lambda2=args.physical_loss_lambda)
            if is_fst_loss:
                loss = l
                loss_g = l_g
                loss_p = l_p
                is_fst_loss = False
            else:
                loss += l
                loss_g += l_g
                loss_p += l_p

        else:
            loss = loss / args.batch_size  # 每个批次的平均损失值
            loss_g = loss_g / args.batch_size
            loss_p = loss_p / args.batch_size
            is_fst_loss = True
            loss.backward()

            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)  # 对梯度进行裁剪，提高训练稳定性和效率

            optimizer.step()
            # Metrics
            loss_batch += loss.item()
            loss_g_batch += loss_g.item()
            loss_p_batch += loss_p.item()
            print('TRAIN:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)  # 实时总的平均批次损失

    metrics['train_loss'].append(loss_batch / batch_count)
    return loss_batch / batch_count, loss_p_batch/batch_count


def vald(args, epoch, model, metrics, loader_val, constant_metrics, checkpoint_dir):
    # global metrics,loader_val,constant_metrics
    model.eval()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_val)
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1

    for cnt, batch in enumerate(loader_val):
        batch_count += 1

        # Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
            loss_mask, V_obs, A_obs, V_tr, A_tr ,valid_ped, frame_idx= batch

        V_obs_abs = obs_traj.permute(0, 2, 3, 1)
        V_obs_abs = V_obs_abs.contiguous()
        V_pred_abs = pred_traj_gt.permute(0, 3, 1, 2)
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        V_pred, Velocity_pred, Accelerate_pred, _ = model(V_obs_tmp, A_obs.squeeze(),V_obs_abs)

        V_pred = V_pred.permute(0, 2, 3, 1)
        Velocity_pred = Velocity_pred.permute(0, 2, 3, 1)
        #Accelerate_pred = Accelerate_pred.permute(0, 2, 3, 1)

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()
        Velocity_pred = Velocity_pred.squeeze()
        Accelerate_pred = Accelerate_pred.squeeze()


        if batch_count % args.batch_size != 0 and cnt != turn_point:
            if (args.is_any_order == False):
                l, l_g, l_p = graph_pinn_loss(V_tr, V_pred, Velocity_pred, Accelerate_pred, lambda1=args.construct_loss_lambda, lambda2=args.physical_loss_lambda)
            else:
                l, l_g, l_p = graph_pinn_loss_any_order(V_tr, V_pred, Velocity_pred, Accelerate_pred, lambda1=args.construct_loss_lambda, lambda2=args.physical_loss_lambda)
            if is_fst_loss:
                loss = l
                loss_g = l_g
                loss_p = l_p
                is_fst_loss = False
            else:
                loss += l
                loss_g += l_g
                loss_p += l_p

        else:
            loss = loss / args.batch_size
            is_fst_loss = True
            # Metrics
            loss_batch += loss.item()
            print('VALD:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)

    metrics['val_loss'].append(loss_batch / batch_count)

    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + 'val_best.pth')  # OK

def init_distributed_backend(backend='gloo', world_size=1, rank=0):
    torch.distributed.init_process_group(
        backend=backend,  # Use 'gloo' if you are on CPU or Windows
        init_method='env://',  # Using environment variables
        world_size=world_size,
        rank=rank
    )
def main(args):

    device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    # Data prep
    obs_seq_len = args.obs_seq_len
    pred_seq_len = args.pred_seq_len
    data_set = './datasets/' + args.dataset + '/'

    dset_train = TrajectoryDataset(
        data_set + 'train/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=5, norm_lap_matr=True)

    loader_train = DataLoader(
        dset_train,
        shuffle =  True,
        batch_size=1,  # This is irrelative to the args batch size parameter
        pin_memory=True,
        num_workers=16)

    dset_val = TrajectoryDataset(
        data_set + 'val/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=5, norm_lap_matr=True)

    loader_val = DataLoader(
        dset_val,
        batch_size=1,  # This is irrelative to the args batch size parameter
        shuffle=False,
        num_workers=1)

    # Defining the model
    #model = self_attention_stgcn(args).cuda()
    if (args.is_any_order == False):
        model = pi_stgcn(args).cuda()
    else:
        model = pi_stgcn_any_order(args).cuda()

    #打印参数量
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # Training settings
    optimizer = optim.SGD(model.parameters(), lr=args.lr)  # 随机梯度下降优化
    if args.use_lrschd:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)  # 基于步数的学习率调度器
    checkpoint_dir = './KDD_checkpoint/' + args.tag + '/'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    print('Data and model loaded')
    print('Checkpoint dir:', checkpoint_dir)

    # Training
    metrics = {'train_loss': [], 'val_loss': []}
    constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999}

    print('Training started ...')
    #model.set_device(device)
    for epoch in range(args.num_epochs):
        start_time = time.time()
        print(epoch)
        l,lp = train(args, epoch, model, loader_train, optimizer, metrics)
        vald(args, epoch, model, metrics, loader_val, constant_metrics, checkpoint_dir)
        if args.use_lrschd:
            scheduler.step()  # 学习率调度器，调整学习率

        print('*' * 30)
        print('Epoch:', args.tag, ":", epoch)
        for k, v in metrics.items():
            if len(v) > 0:
                print(k, v[-1])

        print(constant_metrics)
        print('*' * 30)

        with open(checkpoint_dir + 'metrics.pkl', 'wb') as fp:
            pickle.dump(metrics, fp)

        with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as fp:
            pickle.dump(constant_metrics, fp)  # 在验证集上表现最优的

        end_time = time.time()
        epoch_duration = end_time - start_time
        # 打印每个 epoch 的时间
        print(f"Epoch [{epoch + 1}/{args.num_epochs}] completed in {epoch_duration:.2f} seconds.")


if __name__ == '__main__':
    # 代码
    parser = argparse.ArgumentParser()

    # Model specific parameters
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    parser.add_argument('--n_stgcnn', type=int, default=1, help='Number of ST-GCNN layers')
    parser.add_argument('--n_txpcnn', type=int, default=5, help='Number of TXPCNN layers')
    parser.add_argument('--kernel_size', type=int, default=5)

    # Data specifc paremeters
    parser.add_argument('--obs_seq_len', type=int, default=8)
    parser.add_argument('--pred_seq_len', type=int, default=12)
    parser.add_argument('--dataset', default='eth',
                        help='eth,hotel,univ,zara1,zara2')

    # model architecture
    parser.add_argument('--pos_concat', type=bool, default=True)
    parser.add_argument('--tf_model_dim', type=int, default=128)
    parser.add_argument('--tf_ff_dim', type=int, default=128)
    parser.add_argument('--tf_nhead', type=int, default=4)
    parser.add_argument('--tf_dropout', type=float, default=0.1)
    parser.add_argument('--he_tf_layer', type=int, default=1)  # he = history encoder
    parser.add_argument('--pooling', type=str, default='mean')

    # Training specifc parameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='gadient clipping')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--lr_sh_rate', type=int, default=150,
                        help='number of steps to drop the lr')
    parser.add_argument('--use_lrschd', action="store_true", default=False,
                        help='Use lr rate scheduler')
    parser.add_argument('--tag', default='eth',
                        help='personal tag for the model ')
    parser.add_argument('--construct_loss_lambda', type=float, default=1, help="the weight of bound loss")
    parser.add_argument('--physical_loss_lambda', type=float, default=0.1, help="the weight of physical loss")

    # training options
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--is_any_order', default=True)
    args = parser.parse_args()

    print('*' * 30)
    print("Training initiating....")
    print(args)

    main(args)





