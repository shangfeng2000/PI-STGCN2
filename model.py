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
from Model.multipathtransformer import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from Model.posenc import PositionalAgentEncoding
import torch.optim as optim

class HistoryEncoder(nn.Module):
    def __init__(self, args, pos_enc, in_dim=2):
        super().__init__()

        self.obs_len = args.obs_seq_len
        self.pred_len = args.pred_seq_len

        self.model_dim = args.tf_model_dim
        self.ff_dim = args.tf_ff_dim
        self.nhead = args.tf_nhead
        self.dropout = args.tf_dropout
        self.nlayer = args.he_tf_layer

        self.agent_enc_shuffle = pos_enc['agent_enc_shuffle']

        self.pooling = args.pooling

        self.in_dim = in_dim
        self.input_fc = nn.Linear(self.in_dim, self.model_dim)

        encoder_layers = TransformerEncoderLayer(self.model_dim, self.nhead, self.ff_dim, self.dropout)
        self.tf_encoder = TransformerEncoder(encoder_layers, self.nlayer)

        self.pos_encoder = PositionalAgentEncoding(self.model_dim, self.dropout,
                                                   concat=pos_enc['pos_concat'], max_a_len=pos_enc['max_agent_len'],
                                                   use_agent_enc=pos_enc['use_agent_enc'],
                                                   agent_enc_learn=pos_enc['agent_enc_learn'])

    def forward(self, traj_in, agent_mask, agent_enc_shuffle=None):
        #print("KDD测试2:",traj_in.shape)
        #print("KDD测试2:", agent_mask.shape)
        agent_num = traj_in.shape[1]

        # [N*8 1 model_dim] [N*8 1 model_dim] [8 N 1 model_dim]
        tf_in = self.input_fc(traj_in.view(-1, traj_in.shape[-1])).view(-1, 1, self.model_dim) #增维，由4变为256
        tf_in_pos = self.pos_encoder(tf_in, num_a=agent_num, agent_enc_shuffle=agent_enc_shuffle)
        tf_in_pos = tf_in_pos.reshape([self.obs_len, agent_num, 1, self.model_dim])
        # [N N]
        src_agent_mask = agent_mask.clone()

        # [8 N 1 model_dim] -> [8 N model_dim]
        history_enc = self.tf_encoder(tf_in_pos, mask=src_agent_mask, num_agent=agent_num)
        history_rs = history_enc.view(-1, agent_num, self.model_dim)
        # compute per agent context [N model_dim]
        if self.pooling == 'mean':
            agent_history = torch.mean(history_rs, dim=0)  # [N model_dim]
        else:
            agent_history = torch.max(history_rs, dim=0)[0]
        # print("KDD测试5:", history_enc.shape)
        # print("KDD测试5:", agent_history.shape)
        return history_enc, agent_history



class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical,self).__init__()
        self.kernel_size = kernel_size
        # 创建二维卷积层，在时间和空间维度对输入进行卷积
        # 填充大小、步长、膨胀率
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x) #图上定义的空间卷积运算
        x = torch.einsum('nctv,tvw->nctw', (x, A))
        return x.contiguous(), A
    

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 use_mdn = False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn,self).__init__()
        
#         print("outstg",out_channels)

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1 #通常为奇数卷积核
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,kernel_size[1])
        #self.gcn =nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=(stride, 1))
        

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        #x = self.gcn(x)
        x = self.tcn(x) + res
        
        if not self.use_mdn:
            x = self.prelu(x)

        return x, A

class social_stgcnn(nn.Module):
    def __init__(self,n_stgcnn =1,n_txpcnn=1,input_feat=2,output_feat=5,
                 seq_len=8,pred_seq_len=12,kernel_size=3):
        super(social_stgcnn,self).__init__()
        self.n_stgcnn= n_stgcnn
        self.n_txpcnn = n_txpcnn
                
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(input_feat,output_feat,(kernel_size,seq_len)))
        for j in range(1,self.n_stgcnn):
            self.st_gcns.append(st_gcn(output_feat,output_feat,(kernel_size,seq_len)))
        
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len,pred_seq_len,(kernel_size,1),padding=((kernel_size - 1) // 2,0)))
        for j in range(1,self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len,pred_seq_len,3,padding=1))
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len,pred_seq_len,3,padding=1)
            
            
        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())


        
    def forward(self,v,a):

        for k in range(self.n_stgcnn):
            v,a = self.st_gcns[k](v,a)
            
        v = v.view(v.shape[0],v.shape[2],v.shape[1],v.shape[3])
        
        v = self.prelus[0](self.tpcnns[0](v))

        for k in range(1,self.n_txpcnn-1):
            v =  self.prelus[k](self.tpcnns[k](v)) + v
            
        v = self.tpcnn_ouput(v)
        v = v.view(v.shape[0],v.shape[2],v.shape[1],v.shape[3])
        
        
        return v,a

class self_attention_stgcn(nn.Module):
    def __init__(self,args):
        super(self_attention_stgcn, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')  # 使用 GPU
        else:
            self.device = torch.device('cpu')
        self.obs_seq_len = args.obs_seq_len
        self.pred_seq_len = args.pred_seq_len
        # position encoding
        self.pos_enc = {
            'pos_concat': args.pos_concat,
            'max_agent_len': 128,  # 128
            'use_agent_enc': False,  # False
            'agent_enc_learn': False,  # False
            'agent_enc_shuffle': False  # False
        }
        self.model_dim = args.tf_model_dim
        self.pred_dim = args.output_size
        # models
        self.history_encoder = HistoryEncoder(args, self.pos_enc)
        self.prelu = nn.PReLU()
        self.stgcnn = social_stgcnn(args.n_stgcnn, args.n_txpcnn, self.model_dim, self.pred_dim, self.obs_seq_len, self.pred_seq_len, args.kernel_size)

    def set_device(self, device):
        self.device = device
        self.to(device)
    def encode_history(self,v):
        # v:[b,d,T,N];
        # he_in:[T,N,d]
        v = v.squeeze(0)  # 去除第一个维度，得到 [d, T, N]
        he_in = v.permute(1, 2, 0)  # 将维度从 [d, T, N] 转换为 [T, N, d]
        history_enc, agent_history = self.history_encoder(he_in, self.agent_mask)
        return history_enc, agent_history

    def forward(self,v,a):
        device = self.device
        #v:[b,d,T,N]; a: [T,N,N]
        mask = torch.zeros([a.shape[1], a.shape[1]]).to(device) #[N,N]
        agent_mask = mask  # [N N] #计算掩码
        self.agent_mask = agent_mask
        history_enc, agent_history = self.encode_history(v) #history_enc:[T,N,1,d_model];agent_history:[N,d_model]
        v_e = history_enc.permute(2, 3, 0, 1)
        v_e = self.prelu(v_e)   #测试是否两者间需要添加激活函数
        v_pred, a = self.stgcnn(v_e, a)
        return v_pred, a

#定义一个pinn神经网络模型
#V_position：[1,para,T,N];V_velocity:[1,para,T,N];a_vel:[1,2,T-1,N]
class pinn(nn.Module):
    def __init__(self, N_INPUT=5, N_OUTPUT=2, N_HIDDEN=64, N_LAYERS=4):
        super(pinn, self).__init__()
        activation = nn.Tanh  # 使用双曲正切作为激活函数
        self.N_OUTPUT = N_OUTPUT
        self.fcs = nn.Sequential(*[nn.Linear(4*N_INPUT, N_HIDDEN), activation()])
        # 中间隐藏层，可能有多层
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation(), nn.Dropout(0.5)]) for _ in range(N_LAYERS - 1)])
        # 最后一层全连接层，从隐藏层到输出层
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self,v,v_vel):
        batch_size, para_size, T_size, N_size = v.shape
        #accelerate = torch.zeros(batch_size, 2, T_size - 1, N_size).to('cuda')
        pos_t1 = v[0, :, 0:T_size - 1, :]  # 第t-1时刻位置，形状 [1, feature_size, T_size-1, N_size]
        vel_t1 = v_vel[0, :, 0:T_size - 1, :]  # 第t-1时刻速度，形状 [1, feature_size, T_size-1, N_size]
        pos_t = v[0, :, 1:T_size, :]  # 第t时刻位置，形状 [1, feature_size, T_size-1, N_size]
        vel_t = v_vel[0, :, 1:T_size, :]  # 第t时刻速度，形状 [1, feature_size, T_size-1, N_size]
        # 合并位置和速度数据，形状为 [1, feature_size * 2, T_size-1, N_size]
        input_data = torch.cat([pos_t1, vel_t1, pos_t, vel_t], dim=0)  # 形状为 [1, 4, T_size-1, N_size]
        input_data = input_data.permute(2, 1, 0)  #[N, T, h]
        input_data=input_data.unsqueeze(0)
        # 调整输入数据形状以符合神经网络输入要求
        x = self.fcs(input_data)  # 假设 fcs 是一个线性层
        x = self.fch(x)  # 假设 fch 是后续层
        accelerate = self.fce(x)  # 假设 fce 是最终输出层
        accelerate = accelerate.view(batch_size, self.N_OUTPUT, T_size - 1, N_size)  # 形状 [1, feature_size, T_size-1, N_size]
        return accelerate

class pinn_new(nn.Module):
    def __init__(self, h, T, ORDER=4, N_HIDDEN=64, N_LAYERS=4):
        super(pinn_new, self).__init__()
        activation = nn.Tanh  # 使用双曲正切作为激活函数
        N_INPUT = 2*T * h
        N_OUTPUT = 2*ORDER
        self.fcs = nn.Sequential(*[nn.Linear(N_INPUT, N_HIDDEN), activation()])
        # 中间隐藏层，可能有多层
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation(), nn.Dropout(0.5)]) for _ in range(N_LAYERS - 1)])
        # 最后一层全连接层，从隐藏层到输出层
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self,v,v_vel):
        batch_size, para_size, T_size, N_size = v.shape
        # 合并位置和速度数据，形状为 [1, feature_size * 2, T_size-1, N_size]
        v_position = v[0]
        v_velocity = v_vel[0]
        input_data = torch.cat([v_position, v_velocity], dim=0)  # 形状为 [1, 4, T_size-1, N_size]
        input_data = input_data.permute(2, 1, 0)  #[N, T, h]
        N,T,h = input_data.shape
        input_data=input_data.contiguous().view(N, T * h)
        x = self.fcs(input_data)  # 假设 fcs 是一个线性层
        x = self.fch(x)  # 假设 fch 是后续层
        accelerate = self.fce(x)  # 假设 fce 是最终输出层
        accelerate = accelerate.view(-1, 2, 4)
        accelerate =accelerate.unsqueeze(0)
        return accelerate
#定义一个新的深度学习模型
class pi_stgcn(nn.Module):
    def __init__(self,args):
        super(pi_stgcn, self).__init__()
        self.p_gcn = self_attention_stgcn(args)
        self.v_gcn = self_attention_stgcn(args)
        self.pinn = pinn()
    def forward(self,v,a):
        #根据v计算出速度，
        v_velocity = v.clone()
        v_vel = v_velocity[:, :, 1:, :] - v_velocity[:, :, :-1, :]
        v_vel_first_element = v_vel[:, :, 0:1, :]
        v_vel = torch.cat([v_vel_first_element, v_vel], dim=2)
        V_position,_ = self.p_gcn(v,a)
        V_velocity,a = self.v_gcn(v_vel,a)
        V_accelerate = self.pinn(V_position,V_velocity)
        #V_accelerate=v_velocity[:, :, 1:, :] - v_velocity[:, :, :-1, :]
        return V_position, V_velocity,V_accelerate,a

class pi_stgcn_any_order(nn.Module):
    def __init__(self,args):
        super(pi_stgcn_any_order, self).__init__()
        self.p_gcn = self_attention_stgcn(args)
        self.v_gcn = self_attention_stgcn(args)
        self.pinn = pinn_new(5,12)
    def forward(self,v,a,v_abs):
        #根据v计算出速度，
        V_position,_ = self.p_gcn(v,a)
        V_velocity,a = self.v_gcn(v_abs,a)
        V_poly = self.pinn(V_position,V_velocity)
        #V_accelerate=v_velocity[:, :, 1:, :] - v_velocity[:, :, :-1, :]
        return V_position, V_velocity,V_poly,a

class st_pinn(nn.Module):
    def __init__(self, n_stgcnn =1,n_txpcnn=1,input_feat=2,output_feat=5,
                 seq_len=8,pred_seq_len=12,kernel_size=3,N_INPUT=3, N_OUTPUT=2, N_HIDDEN=16, N_LAYERS=4):
        super(st_pinn, self).__init__()
        activation = nn.Tanh  # 使用双曲正切作为激活函数
        self.stgcnn = social_stgcnn(n_stgcnn,n_txpcnn,input_feat,output_feat,seq_len,pred_seq_len,kernel_size)
        # 第一层全连接层，从输入层到隐藏层
        self.fcs = nn.Sequential(*[
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()])
        # 中间隐藏层，可能有多层
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation(), nn.Dropout(0.5)]) for _ in range(N_LAYERS - 1)])
        # 最后一层全连接层，从隐藏层到输出层
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
    def forward(self,v,a):
        v,a = self.stgcnn(v,a)
        v_position = v.clone().requires_grad_(True)
        v = v.permute(0, 2, 3, 1)  # (1,12,30,5)
        #print('问题调试0408：', v.shape)
        normx = v[:, :, :, 0]
        normy = v[:, :, :, 1]
        T = torch.ones_like(normx)
        for i in range(normx.shape[2]):
            T[:, :, i] = torch.linspace(1, normx.shape[1], normx.shape[1])
        v = torch.stack((T, normx, normy))
        v = v.permute(1, 2, 3, 0)  # (12,30,3)
        v = self.fcs(v)  # 通过第一层全连接层
        v = self.fch(v)  # 通过中间隐藏层
        v = self.fce(v)  # 通过最后一层全连接层
        return v_position, v, a

class st_pinn2(nn.Module):
    def __init__(self, n_stgcnn =1,n_txpcnn=1,input_feat=2,output_feat=5,
                 seq_len=8,pred_seq_len=12,kernel_size=3,N_INPUT=3, N_OUTPUT=2, N_HIDDEN=16, N_LAYERS=3):
        super(st_pinn2, self).__init__()
        activation = nn.Tanh  # 使用双曲正切作为激活函数
        self.stgcnn = social_stgcnn(n_stgcnn,n_txpcnn,input_feat,output_feat,seq_len,pred_seq_len,kernel_size)
        # 第一层全连接层，从输入层到隐藏层
        self.fcs = nn.Sequential(*[
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()])
        # 中间隐藏层，可能有多层
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Conv2d(N_HIDDEN,N_HIDDEN,(kernel_size,1),padding=((kernel_size - 1) // 2,0)),
                activation(), nn.Dropout(0.5)]) for _ in range(N_LAYERS - 1)])
        # 最后一层全连接层，从隐藏层到输出层
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
    def forward(self,v,a):
        # print("KDD测试1_v：",v.shape)
        # print("KDD测试1_a：", a.shape)
        v,a = self.stgcnn(v,a)
        v_position = v.clone().requires_grad_(True)
        v = v.permute(0, 2, 3, 1)  # (1,12,30,5)
        #print('问题调试0408：', v.shape)
        normx = v[:, :, :, 0]
        normy = v[:, :, :, 1]
        T = torch.ones_like(normx)
        for i in range(normx.shape[2]):
            T[:, :, i] = torch.linspace(1, normx.shape[1], normx.shape[1])
        v = torch.stack((T, normx, normy))
        v = v.permute(1, 2, 3, 0)  # (12,30,3)
        v = self.fcs(v)  # 通过第一层全连接层
        v = v.permute(0, 3, 1, 2)
        v = self.fch(v)  # 通过中间隐藏层
        v = v.permute(0, 2, 3, 1)
        v = self.fce(v)  # 通过最后一层全连接层
        return v_position, v, a