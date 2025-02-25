import os
import math
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import argparse
import glob
import torch.distributions.multivariate_normal as torchdist
from utils import *
from metrics import *
from model import *
import copy


def test(KSTEPS=20):
    global loader_test, model
    model.eval()
    ade_meter = AverageMeter()
    fde_meter = AverageMeter()
    ade_bigls = []
    fde_bigls = []
    v_ade_bigls = []
    raw_data_dict = {}
    tcss_data_dict = {}#存储最优的分布id
    step = 0
    for batch in loader_test:
        step += 1
        # Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
            loss_mask, V_obs, A_obs, V_tr, A_tr, valid_ped, frame_idx = batch

        num_of_objs = obs_traj_rel.shape[1]

        # Forward
        # V_obs = batch,seq,node,feat
        # V_obs_tmp = batch,feat,seq,node
        V_obs_abs = obs_traj.permute(0, 2, 3, 1)
        V_obs_abs = V_obs_abs.contiguous()
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        V_pred, _, Accelerate_pred,_ = model(V_obs_tmp, A_obs.squeeze(),V_obs_abs)
        # print(V_pred.shape)
        # torch.Size([1, 5, 12, 2])
        # torch.Size([12, 2, 5])
        V_pred = V_pred.permute(0, 2, 3, 1)
        # torch.Size([1, 12, 2, 5])>>seq,node,feat
        # V_pred= torch.rand_like(V_tr).cuda()

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()

        num_of_objs = obs_traj_rel.shape[1]
        V_pred, V_tr = V_pred[:, :num_of_objs, :], V_tr[:, :num_of_objs, :]
        # print(V_pred.shape)

        # For now I have my bi-variate parameters
        # normx =  V_pred[:,:,0:1]
        # normy =  V_pred[:,:,1:2]
        sx = torch.exp(V_pred[:, :, 2])  # sx
        sy = torch.exp(V_pred[:, :, 3])  # sy
        corr = torch.tanh(V_pred[:, :, 4])  # corr

        cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2).cuda()
        cov[:, :, 0, 0] = sx * sx
        cov[:, :, 0, 1] = corr * sx * sy
        cov[:, :, 1, 0] = corr * sx * sy
        cov[:, :, 1, 1] = sy * sy
        mean = V_pred[:, :, 0:2]

        mvnormal = torchdist.MultivariateNormal(mean, cov)  # 创建多元正态分布

        ### Rel to abs
        ##obs_traj.shape = torch.Size([1, 6, 2, 8]) Batch, Ped ID, x|y, Seq Len

        # Now sample 20 samples
        ade_ls = {}
        fde_ls = {}
        v_ade_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                V_x[0, :, :].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                V_x[-1, :, :].copy())
        motion_eq = Accelerate_pred.data.cpu().numpy().copy()
        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['motion_eq'] = motion_eq
        raw_data_dict[step]['pred'] = []
        raw_data_dict[step]['person'] = valid_ped.data.cpu().numpy().squeeze().copy()

        agent_traj = []
        traj_gt = []
        for k in range(KSTEPS):
            V_pred = mvnormal.sample()
            #V_pred = mean

            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                       V_x[-1, :, :].copy())
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))
            pred = []
            target = []
            obsrvs = []
            for n in range(num_of_objs):
                pred.append(V_pred_rel_to_abs[:, n, :])
                target.append(V_y_rel_to_abs[:, n, :])
                obsrvs.append(V_x_rel_to_abs[:, n, :])
            agent_traj.append(pred)
            traj_gt=target
        #记录均值
        V_mean = mean
        v_mean_rel_to_abs = nodes_rel_to_nodes_abs(V_mean.data.cpu().numpy().squeeze().copy(),
                                                   V_x[-1, :, :].copy())
        raw_data_dict[step]['Gauss'] = copy.deepcopy(v_mean_rel_to_abs)

        pred_arr = np.array(agent_traj)
        pred_arr = np.swapaxes(pred_arr, 0, 1)
        gt_arr = np.array(traj_gt)
        #print("KDD测试:",pred_arr.shape,gt_arr.shape)
        ade_step = compute_ADE(pred_arr, gt_arr)
        fde_step = compute_FDE(pred_arr, gt_arr)
        ade_meter.update(ade_step, n=num_of_objs)
        fde_meter.update(fde_step, n=num_of_objs)
        tcss_data_dict[step]={}
        ade_bigls.append(ade_step)
        fde_bigls.append(fde_step)
        #tcss_data_dict[step][n]=ade_index

    ade_ = sum(ade_bigls) / len(ade_bigls)
    fde_ = sum(fde_bigls) / len(fde_bigls)
    v_ade = 0
    return ade_meter.avg, fde_meter.avg, v_ade, raw_data_dict, tcss_data_dict


paths = ['./KDD_checkpoint/*social-stgcnn*']
KSTEPS = 20
test_file='univ'
print("*" * 50)
print('Number of samples:', KSTEPS)
print("*" * 50)

for feta in range(len(paths)):
    ade_ls = []
    fde_ls = []
    v_ade_ls = []
    path = paths[feta]
    exps = glob.glob(path)  # 查找符合指定模式的文件路径名的函数
    print('Model being tested are:', exps)
    exps = ['H:/GRK/STG/Social-STGCNN-master/KDD_ablation2_checkpoint/'+test_file]  # demo测试，利用生成的tag checkpoint
    for exp_path in exps:
        print("*" * 50)
        print("Evaluating model:", exp_path)

        model_path = exp_path + '/val_best.pth'
        args_path = exp_path + '/args.pkl'
        with open(args_path, 'rb') as f:
            args = pickle.load(f)
        print(args)
        stats = exp_path + '/constant_metrics.pkl'
        with open(stats, 'rb') as f:
            cm = pickle.load(f)
        print("Stats:", cm)

        # Data prep
        obs_seq_len = args.obs_seq_len
        pred_seq_len = args.pred_seq_len
        data_set = './datasets_icdm/' + args.dataset + '/'
        print(data_set)
        dset_test = TrajectoryDataset(
            data_set + 'test/',
            obs_len=obs_seq_len,
            pred_len=pred_seq_len,
            skip=1, norm_lap_matr=True)

        loader_test = DataLoader(
            dset_test,
            batch_size=1,  # This is irrelative to the args batch size parameter
            shuffle=False)

        # Defining the model
        if (args.is_any_order == False):
            model = pi_stgcn(args).cuda()
        else:
            model = pi_stgcn_any_order(args).cuda()
        # model = self_attention_stgcn(args, n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn, kernel_size=args.kernel_size).cuda()

        # model = social_stgcnn(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
        #                       input_feat=args.input_size, output_feat=args.output_size, seq_len=args.obs_seq_len,
        #                       kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len).cuda()
        model.load_state_dict(torch.load(model_path))

        ade_ = 999999
        fde_ = 999999
        v_ade_ = 999999
        print("Testing ....")
        ad, fd, v_ad, raw_data_dic_, tcss_data_dic_ = test()
        """
        with open('./Results/KDD/'+test_file+'_data.pickle', 'wb') as file:
            pickle.dump(raw_data_dic_, file)
        with open('./Results/KDD/'+test_file+'_tcss.pickle', 'wb') as file:
            pickle.dump(tcss_data_dic_, file)
        ade_ = min(ade_, ad)
        """

        #
        ade_ = min(ade_, ad)
        fde_ = min(fde_, fd)
        v_ade_ = min(v_ade_, v_ad)
        ade_ls.append(ade_)
        fde_ls.append(fde_)
        v_ade_ls.append(v_ade_)
        print("ADE:", ade_, " FDE:", fde_, "V-ADE:", v_ade_)

    print("*" * 50)

    print("Avg ADE:", sum(ade_ls) / len(paths))
    print("Avg FDE:", sum(fde_ls) / len(paths))
    print("Avg V-ADE:", sum(v_ade_ls) / len(paths))
    with open('./Results/KDD/ade_'+test_file+'.txt', 'a') as file:
        # 计算并写入平均 ADE、FDE、V-ADE，三者位于一行
        avg_ade = sum(ade_ls) / len(paths)
        avg_fde = sum(fde_ls) / len(paths)
        avg_v_ade = sum(v_ade_ls) / len(paths)
        file.write(f"{avg_ade}\t{avg_fde}\t{avg_v_ade}\n")
