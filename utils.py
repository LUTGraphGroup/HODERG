import numpy as np
import pandas as pd
import random
from torch_geometric.data import Data
import matplotlib.pyplot as plt
# from scipy import interp
from sklearn.decomposition import PCA
from torch.optim.lr_scheduler import LRScheduler
import torch.nn as nn
import math
from typing import Any
import torch.nn.functional as F
import torch
from torch import Tensor
import numpy as np
from pathlib import Path
import os
import csv
from torchdiffeq import odeint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def decrease_to_max_value(x, max_value):
    x[x > max_value] = max_value
    return x


# construct association matrix
def constructNet(association_matrix):
    n, m = association_matrix.shape
    drug_matrix = torch.zeros((n, n), dtype=torch.int8)
    meta_matrix = torch.zeros((m, m), dtype=torch.int8)
    mat1 = torch.cat((drug_matrix, association_matrix), dim=1)
    mat2 = torch.cat((association_matrix.t(), meta_matrix), dim=1)
    adj_0 = torch.cat((mat1, mat2), dim=0)
    return adj_0

def TCMF1(alpha, Y, maxiter,A,B,C):
    iter0=1
    while True:

        a = np.dot(Y,B)
        b = np.dot(np.transpose(B),B)+alpha*C
        A = np.dot(a, np.linalg.inv(b))
        c = np.dot(np.transpose(Y),A)
        d = np.dot(np.transpose(A), A) + alpha * C
        B = np.dot(c, np.linalg.inv(d))

        if iter0 >= maxiter:
            #print('reach maximum iteration!')
            break
        iter0 = iter0 + 1

    Y= np.dot(A,np.transpose(B))
    Y_recover = Y

    return Y_recover

#异构网络矩阵分解算法
def run_MC(Y):
    maxiter = 1000
    alpha = 0.0001
    #SVD
    U, S, V = np.linalg.svd(Y)
    S=np.sqrt(S)
    r = 90
    Wt = np.zeros([r,r])
    for i in range(0,r):
        Wt[i][i]=S[i]
    U= U[:, 0:r]
    V= V[0:r,:]
    A = np.dot(U,Wt)
    B1 = np.dot(Wt,V)
    B=np.transpose(B1)
    C=Wt
    L  = TCMF1(alpha, Y, maxiter,A,B,C)
    return L



def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        # md_data = np.array(md_data)
        return torch.tensor(md_data)
        # return md_data

# W is the matrix which needs to be normalized
def new_normalization (w):
    m = w.shape[0]
    p = np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            if i == j:
                p[i][j] = 1/2
            elif np.sum(w[i,:])-w[i,i]>0:
                p[i][j] = w[i,j]/(2*(np.sum(w[i,:])-w[i,i]))
    return p

# get the KNN kernel, k is the number if first nearest neibors
def KNN_kernel (S, k):
    n = S.shape[0]
    S_knn = np.zeros([n,n])
    for i in range(n):
        sort_index = np.argsort(S[i,:])
        for j in sort_index[n-k:n]:
            if np.sum(S[i,sort_index[n-k:n]])>0:
                S_knn [i][j] = S[i][j] / (np.sum(S[i,sort_index[n-k:n]]))
    return S_knn



def MiRNA_updating(S1, S2, P1, P2):
    it = 0
    P = (P1 + P2) / 2
    dif = 1
    while dif > 0.0000001:
        it += 1
        P111 = np.dot(S1, P2)
        P111 = new_normalization(P111)
        P222 = np.dot(S2, P1)
        P222 = new_normalization(P222)
        P1 = P111
        P2 = P222
        P_New = (P1 + P2) / 2
        dif = np.linalg.norm(P_New - P) / np.linalg.norm(P)
        P = P_New
    print("Iter numb1", it)
    return P


def disease_updating(S1, S2, P1, P2):
    it = 0
    P = (P1 + P2) / 2
    dif = 1
    while dif > 0.0000001:
        it += 1
        P111 = np.dot(np.dot(S1, P2), S1.T)
        P111 = new_normalization(P111)
        P222 = np.dot(np.dot(S2, P1), S2.T)
        P222 = new_normalization(P222)
        P1 = P111
        P2 = P222
        P_New = (P1 + P2) / 2
        dif = np.linalg.norm(P_New - P) / np.linalg.norm(P)
        P = P_New
    print("Iter numb2", it)
    return P

# Multi-source feature fusion using SNF
def get_syn_sim (k1, k2):#k1=78，k2=37

    disease_semantic_sim = read_csv('data/HMDD v4.0/disease_sim/disSSim.csv')
    disease_GIP_sim = read_csv('data/HMDD v4.0/disease_sim/disGIPSim.csv')
    # disease_cos_sim = read_csv('data/HMDD v4.0/disease_sim/disCosSim.csv')
    disease_semantic_sim = disease_semantic_sim.numpy()
    disease_GIP_sim = disease_GIP_sim.numpy()
    # disease_cos_sim = disease_cos_sim.numpy()

    miRNA_GIP_sim = read_csv("data/HMDD v4.0/miRNA_sim/miGIPSim.csv")

    miRNA_func_sim = read_csv("data/HMDD v4.0/miRNA_sim/miFunSim_norm.csv")
    miRNA_GIP_sim = miRNA_GIP_sim.numpy()

    miRNA_func_sim = miRNA_func_sim.numpy()

    # Normalization of the miRNA similarity matrix
    mi_GIP_sim_norm = new_normalization(miRNA_GIP_sim)

    mi_func_sim_norm = new_normalization(miRNA_func_sim)

# Finding knn for miRNA similarity matrices
    mi_GIP_knn = KNN_kernel(miRNA_GIP_sim, k1)
    # mi_cos_knn = KNN_kernel(miRNA_cos_sim, k1)
    mi_func_knn = KNN_kernel(miRNA_func_sim, k1)

# Iteratively update each similarity network
    Pmi= MiRNA_updating(mi_GIP_knn, mi_func_knn, mi_GIP_sim_norm, mi_func_sim_norm)
    Pmi_final = (Pmi + Pmi.T)/2
# Normalization of the disease similarity matrix
    dis_sem_norm = new_normalization(disease_semantic_sim)
    dis_GIP_norm = new_normalization(disease_GIP_sim)
    # dis_cos_norm = new_normalization(disease_cos_sim)

# Disease similarity matrices are solved for knn
    dis_sem_knn = KNN_kernel(disease_semantic_sim, k2)
    dis_GIP_knn = KNN_kernel(disease_GIP_sim, k2)
    # dis_cos_knn = KNN_kernel(disease_cos_sim, k2)
    Pdiease = disease_updating(dis_sem_knn, dis_GIP_knn, dis_sem_norm, dis_GIP_norm)
    Pdiease_final = (Pdiease+Pdiease.T)/2
# Obtaining the final miRNA, Disease Similarity Matrix
    return Pmi_final, Pdiease_final


def load_data(seed, n_components):


    Adj = read_csv('data/HMDD v3.2 788  374/mi_dis_3.0.csv')
    count_ones = np.count_nonzero(Adj == 1)
    print("元素为1的个数：", count_ones)

    # Mi_simi,Dis_simi=get_syn_sim(78,37)
    Mi_simi=read_csv('data/HMDD v3.2 788  374/mi_sim/mi_fun_sim_3.0.csv ')
    Dis_simi = read_csv('data/HMDD v3.2 788  374/dis_sim/dis_sem_sim_3.0.csv')

    # Disease similarity matrix and its network construction
    Dis_simi = np.array(Dis_simi)
    Dis_adj = np.where(Dis_simi > 0.4, 1, 0)
    count_ones_disease = np.count_nonzero(Dis_adj)
    print("Dis_adj 中值为 1 的元素个数:", count_ones_disease)
    Dis_adj = torch.tensor(Dis_adj).to(device)


    Mi_simi = np.array(Mi_simi)
    Mi_adj = np.where(Mi_simi > 0.4, 1, 0)
    count_ones_meta = np.count_nonzero(Mi_adj)
    print("Mi_adj 中值为 1 的元素个数:", count_ones_meta)
    Mi_adj = torch.tensor(Mi_adj).to(device)

    # Initial Cos feature of disease and metabolites
    Dis_Cos = read_csv('data/HMDD v3.2 788  374/dis_sim/dis_cos_sim3.0.csv')
    # Meta_mol2vec, Dis_MESH2vec = get_syn_sim(78, 37)
    Dis_Cos = np.array(Dis_Cos)
    Mi_Cos = read_csv("data/HMDD v3.2 788  374/mi_sim/mi_cos_sim_3.0.csv ")
    Mi_Cos = np.array(Mi_Cos)

    a3 = np.hstack((Mi_Cos, Adj))  # 将参数元组的元素数组按水平方向进行叠加
    a4 = np.hstack((np.transpose(Adj), Dis_Cos))  # 对矩阵b进行转置操作
    H_1 = np.vstack((a3, a4))  # 将参数元组的元素数组按垂直方向进行叠加
    L1 = run_MC(H_1)
    M_1 = L1[0:Mi_Cos.shape[0], Mi_Cos.shape[0]:L1.shape[1]]  # 把补充的关联矩阵原来A位置给取出来



    # PCA Dimensionality Reduction
    pca = PCA(n_components=n_components)
    PCA_dis_feature = pca.fit_transform(Dis_Cos)
    PCA_mi_feature = pca.fit_transform(Mi_Cos)
    Dis_feature = torch.FloatTensor(PCA_dis_feature).to(device)
    Mi_feature = torch.FloatTensor(PCA_mi_feature).to(device)
    feature = torch.cat((Dis_feature, Mi_feature), dim=0).to(device)

    # Five-fold division of positive samples
    index_matrix = np.mat(np.where(Adj == 1))
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(seed)
    random.shuffle(random_index)
    k_folds = 5
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]
    random_index = temp



    return Adj,M_1, Dis_adj, Mi_adj, feature, random_index, k_folds


# laplacian_positional_encoding builds the structure matrix
def laplacian_positional_encoding(adj, pe_dim, device):
    # 确保邻接矩阵是浮点型
    adj = adj.float().to(device)  # 转换为浮点型并移动到指定设备

    # 归一化矩阵 N
    D = torch.diag(torch.pow(torch.sum(adj, dim=1).clamp(min=1), -0.5))
    N = D.to(device)  # 确保 N 在正确的设备上

    # 计算拉普拉斯矩阵 L
    L = torch.eye(adj.shape[0], device=device) - N @ adj @ N

    # 计算特征值和特征向量
    EigVal, EigVec = torch.linalg.eig(L)
    EigVal = EigVal.real
    EigVec = EigVec.real

    # 排序特征值及其对应的特征向量
    sorted_indices = EigVal.argsort()
    EigVec_sorted = EigVec[:, sorted_indices]

    # 获取拉普拉斯位置编码
    lap_pos_enc = EigVec_sorted[:, 1:pe_dim + 1].float()

    return lap_pos_enc

class HybridODE(nn.Module):
    def __init__(self, ode_hidden_dim):  # 修改参数名
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self._clamp_parameters()

    def _clamp_parameters(self):
        """约束参数范围"""
        with torch.no_grad():
            self.alpha.data.clamp_(0.01, 0.99)
            self.beta.data.clamp_(0.1, 10.0)

    def forward(self, t, x, adj):
        # 类型检查与转换
        if adj.dtype != x.dtype:
            adj = adj.to(x.dtype)

        # 1. 原始扩散项
        deg = adj.sum(1).clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        adj_norm = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
        diffusion = torch.mm(adj_norm, x) - x

        # 2. 动态注意力项
        with torch.no_grad():
            sim = torch.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)
            attn = F.softmax(self.beta * sim * adj, dim=1)

        # 3. 混合输出
        return diffusion + torch.sigmoid(self.alpha) * (torch.mm(attn, x) - x)

def compute_ode_features(adj, features, K, ode_func, time_span=0.05, method='dopri5'):
    """
    独立计算ODE扩散特征
    Args:
        adj: 邻接矩阵 [N, N]
        features: 初始特征 [N, D]
        K: 时间点数量-1
        ode_func: HybridODE实例
        time_span: 扩散时间跨度
        method: ODE求解器
    Returns:
        [N, K+1, D] 特征序列
    """

    # 统一转换为float32
    adj = adj.float()  # 确保邻接矩阵是float32
    features = features.float()  # 确保特征矩阵是float32

    t = torch.linspace(0, time_span, K+1).to(features.device)
    hops = odeint(
        lambda t, x: ode_func(t, x, adj),
        features,
        t,
        method=method
    )
    return hops.transpose(0, 1)  # [N, K+1, D]


class PolynomialDecayLR(LRScheduler):
    def __init__(self, optimizer, warmup_updates, tot_updates, lr, end_lr, power, last_epoch=-1, verbose=False):
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            self.warmup_factor = self._step_count / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
        elif self._step_count >= self.tot_updates:
            lr = self.end_lr
        else:
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (
                    self.tot_updates - warmup
            )
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr

        return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False


def glorot(value: Any):
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)


# Constructing positive and negative sample labels
def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1
    return link_labels

# plot the ROC curve for five-fold cross validation
def plot_auc_curves(fprs, tprs, auc, directory='./result', name='test_auc3.2 5'):
    """
    绘制五折交叉验证的ROC曲线

    参数:
        fprs (list of arrays): 各折的假阳性率数组列表，如[fpr_fold1, fpr_fold2,...]
        tprs (list of arrays): 各折的真阳性率数组列表，长度需与fprs一致
        auc (list): 各折的AUC值列表，长度需与fprs一致
        directory (str): 输出目录路径，默认'./result'
        name (str): 输出文件名（不含扩展名），默认'test_auc'
    """
    os.makedirs(directory, exist_ok=True)

    plt.figure(figsize=(8, 6))
    mean_fpr = np.linspace(0, 1, 20000)
    tpr_interp = []

    # 绘制各折曲线
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        tpr_i = np.interp(mean_fpr, fpr, tpr)
        tpr_i[0] = 0.0
        tpr_interp.append(tpr_i)
        plt.plot(fpr, tpr, alpha=0.3, linestyle=':',
                 label=f'Fold {i + 1} AUC: {auc[i]:.4f}')

    # 计算均值曲线
    mean_tpr = np.mean(tpr_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(auc)
    auc_std = np.std(auc)

    # 绘制平均曲线
    plt.plot(mean_fpr, mean_tpr, color='b', linewidth=2,
             label=f'Mean AUC: {mean_auc:.4f} ± {auc_std:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='k', alpha=0.5)

    # 图形设置
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves (5-fold CV)', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)

    # 保存PDF
    save_path = Path(directory) / f"{name}.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC曲线已保存至: {save_path}")


def plot_prc_curves(precisions, recalls, prc, directory='./result', name='test_prc3.2 5'):
    """
    绘制五折交叉验证的PR曲线

    参数:
        precisions (list of arrays): 各折的精确率数组列表
        recalls (list of arrays): 各折的召回率数组列表，长度需与precisions一致
        prc (list): 各折的AUPR值列表，长度需与precisions一致
        directory (str): 输出目录路径，默认'./result'
        name (str): 输出文件名（不含扩展名），默认'test_prc'
    """
    os.makedirs(directory, exist_ok=True)

    plt.figure(figsize=(8, 6))
    mean_recall = np.linspace(0, 1, 20000)
    prec_interp = []

    # 绘制各折曲线
    for i, (rec, prec) in enumerate(zip(recalls, precisions)):
        prec_i = np.interp(1 - mean_recall, 1 - rec, prec)
        prec_i[0] = 1.0
        prec_interp.append(prec_i)
        plt.plot(rec, prec, alpha=0.3, linestyle=':',
                 label=f'Fold {i + 1} AUPR: {prc[i]:.4f}')

    # 计算均值曲线
    mean_prec = np.mean(prec_interp, axis=0)
    mean_prec[-1] = 0
    mean_prc = np.mean(prc)
    prc_std = np.std(prc)

    # 绘制平均曲线
    plt.plot(mean_recall, mean_prec, color='r', linewidth=2,
             label=f'Mean AUPR: {mean_prc:.4f} ± {prc_std:.4f}')
    plt.plot([1, 0], [0, 1], linestyle='--', color='k', alpha=0.5)

    # 图形设置
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('PR Curves (5-fold CV)', fontsize=14)
    plt.legend(loc='lower left', fontsize=10)

    # 保存PDF
    save_path = Path(directory) / f"{name}.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PR曲线已保存至: {save_path}")

