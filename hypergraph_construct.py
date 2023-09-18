import numpy as np
import scipy.io as scio
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import nn
from models import HGNN

'''
def feature_concat(*F_list, normal_col=False):
    features = None
    for f in F_list:
        if f is not None and f != []:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # normal each column
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features
'''

def distance(x):

    x = np.mat(x)
    x = np.transpose(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def hypergraph_construct(dist_mat, k_neig, is_probH, m_prob):
    #  construct  hyperedge
    n_obj = dist_mat.shape[0]
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_edge):
        dist_mat[center_idx, center_idx] = 0
        dis_vec = dist_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H

# weighted
def generate_G_from_H(H, variable_weight=True):
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        # G = []
        DV2_H = []
        W = []
        invDE_HT_DV2 = []
        for sub_H in H:
            DV2_H1_invDE, W1, invDE_HT_DV21 = generate_G_from_H(sub_H, variable_weight)
            DV2_H.append(np.array(DV2_H1_invDE))
            W.append(np.array(W1))
            invDE_HT_DV2.append(np.array(invDE_HT_DV21))
        return DV2_H, W, invDE_HT_DV2

def _generate_G_from_H(H, variable_weight=True):
    # H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -0.5)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H_invDE= DV2 * H * invDE
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H_invDE, W, invDE_HT_DV2            # DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G

class compute_G(nn.Module):
    def __init__(self, W):
        super(compute_G, self).__init__()
        #self.W_diag = W
        #self.W_diag = torch.diag_embed(self.W_diag)
        #self.W_diag = torch.squeeze(self.W_diag)

        self.W = W
        # self.W = Parameter(F.softmax(self.W, dim=2))
        #self.W = torch.diag_embed(self.W)
        #self.W = Parameter(torch.squeeze(self.W))

        #self.W0 = Parameter(torch.ones(286, 1, 10))
        #self.W0 = Parameter(torch.diag_embed(self.W0))
        #self.W0 = Parameter(torch.squeeze(self.W0))

    def forward(self, DV2_H_invDE, invDE_HT_DV2, H):
        W = self.W
        #W = F.softmax(self.W, dim=2)
        #W = torch.diag_embed(W)
        #W = torch.squeeze(W)

        #W_diag =  torch.add(self.W_diag, self.W)
        invDE_HT_DV2 = invDE_HT_DV2
        HW = H.matmul(W)                                 # torch.bmm(H,W)
        #G = DV2_H_invDE.matmul((W+W.T)/2)
        G = DV2_H_invDE.matmul(W)
        G = G.matmul(W.T)
        G = G.matmul(invDE_HT_DV2)
        # G0 = torch.bmm(DV2_H, W)
        # G0 = torch.bmm(G0, invDE_HT_DV2)
        # G0 = torch.bmm(self.W0, G0)

        return G, HW, W



