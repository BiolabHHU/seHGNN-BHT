import numpy as np
from torch import nn


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


def generate_G_from_H(H, variable_weight=True):
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
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
    n_edge = H.shape[1]

    W = np.ones(n_edge)
    DV = np.sum(H * W, axis=1)
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -0.5)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H_invDE= DV2 * H * invDE
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H_invDE, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G


class compute_G(nn.Module):
    def __init__(self, W):
        super(compute_G, self).__init__()
        self.W = W

    def forward(self, DV2_H_invDE, invDE_HT_DV2):
        G = DV2_H_invDE.matmul(self.W)
        G = G.matmul(self.W.T)
        G = G.matmul(invDE_HT_DV2)

        return G



