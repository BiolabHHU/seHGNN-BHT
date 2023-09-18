from torch import nn
from models import HGNN_conv1,HGNN_conv2
import torch.nn.functional as F
from hypergraph_construct import compute_G


class HGNN1(nn.Module):
    def __init__(self, in_ch, n_class, n_hid):
        super(HGNN1, self).__init__()
        #self.dropout = dropout
        self.hgc1 = HGNN_conv1(in_ch, n_hid)
        self.hgc2 = HGNN_conv2(n_hid, n_class)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        #x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        #x = F.relu(x,inplace=True)
        return x

class HGNN2(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout):
        super(HGNN2, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv1(in_ch, n_hid)
        self.hgc2 = HGNN_conv1(n_hid, n_class)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        # x = F.relu(x,inplace=True)
        return x

class HGNN3(nn.Module):
    def __init__(self, in_ch, n_class, n_hid):
        super(HGNN3, self).__init__()
        #self.dropout = dropout
        self.hgc1 = HGNN_conv2(in_ch, n_hid)
        self.hgc2 = HGNN_conv2(n_hid, n_class)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        #x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        #x = F.relu(x,inplace=True)
        return x

class HGNN_weight(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, W, dropout=0.0, momentum=0.1):
        super(HGNN_weight, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv1(in_ch, n_hid)
        self.hgc2 = HGNN_conv1(n_hid, n_class)
        self.computeG = compute_G(W)
        #self.batch_normalization1 = nn.BatchNorm1d(14, momentum=momentum)
        #self.batch_normalization2 = nn.BatchNorm1d(14, momentum=momentum)

    def forward(self, x, DV2_H_invDE, invDE_HT_DV2, H):
        #x = self.batch_normalization1(x)
        G, HW, W = self.computeG(DV2_H_invDE, invDE_HT_DV2, H)
        x, w2 = self.hgc1(x, G)
        #x = self.batch_normalization2(x)
        x = F.relu(x)
        #x = self.batch_normalization2(x)
        x = F.dropout(x, self.dropout)
        x, _ = self.hgc2(x, G)
        return x, HW, W, w2

