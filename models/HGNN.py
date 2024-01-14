from torch import nn
from models import HGNN_conv
import torch.nn.functional as F
from hypergraph_construct import compute_G

class HGNN_weight(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, W, dropout):
        super(HGNN_weight, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)
        self.computeG = compute_G(W)

    def forward(self, x, DV2_H_invDE, invDE_HT_DV2):
        G = self.computeG(DV2_H_invDE, invDE_HT_DV2)
        x = self.hgc1(x, G)
        x = F.relu(x)
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x

