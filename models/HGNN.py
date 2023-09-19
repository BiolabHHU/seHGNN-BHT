from torch import nn
from models import HGNN_conv1
import torch.nn.functional as F
from hypergraph_construct import compute_G
class HGNN_weight(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, W, dropout=0.0, momentum=0.0):
        super(HGNN_weight, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv1(in_ch, n_hid)
        self.hgc2 = HGNN_conv1(n_hid, n_class)
        self.computeG = compute_G(W)

    def forward(self, x, DV2_H_invDE, invDE_HT_DV2, H):
        G, HW, W = self.computeG(DV2_H_invDE, invDE_HT_DV2, H)
        x, w2 = self.hgc1(x, G)
        x = F.relu(x)
        x = F.dropout(x, self.dropout)
        x, _ = self.hgc2(x, G)
        return x, HW, W, w2

