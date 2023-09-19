import torch
import torch.nn as nn
import numpy as np
from abc import ABC
from torch.autograd import Variable
from models import HGNN_weight
from torch.nn.parameter import Parameter


class seHGNN(nn.Module, ABC):
    def __init__(self, num_of_hidden, num_of_hidden_classify, W):
        super(seHGNN, self).__init__()

        self.num_of_hidden = num_of_hidden
        self.num_of_hidden_classify = num_of_hidden_classify
        self.W = W
        self.W = torch.diag_embed(self.W)
        self.W = Parameter(torch.squeeze(self.W))

        self.ReLU = nn.ReLU(inplace=True)

        self.encoder_0 = HGNN_weight(in_ch=3, n_class=3, n_hid=3, W=self.W, dropout=0.0, momentum=0.1)

        self.decoder_0 = HGNN_weight(in_ch=3, n_class=3, n_hid=3, W=self.W, dropout=0.0, momentum=0.1)

        self.classifier_0 = nn.Sequential(nn.Linear(in_features=self.num_of_hidden,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          nn.ReLU(inplace=True))

        # residual
        self.classifier_1 = nn.Sequential(nn.Linear(in_features=self.num_of_hidden_classify,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p=0.2, inplace=False),
                                          nn.Linear(in_features=self.num_of_hidden_classify,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p=0.2, inplace=False),
                                          )
        # residual
        self.classifier_2 = nn.Sequential(nn.Linear(in_features=self.num_of_hidden_classify,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p=0.2, inplace=False),
                                          nn.Linear(in_features=self.num_of_hidden_classify,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p=0.2, inplace=False),
                                          )
        # residual
        self.classifier_3 = nn.Sequential(nn.Linear(in_features=self.num_of_hidden_classify,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p=0.2, inplace=False),
                                          nn.Linear(in_features=self.num_of_hidden_classify,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p=0.2, inplace=False),
                                          )

        self.classifier_out = nn.Sequential(nn.Linear(in_features=self.num_of_hidden_classify,
                                                      out_features=2, bias=True))

    def forward(self, x, DV2_H_invDE, invDE_HT_DV2, H):
        y, HW, W_sparse, w2 = self.encoder_0(x, DV2_H_invDE, invDE_HT_DV2, H)
        y = self.ReLU(y)
        y_h = y
        HW_h = HW

        z, _, _, _ = self.decoder_0(y, DV2_H_invDE, invDE_HT_DV2, H)

        y = torch.flatten(y, 1)
        y0 = self.classifier_0(y)

        y1 = self.classifier_1(y0)
        y2 = self.classifier_2(y1 + y0)
        y3 = self.classifier_3(y2 + y1)

        out = self.classifier_out(y3 + y2 + y0)

        return z, out, y_h, HW_h, W_sparse, w2


