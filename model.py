import torch
import torch.nn as nn
import numpy as np
from abc import ABC
from torch.autograd import Variable
from models import HGNN1, HGNN2, HGNN3, HGNN_weight
from torch.nn.parameter import Parameter


class seHGNN(nn.Module, ABC):
    def __init__(self, num_of_hidden, num_of_hidden_classify, W):
        super(seHGNN, self).__init__()

        self.num_of_hidden = num_of_hidden
        #self.num_of_hidden2 = num_of_hidden2
        #self.num_of_hidden_classify0 = num_of_hidden_classify0
        self.num_of_hidden_classify = num_of_hidden_classify
        self.W = W
        self.W = torch.diag_embed(self.W)
        self.W = Parameter(torch.squeeze(self.W))

        #self.BatchNorm1d = nn.BatchNorm1d(17)
        self.ReLU = nn.ReLU(inplace=True)

        self.encoder_0 = HGNN_weight(in_ch=3, n_class=3, n_hid=3, W=self.W, dropout=0.0, momentum=0.1)
        #self.encoder_0 = nn.Sequential(
        #    HGNN(in_ch=3,n_class=3,n_hid=3,dropout=0.5), nn.ReLU(inplace=True))

        self.decoder_0 = HGNN_weight(in_ch=3, n_class=3, n_hid=3, W=self.W, dropout=0.0, momentum=0.1)

        #self.classifier_in = nn.Sequential(nn.Linear(in_features=self.num_of_hidden,
        #                                            out_features=self.num_of_hidden_classify0, bias=True),
        #                                nn.ReLU(inplace=True))
        self.classifier_0 = nn.Sequential(nn.Linear(in_features=self.num_of_hidden,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          nn.ReLU(inplace=True))

        # residual 1
        self.classifier_1 = nn.Sequential(nn.Linear(in_features=self.num_of_hidden_classify,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          #nn.BatchNorm1d(self.num_of_hidden_classify),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p=0.2, inplace=False),
                                          nn.Linear(in_features=self.num_of_hidden_classify,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          #nn.BatchNorm1d(self.num_of_hidden_classify),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p=0.2, inplace=False),
                                          )
        # residual 1
        self.classifier_2 = nn.Sequential(nn.Linear(in_features=self.num_of_hidden_classify,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          #nn.BatchNorm1d(self.num_of_hidden_classify),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p=0.2, inplace=False),
                                          nn.Linear(in_features=self.num_of_hidden_classify,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          #nn.BatchNorm1d(self.num_of_hidden_classify),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p=0.2, inplace=False),
                                          )
        # residual 1
        self.classifier_3 = nn.Sequential(nn.Linear(in_features=self.num_of_hidden_classify,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          #nn.BatchNorm1d(self.num_of_hidden_classify),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p=0.2, inplace=False),
                                          nn.Linear(in_features=self.num_of_hidden_classify,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          #nn.BatchNorm1d(self.num_of_hidden_classify),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p=0.2, inplace=False),
                                          )

        #self.classifier_4 = nn.Sequential(nn.Linear(in_features=self.num_of_hidden_classify,
        #                                            out_features=self.num_of_hidden2, bias=True),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Dropout(p=0.2, inplace=False),
        #                                  )

        self.classifier_out = nn.Sequential(nn.Linear(in_features=self.num_of_hidden_classify,
                                                      out_features=2, bias=True))

        '''
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if 0 <= m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif -clip_b < m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)
        '''

    def forward(self, x, DV2_H_invDE, invDE_HT_DV2, H):
        y, HW, W_sparse, w2 = self.encoder_0(x, DV2_H_invDE, invDE_HT_DV2, H)
        # y = self.BatchNorm1d(y)
        y = self.ReLU(y)
        y_h = y
        HW_h = HW

        z, _, _,_ = self.decoder_0(y, DV2_H_invDE, invDE_HT_DV2, H)

        y = torch.flatten(y, 1)
        # y_in = self.classifier_in(y)
        y0 = self.classifier_0(y)

        y1 = self.classifier_1(y0)
        y2 = self.classifier_2(y1 + y0)
        y3 = self.classifier_3(y2 + y1)

        out = self.classifier_out(y3 + y2 + y0)

        return z, out, y_h, HW_h, W_sparse, w2


