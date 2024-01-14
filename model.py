import torch
import torch.nn as nn
from abc import ABC
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

        self.ReLU = nn.ReLU()

        self.encoder_0 = HGNN_weight(in_ch=3, n_class=3, n_hid=3, W=self.W, dropout=0)

        self.decoder_0 = HGNN_weight(in_ch=3, n_class=3, n_hid=3, W=self.W, dropout=0)

        self.classifier_0 = nn.Sequential(nn.Linear(in_features=self.num_of_hidden,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          nn.ReLU())

        # residual
        self.classifier_1 = nn.Sequential(nn.Linear(in_features=self.num_of_hidden_classify,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          nn.ReLU(),
                                          nn.Dropout(p=0.2),
                                          nn.Linear(in_features=self.num_of_hidden_classify,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          nn.ReLU(),
                                          nn.Dropout(p=0.2),
                                          )
        # residual
        self.classifier_2 = nn.Sequential(nn.Linear(in_features=self.num_of_hidden_classify,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          nn.ReLU(),
                                          nn.Dropout(p=0.2),
                                          nn.Linear(in_features=self.num_of_hidden_classify,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          nn.ReLU(),
                                          nn.Dropout(p=0.2),
                                          )
        # residual
        self.classifier_3 = nn.Sequential(nn.Linear(in_features=self.num_of_hidden_classify,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          nn.ReLU(),
                                          nn.Dropout(p=0.2),
                                          nn.Linear(in_features=self.num_of_hidden_classify,
                                                    out_features=self.num_of_hidden_classify, bias=True),
                                          nn.ReLU(),
                                          nn.Dropout(p=0.2),
                                          )

        self.classifier_out = nn.Sequential(nn.Linear(in_features=self.num_of_hidden_classify,
                                                      out_features=2, bias=True))

    def forward(self, x, DV2_H_invDE, invDE_HT_DV2):
        y = self.encoder_0(x, DV2_H_invDE, invDE_HT_DV2)
        y = self.ReLU(y)
        y_h = y

        z = self.decoder_0(y, DV2_H_invDE, invDE_HT_DV2)

        y = torch.flatten(y, 1)
        y0 = self.classifier_0(y)

        y1 = self.classifier_1(y0)
        y2 = self.classifier_2(y1 + y0)
        y3 = self.classifier_3(y2 + y1)

        out = self.classifier_out(y3 + y2 + y0)

        return z, out, y_h


