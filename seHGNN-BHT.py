import argparse
import numpy as np
import matlab.engine
from sklearn.metrics import roc_auc_score
from model import *
import torch.optim as optim
from hypergraph_construct import distance, hypergraph_construct, generate_G_from_H

eng = matlab.engine.start_matlab()

parser = argparse.ArgumentParser(description="seHGNN")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
opt = parser.parse_args()

ctr_mse = nn.MSELoss(reduction='mean')
ctr_mse.cuda()
ctr_entropy = nn.CrossEntropyLoss()
ctr_entropy.cuda()

device_ids = [0]


def prepare_data(index):

    train_h0_data, train_h0_label, train_h1_data, train_h1_label, test_h0_label, test_h1_label, testlabel = eng.svm_two_class(index, nargout=7)

    train_h0_data = np.array(train_h0_data)
    train_h0_label = np.array(train_h0_label)
    train_h1_data = np.array(train_h1_data)
    train_h1_label = np.array(train_h1_label)

    num_h0 = train_h0_label.sum()
    num_h1 = train_h1_label.sum()

    train_h0_data = np.reshape(train_h0_data, (99, 3, 10))
    train_h0_data = train_h0_data.transpose((0, 2, 1))
    train_h1_data = np.reshape(train_h1_data, (99, 3, 10))
    train_h1_data = train_h1_data.transpose((0, 2, 1))

    K_neigs = 3

    H_modal1_h0 = [hypergraph_construct(distance(np.squeeze(train_h0_data[i, :, 0])), K_neigs,
                                     is_probH=False, m_prob=1) for i in range(train_h0_data.shape[0])]
    H_modal2_h0 = [hypergraph_construct(distance(np.squeeze(train_h0_data[j, :, 1])), K_neigs,
                                     is_probH=False, m_prob=1) for j in range(train_h0_data.shape[0])]
    H_modal3_h0 = [hypergraph_construct(distance(np.squeeze(train_h0_data[k, :, 2])), K_neigs,
                                     is_probH=False, m_prob=1) for k in range(train_h0_data.shape[0])]
    H_h0 = [np.concatenate([i, j, k], axis=1) for i, j, k in zip(H_modal1_h0, H_modal2_h0, H_modal3_h0)]

    H_modal1_h1 = [hypergraph_construct(distance(np.squeeze(train_h1_data[i, :, 0])), K_neigs,
                                     is_probH=False, m_prob=1) for i in range(train_h1_data.shape[0])]
    H_modal2_h1 = [hypergraph_construct(distance(np.squeeze(train_h1_data[j, :, 1])), K_neigs,
                                    is_probH=False, m_prob=1) for j in range(train_h1_data.shape[0])]
    H_modal3_h1 = [hypergraph_construct(distance(np.squeeze(train_h1_data[k, :, 2])), K_neigs,
                                     is_probH=False, m_prob=1) for k in range(train_h1_data.shape[0])]
    H_h1 = [np.concatenate([i, j, k], axis=1) for i, j, k in zip(H_modal1_h1, H_modal2_h1, H_modal3_h1)]

    DV2_H_invDE_h0, _, invDE_HT_DV2_h0 = generate_G_from_H(H_h0, variable_weight=True)
    DV2_H_invDE_h1, _, invDE_HT_DV2_h1 = generate_G_from_H(H_h1, variable_weight=True)

    return train_h0_data, train_h0_label, train_h1_data, train_h1_label, num_h0, num_h1, testlabel,\
           DV2_H_invDE_h0, invDE_HT_DV2_h0, DV2_H_invDE_h1, invDE_HT_DV2_h1


def train_h0(train_data, DV2_H_invDE, invDE_HT_DV2, train_label):

    train_data = torch.Tensor(train_data).float().cuda()
    DV2_H_invDE = np.array(DV2_H_invDE)
    invDE_HT_DV2 = np.array(invDE_HT_DV2)
    DV2_H_invDE = torch.Tensor(DV2_H_invDE).float().cuda()
    invDE_HT_DV2 = torch.Tensor(invDE_HT_DV2).float().cuda()
    train_label = torch.Tensor(train_label).long().reshape((len(train_label),)).cuda()

    model.train()

    for epoch in range(EPOCH):

        if epoch <= opt.milestone:
            current_lr = opt.lr
        elif 30 < epoch <= 60:
            current_lr = opt.lr
        elif 60 < epoch <= 90:
            current_lr = opt.lr
        elif 90 < epoch <= 120:
            current_lr = opt.lr
        else:
            current_lr = opt.lr

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        optimizer.zero_grad()
        z, p, _ = model(train_data, DV2_H_invDE, invDE_HT_DV2)
        loss = ctr_mse(z, train_data) + ctr_entropy(p, train_label)
        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        _, _, y = model(train_data, DV2_H_invDE, invDE_HT_DV2)

    return y


def train_h1(train_data, DV2_H_invDE, invDE_HT_DV2, train_label):

    train_data = torch.Tensor(train_data).float().cuda()
    DV2_H_invDE = np.array(DV2_H_invDE)
    invDE_HT_DV2 = np.array(invDE_HT_DV2)
    DV2_H_invDE = torch.Tensor(DV2_H_invDE).float().cuda()
    invDE_HT_DV2 = torch.Tensor(invDE_HT_DV2).float().cuda()
    train_label = torch.Tensor(train_label).long().reshape((len(train_label),)).cuda()

    model.train()

    for epoch in range(EPOCH):

        if epoch <= opt.milestone:
            current_lr = opt.lr
        elif 30 < epoch <= 60:
            current_lr = opt.lr
        elif 60 < epoch <= 90:
            current_lr = opt.lr
        elif 90 < epoch <= 120:
            current_lr = opt.lr
        else:
            current_lr = opt.lr

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        optimizer.zero_grad()
        z, p, _ = model(train_data, DV2_H_invDE, invDE_HT_DV2)
        loss = ctr_mse(z, train_data) + ctr_entropy(p, train_label)
        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        _, _, y = model(train_data, DV2_H_invDE, invDE_HT_DV2)

    return y


def judge2(y_h0_x, h0_label, y_h1_x, h1_label, num_h0, num_h1):
    if h0_label is None:
        if h1_label is None:
            pass

    yh0_np = np.array(y_h0_x)
    yh0_np = np.reshape(yh0_np, (99, 30))
    yh0_AD = np.split(yh0_np, (num_h0,))
    yh0_AD = np.array(yh0_AD)
    yh0_HC = np.copy(yh0_AD)
    yh0_AD = np.delete(yh0_AD, 1, axis=0)[0]
    yh0_HC = np.delete(yh0_HC, 0, axis=0)[0]

    # inter- and intra-class distance
    yh0_AD_avg = np.mean(yh0_AD, axis=(0,))
    yh0_HC_avg = np.mean(yh0_HC, axis=(0,))
    yh0_all_avg = np.mean(yh0_np, axis=(0,))

    yh0_intra_AD = np.sum(np.power(np.linalg.norm((yh0_AD - yh0_AD_avg), axis=1, keepdims=True).flatten(), 2))
    yh0_intra_HC = np.sum(np.power(np.linalg.norm((yh0_HC - yh0_HC_avg), axis=1, keepdims=True).flatten(), 2))
    yh0_intra_all = yh0_intra_AD + yh0_intra_HC

    yh0_inter_AD = np.sum(np.power(np.linalg.norm((yh0_all_avg - yh0_AD_avg), axis=0, keepdims=True), 2))
    yh0_inter_HC = np.sum(np.power(np.linalg.norm((yh0_all_avg - yh0_HC_avg), axis=0, keepdims=True), 2))
    yh0_inter_all = num_h0 * yh0_inter_AD + (yh0_np.shape[0] - num_h0) * yh0_inter_HC

    yh0_out_class = yh0_intra_all / yh0_inter_all

    yh1_np = np.array(y_h1_x)
    yh1_np = np.reshape(yh1_np, (99, 30))
    yh1_AD = np.split(yh1_np, (num_h1,))
    yh1_AD = np.array(yh1_AD)
    yh1_HC = np.copy(yh1_AD)
    yh1_AD = np.delete(yh1_AD, 1, axis=0)[0]
    yh1_HC = np.delete(yh1_HC, 0, axis=0)[0]

    # inter- and intra-class distance
    yh1_AD_avg = np.mean(yh1_AD, axis=(0,))
    yh1_HC_avg = np.mean(yh1_HC, axis=(0,))
    yh1_all_avg = np.mean(yh1_np, axis=(0,))

    yh1_intra_AD = np.sum(np.power(np.linalg.norm((yh1_AD - yh1_AD_avg), axis=1, keepdims=True).flatten(), 2))
    yh1_intra_HC = np.sum(np.power(np.linalg.norm((yh1_HC - yh1_HC_avg), axis=1, keepdims=True).flatten(), 2))
    yh1_intra_all = yh1_intra_AD + yh1_intra_HC

    yh1_inter_AD = np.sum(np.power(np.linalg.norm((yh1_all_avg - yh1_AD_avg), axis=0, keepdims=True), 2))
    yh1_inter_HC = np.sum(np.power(np.linalg.norm((yh1_all_avg - yh1_HC_avg), axis=0, keepdims=True), 2))
    yh1_inter_all = num_h1 * yh1_inter_AD + (yh1_np.shape[0] - num_h1) * yh1_inter_HC

    yh1_out_class = yh1_intra_all / yh1_inter_all

    if yh1_out_class >= yh0_out_class:
        return True
    else:
        return False


def train():
    k = 0
    HC2HC = 0
    HC2AD = 0
    AD2AD = 0
    AD2HC = 0
    pred_tyb = []
    pred_real = []

    for i in range(dict_data[name_of_data]):

        train_h0_data, train_h0_label, train_h1_data, train_h1_label, num_h0, num_h1, testlabel, DV2_H_invDE_h0, \
            invDE_HT_DV2_h0, DV2_H_invDE_h1, invDE_HT_DV2_h1 = prepare_data(index=i + 1)

        pred_real.append(np.rint(testlabel))

        y_h0 = train_h0(train_h0_data, DV2_H_invDE_h0, invDE_HT_DV2_h0, train_h0_label)

        y_h1 = train_h1(train_h1_data, DV2_H_invDE_h1, invDE_HT_DV2_h1, train_h1_label)

        judge_result2 = judge2(y_h0.cpu().detach(), train_h0_label, y_h1.cpu().detach(), train_h1_label, num_h0, num_h1)

        if (judge_result2 == True and testlabel == 0) or (judge_result2 == False and testlabel != 0):
            k += 1

        if judge_result2 == True :
            if testlabel == 0:
                HC2HC += 1
                pred_tyb.append(0)

        if judge_result2 == False :
            if testlabel != 0:
                AD2AD += 1
                pred_tyb.append(1)

        if judge_result2 == False :
            if testlabel == 0:
                HC2AD += 1
                pred_tyb.append(1)

        if judge_result2 == True :
            if testlabel != 0:
                AD2HC += 1
                pred_tyb.append(0)

        print('\n current loop:' + str(i + 1) + ' / ' + str(dict_data[name_of_data]) + '-------------')
        print('-------------' + str(j_out + 1) + ' / ' + '50' + '-------------\n')
        print('current accuracy: ' + str(k) + '/' + str(i + 1))

    tyb1 = 'AD2AD: {}, AD2HC: {}, HC2HC: {}, HC2AD: {}'
    print(tyb1.format(AD2AD, AD2HC, HC2HC, HC2AD))
    tyb2 = '1 Accuracy: {}%'
    print(tyb2.format(100 * k / dict_data[name_of_data]))
    tyb3 = '2 Sensitivity: {}%'
    sensitivity = AD2AD / (AD2AD + AD2HC)
    print(tyb3.format(100 * AD2AD / (AD2AD + AD2HC)))
    tyb4 = '3 Specificity: {}%'
    print(tyb4.format(100 * HC2HC / (HC2AD + HC2HC)))
    tyb5 = '4 Precision: {}%'
    precision = AD2AD / (AD2AD + HC2AD)
    print(tyb5.format(100 * AD2AD / (AD2AD + HC2AD)))
    tyb6 = '5 F1 score: {}%'
    print(tyb6.format(100 * 2 * sensitivity * precision / (sensitivity + precision)))
    AUC = roc_auc_score(pred_real, pred_tyb)
    print('6 AUC:{}'.format(AUC))
    print(name_of_data)

    results_txt = str(AD2AD) + '\t' + str(AD2HC) + '\t' + str(HC2HC) + '\t' + str(HC2AD) + '\t' + str(
        100 * k / dict_data[name_of_data]) + '\t' + str(100 * AD2AD / (AD2AD + AD2HC)) + '\t' + str(
        100 * HC2HC / (HC2AD + HC2HC)) + '\t' + str(100 * AD2AD / (AD2AD + HC2AD)) + '\t' + str(
        100 * 2 * sensitivity * precision / (sensitivity + precision)) + '\t' + str(AUC) + '\n'

    with open('./results/' + name_of_data + '.txt', "a+") as f:
        f.write(results_txt)
    '''
    with open('./ensemble/' + name_of_data + '.txt', "a+") as f:
        f.write(str(pred_tyb) + '\n')
    '''

if __name__ == '__main__':
    name_list = ['ADS18']
    dict_data = {'ADS18': 100}
    EPOCH_list = {'ADS18': 100}

    for i_out in range(0, 1):

        name_of_data = name_list[i_out]
        num_of_hidden = 30
        num_of_hidden_classify = 16
        Batch_size = dict_data[name_of_data] - 1
        EPOCH = EPOCH_list[name_of_data]
        W = torch.ones(1, 1, 30).float().cuda()

        for j_out in range(50):
            net = seHGNN(num_of_hidden, num_of_hidden_classify, W)
            model = nn.DataParallel(net, device_ids=device_ids).cuda()

            optimizer = optim.Adam(model.parameters(), lr=opt.lr)

            train()

            del net, model, optimizer