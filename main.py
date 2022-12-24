import dgl
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import argparse
import time
from dataset import Dataset
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix
from models import BWGNN,GCN,GAT
from sklearn.model_selection import train_test_split
import os
from utils import *
import sys
import torch.optim as optim
from para import *

from calibration_methods import (
    HistogramBinning,
    IsotonicRegression,
    BayesianBinningQuantiles,
    TemperatureScaling,
    RBS,
)

def intra_distance_loss(output, labels):
    """
    Marginal regularization from CaGCN (https://github.com/BUPT-GAMMA/CaGCN)
    """
    output = F.softmax(output, dim=1)
    pred_max_index = torch.max(output, 1)[1]
    correct_i = pred_max_index==labels
    incorrect_i = pred_max_index!=labels
    output = torch.sort(output, dim=1, descending=True)
    pred,sub_pred = output[0][:,0], output[0][:,1]
    incorrect_loss = torch.sum(pred[incorrect_i]-sub_pred[incorrect_i]) / labels.size(0)
    correct_loss = torch.sum(1- pred[correct_i] + sub_pred[correct_i]) / labels.size(0)
    return incorrect_loss + correct_loss

def write(file, s):
    '''
    :param file: "cal_metrics.txt"
    :param s: string
    '''
    f = open(file, mode="a+")
    f.writelines(s)
    f.close()

# threshold adjusting for best macro f1
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre

def data_split(labels, args):
    index = list(range(len(labels)))
    if args.dataset == 'amazon':
        index = list(range(3305, len(labels)))
    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                            train_size=args.train_ratio,
                                                            random_state=2, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.67,
                                                            random_state=2, shuffle=True)
    train_mask = torch.zeros([len(labels)]).bool()
    val_mask = torch.zeros([len(labels)]).bool()
    test_mask = torch.zeros([len(labels)]).bool()
    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1
    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
    return train_mask, val_mask, test_mask

def create_save_path(args):
    # All paths: Save model + load data
    checkpoints_path = "checkpoints/{}_{}/".format(args.model_name, args.dataset)
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    save_path = checkpoints_path + str(args.run)
    return save_path

def metrics(labels, preds, probs):
    trec = recall_score(labels, preds)*100 # 异常点的正确率
    tpre = precision_score(labels, preds)*100
    tmf1 = f1_score(labels, preds, average='macro')*100
    tauc = roc_auc_score(labels, probs[:, 1])*100
    acc = accuracy_score(labels, preds)*100
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()  #
    nor_acc = tn/(fp+tn)*100 # 正常点的正确率
    s = f"ACC:{acc:.2f}\t nor_ACC:{nor_acc:.2f}\t REC:{trec:.2f}\t PRE:{tpre:.2f}\t MF1:{tmf1:.2f}\t AUC:{tauc:.2f} "
    return s

def classify_data(labels, logits, probs, type="anomaly"):
    if type == "anomaly":
       bool_labels = labels.astype(np.bool)
    elif type == "normal":
        bool_labels = ~labels.astype(np.bool)
    else:
        print("wrong")
    prob = probs[bool_labels]
    label = labels[bool_labels]
    logit = logits[bool_labels]
    return label, logit, prob

def select(Probs, Labels, sign="label_ano"):
    '''
    Probs: ndarray, (N,K)
    Labels: ndarray, (N,)
    sign: label_ano, label_nor, pred_ano, pred_nor, all
    '''
    Preds = Probs.argmax(axis=-1)
    if sign=="label_ano":
        mask = Labels.astype(np.bool_)
    elif sign=="label_nor":
        mask = ~Labels.astype(np.bool_)
    elif sign=="pred_ano":
        mask = Preds.astype(np.bool_)
    elif sign=="pred_nor":
        mask = ~Preds.astype(np.bool_)
    elif sign=="all":
        mask = np.ones_like(Labels).astype(np.bool_)
    else:
        raise ValueError
    probs = Probs[mask]
    labels = Labels[mask]
    return probs, labels

def calibrate_cal(labels, probs, args, M=15, cal_model_name="CaGNN"):
    '''
    probs: ndarray, (N, K)
    labels: ndarray, (N,)
    '''
    signs = ['label_ano', 'label_nor', 'pred_ano', 'pred_nor', 'all']
    for sign in signs:
        Probs, Labels = select(probs, labels, sign=sign)
        Confs = np.max(Probs, axis=-1)  # 找出prob的最大值，即confidence
        Preds = np.argmax(Probs, axis=-1)  # prob最大值的索引即预测label
        accs, confs, bin_len, bin_rat = get_uncalibrated_res(Labels, Preds, Confs, M=M)
        np.set_printoptions(precision=2)
        ECE = sum([np.abs(confs[i] - accs[i]) * bin_rat[i] for i in range(len(accs))])
        rel_diagram_sub(accs, confs, ECE, data_name=args.dataset, model_name=f"{sign}_{cal_model_name}")
        if sign == "all":
            s1 = metrics(Labels, Preds, Probs)
            # s1 = accuracy_score(Labels, Preds)
        else:
            s1 = f"{sign}"
        s2 = f"ECE:{ECE:.4f}\t M:{M}\t Model:{cal_model_name: <20s}\t {s1} \n"
        write(file=f"log/{args.dataset}_cal_metrics.txt", s=s2)
    print(f"log/{args.dataset}_cal_metrics.txt")
    print(f"{cal_model_name}")
    write(file=f"log/{args.dataset}_cal_metrics.txt", s="\n\n")

def threshold():
    # Threshold
    f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
    print(f"f1:{f1} thres:{thres}")
    preds = np.zeros_like(labels)
    preds[probs[:, 1] > thres] = 1

def loss_label(w, logits, probs, preds, labels_onehot):
    N = logits.shape[0] # 3
    M = preds.sum(dim=-1).item() # 1
    inver_preds = 1- preds
    #mae = torch.abs(probs - labels_onehot).sum(dim=-1) # (N,2) => (N,)
    mae = torch.nn.MSELoss(reduction='mean')(probs, labels_onehot)
    loss_wei = mae * preds * w + mae * inver_preds
    loss = loss_wei.sum() / (N + M*(w-1)) #(N,1)

    return loss # tensor(0.7335)

def loss_distribution(labels, preds, d1, d2):
    N, M = labels.shape[0], labels.sum().item()
    M_a = torch.tensor(preds.sum()/N, requires_grad=True) # anomaly
    M_n = 1 - M_a # normal
    loss_distribution = torch.abs(M_n - d1) + torch.abs(M_a - d2)
    return loss_distribution

def loss_anomaly(labels, probs, preds):
    labels_bool = labels.type(torch.bool)
    preds_anomaly = preds[labels_bool]
    labels_anomaly = labels[labels_bool]
    probs_anomaly = probs[labels_bool]

    correct_bool = preds_anomaly == labels_anomaly
    incorrect_bool = preds_anomaly != labels_anomaly
    probs_anomaly_correct = probs_anomaly[correct_bool]
    probs_anomaly_incorrect = probs_anomaly[incorrect_bool]

    probs_anomaly_correct_dif = (1-torch.abs(probs_anomaly_correct[:, 0]-probs_anomaly_correct[:, 1]))
    probs_anomaly_incorrect_dif = torch.abs(probs_anomaly_incorrect[:, 0]-probs_anomaly_incorrect[:, 1])
    loss = (probs_anomaly_correct_dif.sum() + probs_anomaly_incorrect_dif.sum())/probs_anomaly.shape[0]
    #loss = (probs_anomaly_correct_dif.mean() + probs_anomaly_incorrect_dif.mean())/2
    return loss

def loss_choose(logits, labels, val_mask, sign="CrossEntropy"):
    weight = (1 - labels[val_mask]).sum().item() / labels[val_mask].sum().item()
    N, M = labels.shape[0], labels.sum().item()
    d1, d2 = (N - M) / N, M / N
    #onehot = torch.eye(2)
    onehot = torch.tensor([[d1, d2],[0, 1]])
    # print(onehot)
    # exit()
    labels_onehot = onehot[labels]
    probs = logits.softmax(-1)  # (N,2)
    preds = logits.argmax(-1)  # (N,)
    loss_1 = loss_label(weight, logits[val_mask], probs[val_mask], preds[val_mask], labels_onehot[val_mask])
    loss_2 = loss_distribution(labels[val_mask], preds[val_mask], d1, d2)
    loss_3 = loss_anomaly(labels[val_mask], probs[val_mask], preds[val_mask])
    loss_0 = F.cross_entropy(logits[val_mask], labels[val_mask])
    loss_01 = F.cross_entropy(logits[val_mask], labels[val_mask], weight=torch.tensor([1., weight]))

    if sign=="CrossEntropy":
        loss = loss_0
    elif sign=="CaGCNLoss":
        loss = loss_01
        # dist_reg = intra_distance_loss(calibrated[train_mask], labels[train_mask])
        # margin_reg = 0.
        # loss = loss + margin_reg * dist_reg
    elif sign=="NewLoss":
        # w1 = nn.Parameter(torch.FloatTensor([0.3]))
        # w2 = nn.Parameter(torch.FloatTensor([0.3]))
        # w3 = nn.Parameter(torch.FloatTensor([0.3]))
        w1, w2, w3 = 0.01, 0.01, 1
        loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3
        #print("loss", loss_1, loss_2, loss_3)
    elif sign=="Loss_Ano":
        loss = loss_3
    elif sign=="Loss_Lab":
        loss = loss_1
    elif sign=="Loss_Dis":
        loss = loss_2
    else:
        raise ValueError
    return loss

def GCN(LOGITS, graph, args, cal_model, LOSS, Train=True):
    # Data
    labels = graph.ndata['label']
    train_mask, val_mask, test_mask = data_split(labels, args)

    # Instantiate your GCN model
    #model = GCN(2, args.GCN_hidden_dim, 1)
    model = GAT(2, args.GCN_hidden_dim, 1)
    optimizer = optim.Adam(model.parameters(), lr=args.GCN_lr, weight_decay=args.GCN_weight_decay)

    # save path
    checkpoints_path = "checkpoints/{}_{}_{}/".format(args.dataset, cal_model, LOSS)
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    save_path = checkpoints_path + '1'

    if Train:
        t_total = time.time()
        #best_loss_val, best_acc_val = float("inf"), 0
        best_loss_train, best_acc_train = float("inf"), 0
        for epoch in range(args.GCN_epochs):
            t = time.time()
            model.train()
            logits = model(graph, LOGITS) #(N,2)
            loss_val = loss_choose(logits, labels, val_mask, sign=LOSS)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            print('Epoch: {:04d}'.format(epoch + 1),
                  'time: {:.4f}s'.format(time.time() - t), end=' ')

            with torch.no_grad():
                model.eval()
                logits = model(graph, LOGITS)
                loss_train = loss_choose(logits, labels, train_mask, sign=LOSS)
                loss_val = loss_choose(logits, labels, val_mask, sign=LOSS)
                loss_test = loss_choose(logits, labels, test_mask, sign=LOSS)
                # Compute accuracy on training/validation/test
                preds = logits.argmax(1)
                acc = (preds == labels).float()
                acc_train, acc_val, acc_test = acc[train_mask].mean(), acc[val_mask].mean(), acc[test_mask].mean()
                print('loss_val: {:.4f}'.format(loss_val.item()),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'loss_test: {:.4f}'.format(loss_test.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      'acc_val: {:.4f}'.format(acc_val.item()),
                      'acc_test: {:.4f}'.format(acc_test.item()))

            # Save the best validation accuracy and the corresponding test accuracy.
            if loss_val < best_loss_val:
            #if acc_val > best_acc_val:
                best_acc_val = acc_val
                best_loss_val = loss_val
                torch.save(model.state_dict(), save_path)
                print("Save Model!")
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    else:
        model.load_state_dict(torch.load(save_path))
        with torch.no_grad():
            model.eval()
            logits = model(graph, LOGITS)
            probs = logits.softmax(1)
            Probs = probs[test_mask].detach().numpy()
            Labels = labels[test_mask].detach().numpy()
            calibrate_cal(Labels, Probs, args, M=15, cal_model_name=cal_model)

if __name__ == '__main__':
    args = parameter()

    # python main.py --dataset tsocial --train_ratio 0.4 --hid_dim 10 --order 5 --homo 1 --epoch 100 --run 1
    # tfinance: torch.Size([39357, 10]) torch.Size([39357])
    # amazon: torch.Size([11944, 25]) torch.Size([11944])
    graph = Dataset(args.dataset, args.homo).graph
    features = graph.ndata['feature']
    labels = graph.ndata['label']
    in_feats = features.shape[1]
    num_classes = 2

    # File
    save_path = create_save_path(args)

    # Data split
    train_mask, val_mask, test_mask = data_split(labels, args)
    # Build model
    model = BWGNN(in_feats, args.hid_dim, num_classes, graph, d=args.order)

    if args.train:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.
        weight = (1 - labels[train_mask]).sum().item() / labels[train_mask].sum().item()
        print('cross entropy weight: ', weight)
        time_start = time.time()
        for e in range(args.epoch):
            model.train()
            logits = model(features)
            loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.eval()
            probs = logits.softmax(dim=1)
            f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
            preds = np.zeros_like(labels)
            preds[probs[:, 1] > thres] = 1
            trec, tpre, tmf1, tauc = metrics(labels, preds, probs.detach().numpy(), test_mask)

            if best_f1 < f1:
                best_f1 = f1
                final_trec = trec
                final_tpre = tpre
                final_tmf1 = tmf1
                final_tauc = tauc
                if args.save_model:
                    torch.save(model.state_dict(), save_path)
                    print("Save Model!")
            print('Epoch {}, loss: {:.4f}, val mf1: {:.4f}, (best {:.4f})'.format(e, loss, f1, best_f1))

        time_end = time.time()
        print('time cost: ', time_end - time_start, 's')
        s = 'Train: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}\n'.format(final_trec * 100,
                                                                         final_tpre * 100, final_tmf1 * 100,
                                                                         final_tauc * 100)
    else:
        model.load_state_dict(torch.load(save_path)) # checkpoints/BWGNN_amazon/1
        with torch.no_grad():
            model.eval()
            logits = model(features)
            probs = logits.softmax(-1)
            probs_n = probs.detach().numpy()  # (N,K),ndarray
            labels_n = labels.detach().numpy().astype(np.int32)  # (N,), ndarray
            logits_n = logits.detach().numpy()  # (N,K) ndarray
            preds_n = np.argmax(probs_n, axis=-1)  # prob最大值的索引即预测label

        models = ["BWGNN","CaGCN","CaGAT","His","Iso","BBQ","TS"]
        cal_models_list = [int(i) for i in args.cal_models_list.split(",")]
        cal_models = [models[i] for i in cal_models_list]
        bins = [int(i) for i in args.bins_list.split(",")]

        # f = open(f"log/{args.dataset}_{args.cal_model}_{args.LOSS}.txt", mode="a+")
        # # sys.stdout = f
        # # sys.stderr = f
        # print(args)

        for b in bins:
            for model_name in cal_models:
                if model_name == "CaGCN":
                    GCN(logits, graph, args, model_name, "CrossEntropy", Train=args.GCN_train)
                elif model_name == "CaGAT":
                    GCN(logits, graph, args, model_name, "NewLoss", Train=args.GCN_train)
                else:
                    if model_name == "His":
                        cal_model = HistogramBinning()
                        cal_model.fit(probs_n[val_mask], labels_n[val_mask])
                        cal_test_probs = cal_model.predict_proba(probs_n[test_mask])
                    elif model_name == "Iso":
                        cal_model = IsotonicRegression()
                        cal_model.fit(probs_n[val_mask], labels_n[val_mask])
                        cal_test_probs = cal_model.predict_proba(probs_n[test_mask])
                    elif model_name == "BBQ":
                        cal_model = BayesianBinningQuantiles()
                        cal_model.fit(probs_n[val_mask], labels_n[val_mask])
                        cal_test_probs = cal_model.predict_proba(probs_n[test_mask])
                    elif model_name == "TS":
                        cal_model = TemperatureScaling()
                        cal_model.fit(logits_n[val_mask], labels_n[val_mask])
                        cal_test_probs = cal_model.predict_proba(logits_n[test_mask])
                    else:
                        cal_test_probs = probs_n[test_mask]
                    cal_test_labels = labels_n[test_mask]
                    calibrate_cal(cal_test_labels, cal_test_probs, args, M=b, cal_model_name=model_name)
