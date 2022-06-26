import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import metrics
from data import Company
from functions import FocalLoss
from sklearn.svm import SVC
import pandas as pd
import pickle
from model import EmbeddingClassify, TaxPayModel, TaxReturnLSTM
import torch.nn as nn
import torch.nn.functional as F

test = False
train_f = open("../data_jk/train_data.pkl", 'rb')
eval_f = open("../data_jk/eval_data.pkl", 'rb')
test_f = open("../data_jk/test.pkl", 'rb')
train_data = pickle.load(eval_f)
eval_data = pickle.load(train_f)
test_data = pickle.load(test_f)
lr=0.001
EPOCH = 10
batch_size = 16
seq_len = 16
model1 = EmbeddingClassify()
model2 = TaxReturnLSTM()
model3 = TaxPayModel()
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss()
criterion3 = nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr, weight_decay=0.99)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr, weight_decay=0.99)
optimizer3 = torch.optim.Adam(model3.parameters(), lr=lr, weight_decay=0.99)

def draw_roc(save_path, gt, pred, title="Val"):
    fpr, tpr, threshold = metrics.roc_curve(gt, pred)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.title(title+' ROC')
    plt.plot(fpr, tpr, 'b', label=title+' AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(save_path)
    plt.show()

def model_train():
    i=0
    for e in range(EPOCH):
        model1.train()
        model2.train()
        model3.train()
        pred1_list, pred2_list, pred3_list = [], [], []
        gt = []
        for id, comp in train_data.items():
            hy, inv, tax_re, tax_pay = comp.get_train_data()
            gt.append(comp.label)
            re_pred, re_info = model2(tax_re)
            pay_pred, pay_info = model3(tax_pay)

            pred = model1(hy, inv, re_info, pay_info)
            pred1_list.append(F.softmax(pred, dim=-1).detach().cpu().numpy()[1])
            pred2_list.append(F.softmax(re_pred, dim=-1).detach().cpu().numpy()[1])
            pred3_list.append(F.softmax(pay_pred, dim=-1).detach().cpu().numpy()[1])

            label = torch.LongTensor([comp.label])
            loss1 = criterion1(pred.unsqueeze(0), label)
            optimizer1.zero_grad()

            loss2 = criterion2(re_pred.unsqueeze(0), label)
            optimizer2.zero_grad()

            loss3 = criterion3(pay_pred.unsqueeze(0), label)
            optimizer3.zero_grad()
            loss1.backward(retain_graph=True)
            loss2.backward(retain_graph=True)
            loss3.backward()
            optimizer2.step()
            optimizer1.step()
            optimizer3.step()
            if i % 30 == 0:
                print("com %d, loss: %.3f, %.3f, %.3f"%(i, loss1, loss2, loss3))
            i+=1
        draw_roc("./vis/train_rocmodel1%d.jpg"%e, gt, pred1_list, title='train')
        draw_roc("./vis/train_rocmodel2%d.jpg"%e, gt, pred2_list, title='train')
        draw_roc("./vis/train_rocmodel3%d.jpg"%e, gt, pred3_list, title='train')
        # test 开始
        acc1, acc2, acc3 = 0,0,0
        pred1_list, pred2_list, pred3_list = [], [], []
        gt = []
        model1.eval()
        model2.eval()
        model3.eval()
        for id, comp in eval_data.items():
            hy, inv, tax_re, tax_pay = comp.get_train_data()
            label = comp.label
            gt.append(label)
            re_pred, re_info = model2(tax_re)
            pred2 = torch.argmax(re_pred, dim=0)
            acc2 += (pred2 == label)
            pay_pred, pay_info = model3(tax_pay)
            pred3 = torch.argmax(pay_pred, dim=0)
            acc3 += (pred3 == label)
            pred = model1(hy, inv, re_info, pay_info)
            pred1 = torch.argmax(pred, dim=0)
            acc1 += (pred1==label)
            # print(pred, pred1, label, F.softmax(pred, dim=-1).detach().cpu().numpy())
            pred1_list.append(F.softmax(pred, dim=-1).detach().cpu().numpy()[1])
            pred2_list.append(F.softmax(re_pred, dim=-1).detach().cpu().numpy()[1])
            pred3_list.append(F.softmax(pay_pred, dim=-1).detach().cpu().numpy()[1])
        draw_roc("./vis/eval_rocmodel1%d.jpg"%e, gt, pred1_list)
        draw_roc("./vis/eval_rocmodel2%d.jpg"%e, gt, pred2_list)
        draw_roc("./vis/eval_rocmodel3%d.jpg"%e, gt, pred3_list)
        print("in epoch %d , acc1,2,3 : %.3f, %.3f, %.3f"%(e, acc1/len(eval_data),
                                                           acc2/len(eval_data), acc3/len(eval_data)))
        if test:
            test_res = []
            for id, comp in test_data.items():
                hy, inv, tax_re, tax_pay = comp.get_train_data()
                pay_pred, _ = model3(tax_pay)
                pop = F.softmax(pay_pred, dim=-1).detach().cpu().numpy()[1]
                test_res.append({'SHXYDM':id, 'predict_prob':pop})
            df = pd.DataFrame(test_res)
            df.to_csv('test_res_ep%d.csv'%e)


def svm_data_pre(tax_re, tax_pay):
    m, c = tax_re.shape
    r_m = 52
    r_year=15
    new_tax_re = [0]*r_m
    j=r_year
    new_tax_re[0:min(r_year, m)] = tax_re[0:min(r_year,m), 0]
    for ci in range(1,c):
        for mi in range(m):
            if(tax_re[mi,ci]!=0):
                new_tax_re[j] = tax_re[mi,ci]
                j+=1
                if j >=r_m:
                    break
        if j >= r_m:
            break
    # 处理交税
    p_m = 7
    new_tax_pay = [0]*p_m*3
    j = 0
    for it, info in tax_pay.items():
        for info_m in info:
            new_tax_pay[j*3:(j+1)*3] = info_m
            j += 1
            if j >= p_m:
                break
        if j >= p_m:
            break
    return new_tax_re, new_tax_pay

def compress_data(in_data, get_label=False):
    svm_train_x = []
    if get_label:
        svm_train_y = []
    for id, comp in in_data.items():
        hy_dm = comp.HY_DM  # 6位行业编码
        inv = comp.get_four_investor().reshape(-1)  # 8位投资信息
        tax_re = comp.get_tax_return()  # month * 商品种类， 选取2个商品种类，12个月的数据
        tax_pay = comp.get_tax_pay()  # 商品种类： 各个月份， 选取2个商品种类，2个月的数据？
        tax_re, tax_pay = svm_data_pre(tax_re, tax_pay)
        new_info = []
        # new_info.extend(hy_dm)
        # new_info.extend(inv)
        new_info.extend(tax_re)
        new_info.extend(tax_pay)
        svm_train_x.append(new_info)
        if get_label:
            label = comp.label
            svm_train_y.append(label)
    if get_label:
        return np.array(svm_train_x), np.array(svm_train_y)
    else:
        return np.array(svm_train_x)

def svm_train():
    svm_train_x, svm_train_y = compress_data(train_data, get_label=True)
    svm_eval_x, svm_eval_y = compress_data(eval_data, get_label=True)
    clf = SVC(probability=True)
    clf.fit(svm_train_x, svm_train_y)
    pred = clf.predict( svm_eval_x )
    eval_acc = ( np.sum( pred == svm_eval_y ) / len(svm_eval_y))
    pred = clf.predict( svm_train_x )
    train_acc = ( np.sum( pred == svm_train_y ) / len(svm_train_y))
    print("eval_acc: %.3f, train acc: %.3f"%(eval_acc, train_acc))
    predp = clf.predict_proba( svm_eval_x )
    draw_roc("./vis/svm_eval.jpg", svm_eval_y, predp[:,1], "eval")
    predp = clf.predict_proba(svm_train_x)
    draw_roc("./vis/svm_train.jpg", svm_train_y, predp[:, 1], "train")


if __name__ == '__main__':
    svm_train()
