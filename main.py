import torch
import matplotlib.pyplot as plt
from sklearn import metrics
from data import Company
from functions import FocalLoss
import pandas as pd
import pickle
from model import EmbeddingClassify, TaxPayModel, TaxReturnLSTM
import torch.nn as nn
import torch.nn.functional as F


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

def draw_roc(save_path, gt, pred):
    fpr, tpr, threshold = metrics.roc_curve(gt, pred)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.title('Validation ROC')
    plt.plot(fpr, tpr, 'b', label='Val AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(save_path)
    plt.show()

i=0
for e in range(EPOCH):
    model1.train()
    model2.train()
    model3.train()
    for id, comp in train_data.items():
        hy, inv, tax_re, tax_pay = comp.get_train_data()

        re_pred, re_info = model2(tax_re)
        pay_pred, pay_info = model3(tax_pay)

        pred = model1(hy, inv, re_info, pay_info).unsqueeze(0)
        label = torch.LongTensor([comp.label])
        loss1 = criterion1(pred, label)
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
        if i % 10 == 0:
            print("com %d, loss: %.3f, %.3f, %.3f"%(i, loss1, loss2, loss3))
        i+=1
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
    draw_roc("./rocmodel1.jpg", gt, pred1_list)
    draw_roc("./rocmodel2.jpg", gt, pred2_list)
    draw_roc("./rocmodel3.jpg", gt, pred3_list)
    print("in epoch %d , acc1,2,3 : %.3f, %.3f, %.3f"%(e, acc1/len(eval_data),
                                                       acc2/len(eval_data), acc3/len(eval_data)))
    test_res = []
    for id, comp in test_data.items():
        hy, inv, tax_re, tax_pay = comp.get_train_data()
        pay_pred, _ = model3(tax_pay)
        pop = F.softmax(pay_pred, dim=-1).detach().cpu().numpy()[1]
        test_res.append({'SHXYDM':id, 'predict_prob':pop})
    df = pd.DataFrame(test_res)
    df.to_csv('test_res_ep%d.csv'%e)

