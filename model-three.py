import numpy as np
import torch
from data import Company, SVMDataSet
from torch.utils.data import Dataset, DataLoader
from functions import FocalLoss, draw_roc
import pandas as pd
import pickle
from model import EmbeddingClassify, TaxPayModel, TaxReturnLSTM
from model import MixFCModel as FCModel
import torch.nn as nn
import torch.nn.functional as F

test = False
train_f = "../data_jk/train_dataR099_fill0.pkl"
eval_f = "../data_jk/train_dataR099_fill0.pkl"
test_f = "../data_jk/test_fill0.pkl"

lr=0.001
EPOCH = 20
batch_size = 32
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


def model_train():
    with open(train_f, 'rb') as f1:
        train_data = pickle.load(f1)
    with open(eval_f, 'rb') as f2:
        eval_data = pickle.load(f2)
    with open(test_f, 'rb') as f3:
        test_data = pickle.load(f3)
    # eval_data = pickle.load(open(eval_f, 'rb'))
    # test_data = pickle.load(open(test_f, 'rb'))
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


def model_load(ckpt, model):
    model.load_state_dict(torch.load(ckpt))
    return model

def FCtest(model, ckpt):
    record_acc = ckpt.split('.')[-2].split('_')[-1]
    test_loader = DataLoader(SVMDataSet(test_f), batch_size=1, shuffle=False)
    model = model_load(ckpt, model).cuda()
    model.eval()
    test_res = []
    for batch, label, id in test_loader:
        data = torch.FloatTensor(batch).cuda()
        pred = model(data)
        pred_p = F.softmax(pred, dim=-1)[0, 1].detach().cpu().numpy()
        print(id[0], pred_p)
        test_res.append({'SHXYDM': id[0], 'predict_prob': pred_p})
    df = pd.DataFrame(test_res)
    df.to_csv('test_res_fc%s.csv'%record_acc)


def fc_eval(model, e, ckpt=None):
    eval_loader = DataLoader(SVMDataSet(eval_f), batch_size=batch_size, shuffle=True)
    if ckpt is not None:
        # model = model_load(ckpt, model).cuda()
        model.load(ckpt)
    eval_acc = 0.0
    eval_num = 0
    pred_p_list = []
    gt_list = []
    model.eval()
    for batch, label, _ in eval_loader:
        data = torch.FloatTensor(batch).cuda()
        label = torch.LongTensor(label).cuda()
        pred = model(data)
        pred_p = F.softmax(pred, dim=-1)[:, 1]
        label = label.detach().cpu().numpy()
        gt_list.extend(list(label))
        pred_p_list.extend(list(pred_p.detach().cpu().numpy()))
        eval_acc += np.sum(torch.argmax(pred, dim=-1).detach().cpu().numpy() == label)
        eval_num += pred.shape[0]
    print("epoch %d, eval acc: %.3f" % (e, eval_acc / eval_num))
    test_auc = draw_roc("./vis/fc_model_eval%d.jpg" % e, gt_list, pred_p_list, title="eval")
    return test_auc

def fc_model_train():
    train_loader = DataLoader(SVMDataSet(train_f), batch_size=batch_size, shuffle=True)

    model = FCModel().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.99)
    best_auc = 0
    for e in range(EPOCH):
        train_acc = 0.0
        train_num = 0
        pred_p_list = []
        gt_list = []
        model.train()
        for batch, label, _ in train_loader:
            data = torch.FloatTensor(batch).cuda()
            label = torch.LongTensor(label).cuda()
            pred = model(data)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred_p = F.softmax(pred, dim=-1)[:,1]
            label = label.detach().cpu().numpy()
            gt_list.extend(list(label))
            pred_p_list.extend(list(pred_p.detach().cpu().numpy()))
            train_acc += np.sum(torch.argmax(pred, dim=-1).detach().cpu().numpy() == label)
            train_num += pred.shape[0]
        print("epoch %d, train acc:%.3f"%(e, train_acc/train_num))
        train_auc = draw_roc("./vis/fc_model_train%d.jpg"%e, gt_list, pred_p_list, title="train")
        test_auc = fc_eval(model, e)
        if min(train_auc, test_auc) > best_auc:
            torch.save(model.state_dict(), "./ckpt/fc_model_best.pth")
            print("saving best fc model")
            best_auc = min(train_auc, test_auc)
    print("the best auc = %.3f"%best_auc)


if __name__ == '__main__':
    # fc_model_train()
    # F:\yanjiusheng\2022春\赛道三附件\-\ckpt\按照roc最高算的不正确
    model = FCModel().cuda()
    ckpt = [
        "./ckpt/fc_model1_668_fill0.pth",
        "./ckpt/fc_model_682_fill0.pth"]
    # # FCtest(model, ckpt)
    fc_eval(model, 0, ckpt)