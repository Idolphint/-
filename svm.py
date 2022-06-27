from sklearn.svm import SVC
import numpy as np
from functions import draw_roc
import pickle


train_f = open("../data_jk/train_data.pkl", 'rb')
eval_f = open("../data_jk/eval_data.pkl", 'rb')
test_f = open("../data_jk/test.pkl", 'rb')


def svm_data_pre(tax_re, tax_pay):
    m, c = tax_re.shape
    r_m = 52
    r_year=1
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
    train_data = pickle.load(eval_f)
    eval_data = pickle.load(train_f)
    test_data = pickle.load(test_f)
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
