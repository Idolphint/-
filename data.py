import random
from imblearn.over_sampling import RandomOverSampler
from scipy import signal
import pandas as pd
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
code_regi_redict = {}
hy_dm_redict = {}


import random
from imblearn.over_sampling import RandomOverSampler
from scipy import signal
import pandas as pd
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
code_regi_redict = {}
hy_dm_redict = {}
code_ret = {}
code_pay = {}

mode = "train"
base_info = pd.read_csv("../data_jk/%s_basic_info.csv" % mode, encoding='utf-8')
tax_return = pd.read_csv("../data_jk/%s_tax_return_.csv" % mode, encoding='utf-8')
tax_payment = pd.read_csv("../data_jk/%s_tax_payment_.csv" % mode, encoding='utf-8')
tax_invest = pd.read_csv("../data_jk/%s_investor_info.csv" % mode, encoding='utf-8')
base_info.fillna({'code_enterprise_ratal_classes': '0', 'HYML_DM': 'A'}, inplace=True)


class Company(object):
    def __init__(self, data):
        id, open_date, ratal_class, code_regi, hy_dm, hyzl_dm, hydl_dm, hyml_dm = data
        open_day = open_date.split(' ')[0]
        open_day = open_day.split('-')
        open_date = [int(open_day[0]), int(open_day[1]), int(open_day[2])]
        hy_dm = str(hy_dm)
        # print(ratal_class)
        self.id = id
        self.enterprise_opening_date = open_date
        self.max_info_date = open_date
        if code_regi not in code_regi_redict.keys():
            code_regi_redict[code_regi] = len(code_regi_redict.keys())
        if hy_dm[:-2] not in hy_dm_redict.keys():
            hy_dm_redict[hy_dm[:-2]] = len(hy_dm_redict.keys())
        self.HY_DM = [int(ratal_class), code_regi_redict[code_regi], int(hy_dm[-1]),
                      int(hy_dm[-2]), hy_dm_redict[hy_dm[:-2]], ord(hyml_dm) - ord('A')]  # 行业代码、中类、大类、门类

        self.tax_ret = {}
        self.tax_pay = {}
        self.investor_info = []
        self.label = -1
        self.sum_sales = 0.0
        self.sum_ratal = 0.0
        self.sum_inv = 0.0

    def add_tax_return(self, data):
        id, tax_return_date, return_ddl, code_account, code_item, tax_begin, tax_end, sales = data
        return_day = tax_return_date.split("-")
        tax_return_date = [int(i) for i in return_day]
        if tax_return_date > self.max_info_date:
            self.max_info_date = tax_return_date
        code_account = str(code_account)
        if code_account not in code_ret.keys():
            code_ret[code_account] = len(code_ret.keys())
        tax_begin_day = tax_begin.split('-')
        tax_begin = [int(i) for i in tax_begin_day]
        rela_tax_begin = (tax_begin[0] - self.enterprise_opening_date[0]) * 12 + \
                         tax_begin[1] - self.enterprise_opening_date[1]
        if rela_tax_begin not in self.tax_ret.keys():
            self.tax_ret[rela_tax_begin] = [0]*5
        self.tax_ret[rela_tax_begin][code_ret[code_account]] += sales
        self.sum_sales += sales

    def add_tax_payment(self, data):
        id, ddl, pay_date, pay_begin, pay_end, code_account, code_item, tax_class, ratal = data
        pay_day = pay_date.split("-")
        pay_date = [int(i) for i in pay_day]
        tax_class = str(tax_class)
        if pay_date > self.max_info_date:
            self.max_info_date = pay_date
        code_account = str(code_account)
        if code_account not in code_pay.keys():
            code_pay[code_account] = len(code_pay.keys())
        rela_tax_begin = (pay_date[0] - self.enterprise_opening_date[0]) * 12 + \
                         pay_date[1] - self.enterprise_opening_date[1]
        if rela_tax_begin not in self.tax_pay.keys():
            self.tax_pay[rela_tax_begin] = [0]*6
        self.tax_pay[rela_tax_begin][code_pay[code_account]] += ratal
        self.sum_ratal += ratal

    def add_investor_info(self, data):
        id, investor_id, invest_class, invest_rate, invest_amount = data
        self.sum_inv += invest_amount

    def add_label(self, data):
        self.label = int(data['label'])

    def get_tax_data(self):
        full_mon = np.array(sorted(list(set(self.tax_pay.keys()).union(set(self.tax_ret.keys())))))
        full_mon_1 = (full_mon - min(full_mon))/(max(full_mon)-min(full_mon))

        full_data = np.zeros([len(full_mon),11])
        for i, a in enumerate(full_mon):
            if a in self.tax_ret.keys():
                full_data[i,0:5] = self.tax_ret[a]
            if a in self.tax_pay.keys():
                full_data[i,5:] = self.tax_pay[a]
        x = np.linspace(0,1,32)
        out_data = np.zeros([32,11])
        for i in range(11):
            out_data[:,i] = np.interp(x, full_mon_1, full_data[:,i])
        # print(out_data.shape, out_data)
        return out_data

    def get_train_data(self):
        out = list(self.get_tax_data().reshape(-1))
        out.extend([self.sum_inv, self.sum_sales, self.sum_ratal])
        # out = out.append(self.sum_sales)
        # out = out.append(self.sum_ratal)

        return out


class Company1(object):
    def __init__(self, data):
        id, open_date, ratal_class, code_regi, hy_dm, hyzl_dm, hydl_dm, hyml_dm = data
        open_day = open_date.split(' ')[0]
        open_day = open_day.split('-')
        open_date = [int(open_day[0]), int(open_day[1]), int(open_day[2])]
        hy_dm = str(hy_dm)
        # print(ratal_class)
        self.id = id
        self.enterprise_opening_date = open_date
        self.max_info_date = open_date
        if code_regi not in code_regi_redict.keys():
            code_regi_redict[code_regi] = len(code_regi_redict.keys())
        if hy_dm[:-2] not in hy_dm_redict.keys():
            hy_dm_redict[hy_dm[:-2]] = len(hy_dm_redict.keys())
        self.HY_DM = [int(ratal_class), code_regi_redict[code_regi], int(hy_dm[-1]),
                      int(hy_dm[-2]), hy_dm_redict[hy_dm[:-2]], ord(hyml_dm) - ord('A')]  # 行业代码、中类、大类、门类

        self.tax_return = {}
        self.tax_pay = {}
        self.investor_info = []
        self.label = -1

    def get_train_data(self):
        hy = self.HY_DM
        inv = self.get_four_investor().reshape(-1)
        tax_re = self.get_tax_return()
        tax_pay = self.get_tax_pay()
        # print(hy, inv, tax_re, tax_pay)
        # print(ord(hy[5]) - ord('A'))
        hy = torch.IntTensor(hy)
        inv = torch.Tensor(inv)
        tax_re = torch.Tensor(tax_re)
        tax_pay = torch.Tensor(tax_pay)

        return hy, inv, tax_re, tax_pay

    # 数据归一化
    def get_four_investor(self):
        sort_inv = sorted(self.investor_info, key=lambda x: x[2], reverse=True)
        out_inv = np.zeros((4, 2))
        for i in range(min(len(sort_inv), 4)):
            out_inv[i, 0] = sort_inv[i][2]
            out_inv[i, 1] = sort_inv[i][3]
        return out_inv

    def get_tax_return(self):
        key = sorted(self.tax_return.keys())
        out_return = []
        item_dict = {}
        for i, k in enumerate(key): #遍历月份
            for item, info in self.tax_return[k].items():
                if item not in item_dict.keys():
                    item_dict[item] = [info[0]]
                else:
                    item_dict[item].append(info[0])
        out_ret = np.zeros((22, len(item_dict.keys())))
        for i, k in enumerate(item_dict.keys()):
            out_ret[:,i] = signal.resample(item_dict[k], 22)
        # for i, k in enumerate(key):
        #     new_month = [0] * len(item_dict.keys())
        #     for item, info in self.tax_return[k].items():  # 类别应该分开来看吗？
        #         if item not in item_dict.keys():
        #             item_dict[item] = [len(item_dict.keys()), 1]
        #             new_month.append(info[0])
        #         else:
        #             new_month[item_dict[item][0]] = info[0]
        #             item_dict[item][1] += 1
        #     out_return.append(new_month)
            #     # 补0还是填平均值
            #     out_return[i, 0] += info[0] / 10000
            #     out_return[i, 1] = info[1]
            # if out_return[i,0] < 0:
            #     out_return[i,0] = -1*np.log(abs(out_return[i,0]))
            # else:
            #     out_return[i,0] = np.log(out_return[i,0]+1)
        # item_sort = sorted(list(item_dict.values()), key=lambda x: x[1], reverse=True)
        # item_sort = np.array(item_sort)[:,0]
        # print("item_sort", item_sort)
        # # smo = RandomOverSampler(random_state=35, sampling_strategy='auto')
        # for i, m in enumerate(out_return):
        #     m += [0] * (len(item_dict.keys()) - len(m))
        #     for j in range(len(item_sort)):
        #         out_ret[i,j] = m[item_sort[j]]

        return out_ret

    def get_tax_pay(self):
        out_pay = {}
        for key, item in self.tax_pay.items():
            if item is None:
                continue
            for code_item, info in item.items():
                if key in self.tax_return and code_item in self.tax_return[key]:
                    tax_re = self.tax_return[key][code_item][0]
                else:
                    tax_re = 0
                if code_item not in out_pay:
                    out_pay[code_item] = []  # 交税，报税，税务类别
                out_pay[code_item].append([info[0], tax_re, int(info[2])])
        return out_pay

    def add_tax_return(self, data):
        id, tax_return_date, return_ddl, code_account, code_item, tax_begin, tax_end, sales = data
        return_day = tax_return_date.split("-")
        ddl = return_ddl.split("-")
        tax_begin_day = tax_begin.split('-')
        tax_return_date = [int(i) for i in return_day]
        return_ddl = [int(i) for i in ddl]
        tax_begin = [int(i) for i in tax_begin_day]
        code_account = str(code_account)
        code_item = str(code_item)

        assert code_account + code_item[len(code_account):] == code_item
        code_item = code_item[len(code_account):]
        sales = float(sales)
        # 相对于公司成立日期的月数
        until_ddl = (return_ddl[1] - tax_return_date[1]) * 30 + return_ddl[2] - tax_return_date[2]
        rela_tax_begin = (tax_begin[0] - self.enterprise_opening_date[0]) * 12 + \
                         tax_begin[1] - self.enterprise_opening_date[1]
        if rela_tax_begin not in self.tax_return.keys():
            self.tax_return[rela_tax_begin] = {code_item: [sales, until_ddl, code_account]}
        else:
            self.tax_return[rela_tax_begin][code_item] = [sales, until_ddl, code_account]

    def add_tax_payment(self, data):
        id, ddl, pay_date, pay_begin, pay_end, code_account, code_item, tax_class, ratal = data
        ddl_day = ddl.split("-")
        pay_day = pay_date.split("-")
        pay_begin_day = pay_begin.split("-")
        ddl = [int(i) for i in ddl_day]
        pay_date = [int(i) for i in pay_day]
        pay_begin = [int(i) for i in pay_begin_day]
        code_item = str(code_item)
        tax_class = str(tax_class)
        code_account = str(code_account)
        assert code_account + code_item[len(code_account):] == code_item
        code_item = code_item[len(code_account):]
        until_ddl = (ddl[1] - pay_date[1]) * 30 + (ddl[2] - pay_date[2])
        rela_tax_begin = (pay_begin[0] - self.enterprise_opening_date[0]) * 12 + \
                         pay_begin[1] - self.enterprise_opening_date[1]
        if rela_tax_begin not in self.tax_pay.keys() or self.tax_pay[rela_tax_begin] is None:
            self.tax_pay[rela_tax_begin] = {code_item: [ratal, until_ddl, tax_class, code_account]}
        else:
            self.tax_pay[rela_tax_begin][code_item] = [ratal, until_ddl, tax_class, code_account]

    def add_investor_info(self, data):
        id, investor_id, invest_class, invest_rate, invest_amount = data
        self.investor_info.append([investor_id, invest_class, invest_rate, invest_amount])

    def add_label(self, data):
        self.label = int(data['label'])


class ComDataset(Dataset):
    def __init__(self, data_f):
        super(ComDataset, self).__init__()
        self.data = list(pickle.load(data_f).values())

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def init_dataset(mode="train", eval_rate=0.7):
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    for mode in ["train", "test"]:
        dataset = {}
        base_info = pd.read_csv("../data_jk/%s_basic_info.csv" % mode, encoding='utf-8')
        tax_return = pd.read_csv("../data_jk/%s_tax_return_.csv" % mode, encoding='utf-8')
        tax_payment = pd.read_csv("../data_jk/%s_tax_payment_.csv" % mode, encoding='utf-8')
        tax_invest = pd.read_csv("../data_jk/%s_investor_info.csv" % mode, encoding='utf-8')
        base_info.fillna({'code_enterprise_ratal_classes': '0', 'HYML_DM': 'A'}, inplace=True)
        # tax_return['sales'] += 0.01
        # tax_payment['ratal'] += 0.01
        for index, row in base_info.iterrows():
            id = row['ID']
            dataset[id] = Company(row)
        for index, row in tax_return.iterrows():
            id = row['ID']
            comp = dataset[id]
            comp.add_tax_return(row)
            dataset[id] = comp
        for index, row in tax_payment.iterrows():
            id = row['ID']
            comp = dataset[id]
            comp.add_tax_payment(row)
            dataset[id] = comp
        for index, row in tax_invest.iterrows():
            id = row['ID']
            comp = dataset[id]
            comp.add_investor_info(row)
            dataset[id] = comp
        if mode == 'train':
            label = pd.read_csv("../data_jk/%s_label.csv" % mode, encoding='utf-8')
            for index, row in label.iterrows():
                id = row['SHXYDM']
                comp = dataset[id]
                comp.add_label(row)
                dataset[id] = comp
            eval_dataset = random.sample(list(dataset.keys()), int(len(dataset.keys()) * eval_rate))
            eval_out = {ev: dataset[ev] for ev in eval_dataset}
            [dataset.pop(ev) for ev in eval_dataset]
            pickle.dump(dataset, open("../data_jk/train_dataR_yty.pkl", 'wb'))
            pickle.dump(eval_out, open("../data_jk/eval_dataR_yty.pkl", 'wb'))
        else:
            pickle.dump(dataset, open("../data_jk/%s_yty.pkl" % mode, 'wb'))

    return dataset


def svm_data_pre(tax_re, tax_pay):
    m, c = tax_re.shape
    # print(tax_re.shape, tax_re)
    r_m = 50
    new_tax_re = [0]*r_m
    new_tax_re[0:min(m*c,r_m)] = list(tax_re.reshape(m * c)[:r_m])
    # r_year=0
    # new_tax_re = [0]*r_m
    # j=r_year*1
    # new_tax_re[0:min(r_year, m)] = tax_re[0:min(r_year,m), 0]
    # # if c >= 2:
    # #     new_tax_re[r_year:r_year+min(r_year,m)] = tax_re[0:min(r_year,m), 1]
    # for ci in range(0,c):
    #     for mi in range(m):
    #         if(tax_re[mi,ci]!=0):
    #             new_tax_re[j] = tax_re[mi,ci]
    #             j+=1
    #             if j >=r_m:
    #                 break
    #     if j >= r_m:
    #         break

    # 处理交税
    p_m = 18
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


class SVMDataSet(Dataset):
    def __init__(self, data_dir):
        data_f = open(data_dir, 'rb')
        in_data = pickle.load(data_f)
        self.svm_train_x = []
        self.svm_train_y = []
        self.svm_id = []
        for id, comp in in_data.items():
            # hy_dm = comp.HY_DM  # 6位行业编码
            # inv = comp.get_four_investor().reshape(-1)  # 8位投资信息
            # tax_re = comp.get_tax_return()  # month * 商品种类， 选取2个商品种类，12个月的数据
            # tax_pay = comp.get_tax_pay()  # 商品种类： 各个月份， 选取2个商品种类，2个月的数据？
            # tax_re, tax_pay = svm_data_pre(tax_re, tax_pay)
            # new_info = []
            # new_info.extend(tax_re)
            # new_info.extend(tax_pay)
            # new_info.extend(hy_dm)
            new_info = comp.get_train_data()
            # new_info.extend(inv)
            # assert len(new_info) == 87
            self.svm_train_x.append(new_info)
            label = comp.label
            self.svm_train_y.append(label)
            self.svm_id.append(id)
        self.svm_train_x = np.array(self.svm_train_x, dtype=np.float32)
        self.svm_train_y = np.array(self.svm_train_y, dtype=np.int64)

    def __getitem__(self, item):
        # if isinstance(self.svm_train_x, list):
        #     self.svm_train_x = np.array(self.svm_train_x, dtype=np.float32)
        #     self.svm_train_y = np.array(self.svm_train_y, dtype=np.int64)
        x = self.svm_train_x[item]
        y = self.svm_train_y[item]
        z = self.svm_id[item]
        return x,y,z

    def __len__(self):
        return len(self.svm_train_x)


def see_data_feature():
    # 一个公司平均交9次税
    # 一个公司平均2个投资人
    # 报税信息，平均有22个月，平均每家公司有2.3个商品
    # 最早的公司成立时间为1975年1月5日
    test_data = pickle.load(open("../data_jk/train_dataR099.pkl", 'rb'))
    el_re_date = None
    max_date = None
    return_month = 0
    return_item = 0
    return_cnt = 0
    for id, comp in test_data.items():
        data = comp.enterprise_opening_date
        if el_re_date is None:
            el_re_date = data
            max_date = data
        elif data<el_re_date:
            el_re_date = data
        elif data>max_date:
            max_date = data
    print(el_re_date, max_date)
    # print("平均return month个数：", return_month/return_cnt, "平均商品个数",return_item/return_cnt)


if __name__ == '__main__':
    init_dataset(mode='train', eval_rate=0.3)
    # see_data_feature()
    # print("hello")
