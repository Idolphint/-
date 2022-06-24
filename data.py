import pandas as pd
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch


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
        # self.code_enterprise_ratal_classes = ratal_class
        # self.code_enterprise_registration = code_regi
        self.HY_DM = [int(ratal_class), code_regi, int(hy_dm[-1]),
                      int(hy_dm[-2]), int(hy_dm[:-2]), ord(hyml_dm)-ord('A')]  # 行业代码、中类、大类、门类
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
        sort_inv = sorted(self.investor_info, key=lambda x: x[2])
        out_inv = np.zeros((4,2))
        for i in range(min(len(sort_inv), 4)):
            out_inv[i,0] = sort_inv[i][2]
            out_inv[i,1] = sort_inv[i][3] / 10000
        return out_inv

    def get_tax_return(self):
        key = sorted(self.tax_return.keys())
        out_return = np.zeros((len(key), 2))
        for i,k in enumerate(key):
            for item, info in self.tax_return[k].items():  # 类别应该分开来看吗？
                # 补0还是填平均值
                out_return[i, 0] += info[0] / 10000
                out_return[i, 1] = info[1]
        return out_return

    def get_tax_pay(self):
        out_pay = []
        for key, item in self.tax_pay.items():
            if item is None:
                continue
            for code_item, info in item.items():
                if key in self.tax_return and code_item in self.tax_return[key]:
                    tax_re = self.tax_return[key][code_item][0]
                else:
                    tax_re = 0
                    cnt=0
                    if key in self.tax_return:
                        for code_item_r, re in self.tax_return[key].items():
                            tax_re += re[0]
                            cnt+=1
                    if cnt > 0:
                        tax_re /= cnt
                out_pay.append([info[0]/10000, tax_re/10000, info[1], int(info[2])])
        if len(out_pay) == 0:
            out_pay.append([0,0,0,0])
        return np.array(out_pay)

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

        assert code_account+code_item[len(code_account):] == code_item
        code_item = code_item[len(code_account):]
        sales = float(sales)
        # 相对于公司成立日期的月数
        until_ddl = (return_ddl[1] - tax_return_date[1])*30 + return_ddl[2] - tax_return_date[2]
        rela_tax_begin = (tax_begin[0]-self.enterprise_opening_date[0])*12 +\
                         tax_begin[1]-self.enterprise_opening_date[1]
        if rela_tax_begin not in self.tax_return.keys():
            self.tax_return[rela_tax_begin] = {code_item:[sales, until_ddl, code_account]}
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
        until_ddl = (ddl[1]-pay_date[1])*30+(ddl[2]-pay_date[2])
        rela_tax_begin = (pay_begin[0]-self.enterprise_opening_date[0])*12 +\
                         pay_begin[1]-self.enterprise_opening_date[1]
        if rela_tax_begin not in self.tax_pay.keys() or self.tax_pay[rela_tax_begin] is None:
            self.tax_pay[rela_tax_begin] = {code_item:[ratal, until_ddl, tax_class, code_account]}
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


def init_dataset(mode = "train", eval_rate=0.9):
    dataset = {}
    base_info = pd.read_csv("../data_jk/%s_basic_info.csv"%mode, encoding='utf-8')
    tax_return = pd.read_csv("../data_jk/%s_tax_return_.csv"%mode, encoding='utf-8')
    tax_payment = pd.read_csv("../data_jk/%s_tax_payment_.csv" % mode, encoding='utf-8')
    tax_invest = pd.read_csv("../data_jk/%s_investor_info.csv" % mode, encoding='utf-8')
    base_info.fillna({'code_enterprise_ratal_classes':'0', 'HYML_DM':'A'}, inplace=True)
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
        eval_dataset = list(dataset.keys())[int(len(dataset.keys())*eval_rate):]
        eval_out = {ev: dataset[ev] for ev in eval_dataset}
        [dataset.pop(ev) for ev in eval_dataset]
        pickle.dump(dataset, open("../data_jk/train_data.pkl", 'wb'))
        pickle.dump(eval_out, open("../data_jk/eval_data.pkl", 'wb'))
    else:
        pickle.dump(dataset, open("../data_jk/%s.pkl"%mode, 'wb'))

    return dataset


if __name__ == '__main__':
    # init_dataset(mode='train')
    print("hello")