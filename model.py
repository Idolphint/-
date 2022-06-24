import torch
import torch.nn as nn

# 贝叶斯， svm

class TaxReturnLSTM(nn.Module):
    def __init__(self):
        super(TaxReturnLSTM, self).__init__()

        self.rnn = nn.LSTM(
            input_size=10,
            hidden_size=32,
            num_layers=2,
            dropout=0.5,
            batch_first=False
            # 输入输出数据格式是(seq_len, batch, feature)，
        )
        self.pre_proce = nn.Linear(2,10)
        self.fc1 = nn.Linear(32,32)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.hidden_out = nn.Linear(32, 2)  # 最后一个时序的输出接一个全连接层
        self.h_s = None
        self.h_c = None

    def forward(self, x):  # x是输入数据集
        x = self.sigmoid(self.pre_proce(x)).unsqueeze(1)
        r_out, (self.h_s, self.h_c) = self.rnn(x)  # 如果不导入h_s和h_c，默认每次都进行0初始化
        #  h_s和h_c表示每一个隐层的上一时间点输出值和输入细胞状态
        # h_s和h_c的格式均是(num_layers * num_directions, batch, HIDDEN_SIZE)
        # 如果是双向LSTM，num_directions是2，单向是1
        # print("look lstm", r_out[-1])
        x = self.relu(self.fc1(r_out[-1]))
        # print(x)
        output = self.hidden_out(x)
        # print(output)
        return output.squeeze(), x.squeeze()


class TaxPayModel(nn.Module):
    def __init__(self):
        super(TaxPayModel, self).__init__()
        self.fc1 = nn.Linear(in_features=4, out_features=32)
        self.add_pool_pay = nn.AdaptiveAvgPool1d(1)
        self.fc2 = nn.Linear(32,32)
        self.fc3 = nn.Linear(in_features=32, out_features=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout()

    def forward(self, tax_pay):
        x = self.sigmoid(self.drop(self.fc1(tax_pay)))
        x = self.add_pool_pay(x.transpose(1,0)).squeeze()
        x = self.relu(self.drop(self.fc2(x)))
        y = self.fc3(x)
        return y, x

class EmbeddingClassify(nn.Module):
    def __init__(self):
        super(EmbeddingClassify, self).__init__()
        self.emd_hy_layer1 = nn.Embedding(10, 4)
        self.emd_hy_layer2 = nn.Embedding(901, 5)
        self.emd_hy_layer3 = nn.Embedding(10, 4)
        self.emd_hy_layer4 = nn.Embedding(10, 4)
        self.emd_hy_layer5 = nn.Embedding(100, 4)
        self.emd_hy_layer6 = nn.Embedding(20, 5)
        self.hy_fc = nn.Linear(26, 10)
        self.emb_inv_fc1 = nn.Linear(8, 8)
        self.emb_inv_fc2 = nn.Linear(8,32)

        self.fc_out1 = nn.Linear(in_features=106, out_features=32)
        self.fc_out2 = nn.Linear(in_features=32, out_features=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0.5)

    def forward(self, hy, inv, re_out, tax_pay):
        hy0 = self.emd_hy_layer1(hy[0])
        hy1 = self.emd_hy_layer2(hy[1])
        hy2 = self.emd_hy_layer3(hy[2])
        hy3 = self.emd_hy_layer4(hy[3])
        hy4 = self.emd_hy_layer5(hy[4])
        hy5 = self.emd_hy_layer6(hy[5])
        hy = torch.cat([hy0,hy1,hy2,hy3,hy4,hy5], dim=-1)
        hy = self.relu(self.drop(self.hy_fc(hy)))
        inv = self.sigmoid(self.drop(self.emb_inv_fc1(inv)))
        inv = self.relu(self.drop(self.emb_inv_fc2(inv)))
        x = torch.cat([re_out, tax_pay, hy, inv], dim=-1)
        x = self.relu(self.fc_out1(x))
        x = self.fc_out2(x)
        return x

