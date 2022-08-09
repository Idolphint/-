from data import SVMDataSet, Company
import pickle
from math import log
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, mean_squared_error
from sklearn.model_selection import train_test_split
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import chi2

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 主要是对特征进行分箱和分析
# 第一步就是将特征列出来，并且与结果对应
# 筛选iv在0.02-0.5的特征
# 预计可以开发的特征有，公司成立时间，有交税数据的月数，交税平均值，带空缺的交税平均值
# 用svm拟合的交税变化的参数
# 平均投资金额，交税最大值，交税最小值……
# boost和随机森林真的没有输出吗？？


def make_feature(df):
    for i in range(11):  # 11个商品项目 32个月的税务总和
        fea_name = ['fea_%d' % (j * 11 + i) for j in range(32)]
        df['item%d_sum' % i] = df[fea_name].sum(axis=1)
    # 32个月内全部商品，交税，纳税信息的和
    df['sum_ratal_32month'] = df[['fea_%d' % j for j in range(352)]].sum(axis=1)
    # print(list(df))
    df['sum_tax_ret'] = df.apply(lambda x:
                                 x['item0_sum'] + x['item1_sum'] + x['item2_sum'] +
                                 x['item3_sum'] + x['item4_sum'], axis=1)
    df['sum_tax_pay'] = df.apply(
        lambda x: x['item5_sum'] + x['item6_sum'] + x['item7_sum'] + x['item8_sum'] + x['item9_sum'] + x['item10_sum'],
        axis=1)
    # 交税/报税的比例
    df['ratal_rate'] = df.apply(
        lambda x: min(x['sum_tax_pay'] / (x['sum_tax_ret'] + 1), 1), axis=1
    )
    # 以4个月为一阶段计算这段时间交税，报税占的比例
    for i in range(32 // 4):
        buf_ret = df[['fea_%d' % (j * 11 + k) for j in range(
            i * 4, (i + 1) * 4) for k in range(0, 5)]].sum(axis=1)
        buf_pay = df[['fea_%d' % (j * 11 + k) for j in range(
            i * 4, (i + 1) * 4) for k in range(5, 11)]].sum(axis=1)
        df['stage_%d_ret_rate' % i] = buf_ret / df['sum_tax_ret']
        df['stage_%d_pay_rate' % i] = buf_pay / df['sum_tax_pay']
    # 每个月的交税总额
    for i in range(32):
        df['month%d_ret' % i] = df[['fea_%d' % (i * 11 + j) for j in range(0, 5)]].sum(axis=1)
        df['month%d_pay' % i] = df[['fea_%d' % (i * 11 + j) for j in range(5, 11)]].sum(axis=1)
    # 交/报税数据不为0的月数
    # df.apply(lambda x: print([df['month%d_ret' % i] > 0 for i in range(32)]), axis=1)
    ret_non0_month = []
    df_buf = df[['month%d_ret' % i for i in range(32)]]
    df['ret_non0_month'] = (df_buf > 0).astype(int).sum(axis=1)
    df_buf = df[['month%d_pay' % i for i in range(32)]]
    df['pay_non0_month'] = (df_buf > 0).astype(int).sum(axis=1)
    # 平均投资金额, 投资人数量
    # 已经在数据源头解决
    # 最大交税额和最小交税额
    df['min_tax_pay'] = df[['month%d_pay' % i for i in range(32)]].min(axis=1)
    df['max_tax_pay'] = df[['month%d_pay' % i for i in range(32)]].max(axis=1)
    return df


def from_pkl2_csv(data_dir):
    data_dir = "../data_final/all_data.pkl"
    fea_list = ['fea_%d' % i for i in range(352)]
    fea_list.extend(['sum_inv', 'sum_sales', 'sum_ratal', 'inv_mean', 'inv_n'])
    # eval_data = SVMDataSet(data_dir)
    # print(eval_data.svm_train_x.shape, len(fea_list))
    # df = pd.DataFrame(eval_data.svm_train_x, columns=fea_list)
    df = pd.read_csv('../data_final/all_data2.csv')
    # df2.insert(df2.shape[1], 'inv_mean', df['inv_mean'])
    # df2.insert(df2.shape[1], 'inv_n', df['inv_n'])
    # df2.to_csv('../data_final/all_data2.csv')
    df = make_feature(df)
    df.to_csv('../data_final/all_data.csv')


def woe_iv_demo(df):
    fea_list = ['fea_%d' % i for i in range(352)]
    fea_list.extend(['sum_inv', 'sum_sales', 'sum_ratal', 'inv_mean', 'inv_n'])
    totalG_B = df.groupby(['label'])['label'].count()  # 计算正负样本多少个
    good_T, bad_T = totalG_B[0], totalG_B[1]
    inv_bins = [-1, 0, 1e5, 1e6, 5e6, 1e7, 1e8, 1e13]
    inv_label = ['0元', '10w以下', '10-100w', '100w-500w', '500w-1kw', '1kw-1亿', '1亿以上']

    sales_bins = [0, 1e5, 1e6, 5e6, 1e7, 3e7, 5e7, 1e8, 1e13]
    sales_label = ['10w以下', '10-100w', '100w-300w', '300w-500w', '500w-1kw', '1kw-5kw', '5kw-1亿', '1亿以上']

    ratal_bins = [0, 1e4, 5e4, 1e5, 5e5, 1e6, 1e7, 1e13]
    ratal_label = ['1w以下', '1w-5w', '5w-10w', '10w-50w', '50w-100w', '100w-1kw', '1kw以上']
    # 预处理，查看如何分箱
    d0 = df['sum_ratal']
    # x0 = np.linspace(0, d0.shape[0], d0.shape[0])
    plt.hist(d0)
    plt.show()
    # print(max())
    # 逐个计算特征分箱
    # 共5000个数据，最小分箱不宜小于5%，为250个
    iv_list = []
    for fea in fea_list[-3:]:
        fea_fix = fea[4:]
        d0 = df[fea]
        sel = pd.cut(df[fea],
                     eval('%s_bins' % fea_fix),
                     labels=eval('%s_label' % fea_fix))
        df.insert(df.shape[1], '%s_cut' % fea, sel)
        var1 = df.groupby(['%s_cut' % fea, 'label'])['label'].count()
        # print(var1)
        woe_inv = [0] * len(eval('%s_label' % fea_fix))
        sample_rate = [0] * len(eval('%s_label' % fea_fix))
        bad_rate = [0] * len(eval('%s_label' % fea_fix))
        iv = 0
        for i, invl in enumerate(eval('%s_label' % fea_fix)):
            woe_inv[i] = math.log((var1[invl][1] + 0.5 / bad_T) / (var1[invl][0] + 0.5 / good_T), math.e)
            sample_rate[i] = (var1[invl][1] + var1[invl][0]) / 5000
            bad_rate[i] = var1[invl][1] / (var1[invl][1] + var1[invl][0])
            iv += woe_inv[i] * ((var1[invl][1] / bad_T) - (var1[invl][0] / good_T))
        iv_list.append(iv)
        fig = plt.figure(figsize=(8, 5))
        ax1 = fig.add_subplot(111)
        ax1.bar(eval('%s_label' % fea_fix), sample_rate)
        ax1.set_ylabel('箱内样本数量占比')
        ax2 = ax1.twinx()
        ax2.plot(eval('%s_label' % fea_fix), bad_rate, linewidth=2, color='red')
        ax2.scatter(eval('%s_label' % fea_fix), bad_rate, color='red')
        ax2.set_ylabel('箱内坏样本占比')
        plt.title('特征%s箱线图' % fea)
        plt.legend(['箱内样本数量占比', '箱内坏样本占比'])
        plt.savefig('箱线图%d份_%s.jpg' % (len(eval('%s_label' % fea_fix)), fea))
        # plt.show()

    # print(iv_list)


def optimal_binning_boundary(x: pd.Series, y: pd.Series, nan: float = -999.) -> list:
    '''
        利用决策树获得最优分箱的边界值列表
    '''
    boundary = []  # 待return的分箱边界值列表

    x = x.fillna(nan).values  # 填充缺失值
    y = y.values

    clf = DecisionTreeClassifier(criterion='entropy',  # “信息熵”最小化准则划分
                                 max_leaf_nodes=10,  # 最大叶子节点数
                                 min_samples_leaf=0.05)  # 叶子节点样本数量最小占比

    clf.fit(x.reshape(-1, 1), y)  # 训练决策树

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    threshold = clf.tree_.threshold

    for i in range(n_nodes):
        if children_left[i] != children_right[i]:  # 获得决策树节点上的划分边界值
            boundary.append(threshold[i])

    boundary.sort()

    min_x = min(x)
    max_x = max(x) + 0.1  # +0.1是为了考虑后续groupby操作时，能包含特征最大值的样本
    boundary = [min_x] + boundary + [max_x]
    # print(boundary)
    return boundary


# 基于CART的最优分箱，也就是选择基尼指数下降最大的点
def get_var_median(data, var):
    """ 得到指定连续变量的所有元素的中位数列表
    Args:
        data: DataFrame，拟操作的数据集
        var: String，拟分箱的连续型变量名称
    Returns:
        关于连续变量的所有元素的中位列表，List
    """
    var_value_list = sorted(list(np.unique(data[var])))
    var_median_list = []
    # 可以假设这里data长度最少100,否则低于了划分条件
    if len(data[var]) < 100:
        return [var_value_list[int(len(var_value_list) * 0.5)]]
    idx = [0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875]
    for i in idx:
        var_median = var_value_list[int(len(var_value_list) * i)]
        var_median_list.append(var_median)
    return var_median_list


def calculate_gini(y):
    """ 计算基尼指数
    Args:
        y: Array，待计算数据的target，即0和1的数组
    Returns:
        基尼指数，float
    """
    # 将数组转化为列表
    y = y.tolist()
    probs = [y.count(i) / len(y) for i in np.unique(y)]
    gini = sum([p * (1 - p) for p in probs])
    return gini


def get_cart_split_point(data, var, target, min_sample):
    """ 获得最优的二值划分点（即基尼指数下降最大的点）
    Args:
        data: DataFrame，拟操作的数据集
        var: String，拟分箱的连续型变量名称
        target: String，Y列名称
        min_sample: int，分箱的最小数据样本，也就是数据量至少达到多少才需要去分箱，一般作用在开头或者结尾处的分箱点

    Returns:
        BestSplit_Point: 返回本次迭代的最优划分点，float
        BestSplit_Position: 返回最优划分点的位置，最左边为0，最右边为1，float
    """

    # 初始化
    Gini = calculate_gini(data[target].values)
    Best_Gini = 0.0
    BestSplit_Point = -99999
    BestSplit_Position = 0.0
    median_list = get_var_median(data, var)  # 获取当前数据集指定元素的所有中位数列表

    for i in range(len(median_list)):
        left = data[data[var] < median_list[i]]
        right = data[data[var] > median_list[i]]

        # 如果切分后的数据量少于指定阈值，跳出本次分箱计算
        if len(left) < min_sample or len(right) < min_sample:
            continue

        Left_Gini = calculate_gini(left[target].values)
        Right_Gini = calculate_gini(right[target].values)
        Left_Ratio = len(left) / len(data)
        Right_Ratio = len(right) / len(data)

        Temp_Gini = Gini - (Left_Gini * Left_Ratio + Right_Gini * Right_Ratio)
        if Temp_Gini > Best_Gini:
            Best_Gini = Temp_Gini
            BestSplit_Point = median_list[i]
            # 获取切分点的位置，最左边为0，最右边为1
            if len(median_list) > 1:
                BestSplit_Position = i / (len(median_list) - 1)
            else:
                BestSplit_Position = i / len(len(median_list))
        else:
            continue
    Gini = Gini - Best_Gini
    # print("最优切分点：", BestSplit_Point)
    return BestSplit_Point, BestSplit_Position


def get_cart_bincut(data, var, target, leaf_stop_percent=0.05):
    """ 计算最优分箱切分点
    Args:
        data: DataFrame，拟操作的数据集
        var: String，拟分箱的连续型变量名称
        target: String，Y列名称
        leaf_stop_percent: 叶子节点占比，作为停止条件，默认5%

    Returns:
        best_bincut: 最优的切分点列表，List
    """
    if var not in list(data):
        print("var not in data", var)
        return [0, 1]
    min_sample = len(data) * leaf_stop_percent
    best_bincut = []

    def cutting_data(data, var, target, min_sample, best_bincut):
        split_point, position = get_cart_split_point(data, var, target, min_sample)

        if split_point != -99999:
            best_bincut.append(split_point)

        # 根据最优切分点切分数据集，并对切分后的数据集递归计算切分点，直到满足停止条件
        # print("本次分箱的值域范围为{0} ~ {1}".format(data[var].min(), data[var].max()))
        left = data[data[var] < split_point]
        right = data[data[var] > split_point]

        # 当切分后的数据集仍大于最小数据样本要求，则继续切分
        if len(left) >= min_sample and position not in [0.0, 1.0]:
            cutting_data(left, var, target, min_sample, best_bincut)
        else:
            pass
        if len(right) >= min_sample and position not in [0.0, 1.0]:
            cutting_data(right, var, target, min_sample, best_bincut)
        else:
            pass
        return best_bincut

    best_bincut = cutting_data(data, var, target, min_sample, best_bincut)

    # 把切分点补上头尾
    best_bincut.append(data[var].min())
    best_bincut.append(data[var].max())
    best_bincut_set = set(best_bincut)
    best_bincut = list(best_bincut_set)

    best_bincut.remove(data[var].min())
    best_bincut.append(data[var].min() - 1)
    # 排序切分点
    best_bincut.sort()

    return best_bincut


# 基于卡方检验的最优化分
def calculate_chi(freq_array):
    """ 计算卡方值
    Args:
        freq_array: Array，待计算卡方值的二维数组，频数统计结果
    Returns:
        卡方值，float
    """
    # 检查是否为二维数组
    assert (freq_array.ndim == 2)

    # 计算每列的频数之和
    col_nums = freq_array.sum(axis=0)
    # 计算每行的频数之和
    row_nums = freq_array.sum(axis=1)
    # 计算总频数
    nums = freq_array.sum()
    # 计算期望频数
    E_nums = np.ones(freq_array.shape) * col_nums / nums
    E_nums = (E_nums.T * row_nums).T
    # 计算卡方值
    tmp_v = (freq_array - E_nums) ** 2 / E_nums
    # 如果期望频数为0，则计算结果记为0
    tmp_v[E_nums == 0] = 0
    chi_v = tmp_v.sum()
    return chi_v


def get_chimerge_bincut(data, var, target, max_group=None, chi_threshold=None):
    """ 计算卡方分箱的最优分箱点
    Args:
        data: DataFrame，待计算卡方分箱最优切分点列表的数据集
        var: 待计算的连续型变量名称
        target: 待计算的目标列Y的名称
        max_group: 最大的分箱数量（因为卡方分箱实际上是合并箱体的过程，需要限制下最大可以保留的分箱数量）
        chi_threshold: 卡方阈值，如果没有指定max_group，我们默认选择类别数量-1，置信度95%来设置阈值
        如果不知道卡方阈值怎么取，可以生成卡方表来看看，代码如下：
        import pandas as pd
        import numpy as np
        from scipy.stats import chi2
        p = [0.995, 0.99, 0.975, 0.95, 0.9, 0.5, 0.1, 0.05, 0.025, 0.01, 0.005]
        pd.DataFrame(np.array([chi2.isf(p, df=i) for i in range(1,10)]), columns=p, index=list(range(1,10)))
    Returns:
        最优切分点列表，List
    """
    df_buf = sorted(data[var])
    tmp_bin = np.unique([df_buf[i * 20] for i in range(len(df_buf) // 20)])
    tmp_bin = tmp_bin[np.isfinite(tmp_bin)]
    sel = pd.cut(data[var], tmp_bin, duplicates='drop', labels=tmp_bin[1:], right=False)
    freq_df = pd.crosstab(index=sel, columns=data[target])
    # 转化为二维数组
    freq_array = freq_df.values

    # 初始化箱体，每个元素单独一组
    best_bincut = freq_df.index.values

    # 初始化阈值 chi_threshold，如果没有指定 chi_threshold，则默认选择target数量-1，置信度95%来设置阈值
    if max_group is None:
        if chi_threshold is None:
            chi_threshold = chi2.isf(0.05, df=freq_array.shape[-1])

    # 开始迭代
    while True:
        min_chi = None
        min_idx = None
        for i in range(len(freq_array) - 1):
            # 两两计算相邻两组的卡方值，得到最小卡方值的两组
            v = calculate_chi(freq_array[i: i + 2])
            if min_chi is None or min_chi > v:
                min_chi = v
                min_idx = i

        # 是否继续迭代条件判断
        # 条件1：当前箱体数仍大于 最大分箱数量阈值
        # 条件2：当前最小卡方值仍小于制定卡方阈值
        if (max_group is not None and max_group < len(freq_array)) or (
                chi_threshold is not None and min_chi < chi_threshold):
            tmp = freq_array[min_idx] + freq_array[min_idx + 1]
            freq_array[min_idx] = tmp
            freq_array = np.delete(freq_array, min_idx + 1, 0)
            best_bincut = np.delete(best_bincut, min_idx + 1, 0)
        else:
            break

    # 把切分点补上头尾
    best_bincut = best_bincut.tolist()
    best_bincut.append(data[var].min())
    best_bincut.append(data[var].max())
    best_bincut_set = set(best_bincut)
    best_bincut = list(best_bincut_set)

    best_bincut.remove(data[var].min())
    best_bincut.append(data[var].min() - 1)
    # 排序切分点
    best_bincut.sort()

    return best_bincut


def woe_iv(df, fea, fea_bins=None, fea_label=None, draw=None):
    '''
    计算分箱后的iv值，如不指定分箱方式，则采用最高iv法分箱
    :param df:
    :param fea:
    :param good_T:
    :param bad_T:
    :param fea_bins:
    :param fea_label:
    :return:
    '''
    if fea_bins is None:
        fea_bins = optimal_binning_boundary(df[fea], df['label'])
        fea_label = ['%.3e' % (fea_bins[i]) for i in range(1, len(fea_bins))]
        if len(fea_bins) < 5:
            return -1
    totalG_B = df.groupby(['label'])['label'].count()  # 计算正负样本多少个
    good_T, bad_T = totalG_B[0], totalG_B[1]
    sel = pd.cut(df[fea],
                 fea_bins,
                 labels=fea_label,
                 right=False)
    if fea + '_cut' in list(df):
        df.pop(fea + '_cut')
    df.insert(df.shape[1], '%s_cut' % fea, sel)
    var1 = df.groupby(['%s_cut' % fea, 'label'])['label'].count()
    iv = 0
    woe_inv = [0] * len(fea_label)
    sample_rate = [0] * len(fea_label)
    bad_rate = [0] * len(fea_label)
    for i, invl in enumerate(fea_label):
        woe_inv[i] = math.log((var1[invl][1] + 0.5 / bad_T) / (var1[invl][0] + 0.5 / good_T), math.e)
        if woe_inv[i] == np.nan:
            print("check", (var1[invl][1] + 0.5 / bad_T), (var1[invl][0] + 0.5 / good_T), bad_T, var1[invl])
        sample_rate[i] = (var1[invl][1] + var1[invl][0]) / 5000
        bad_rate[i] = var1[invl][1] / (var1[invl][1] + var1[invl][0])
        iv += woe_inv[i] * ((var1[invl][1] / bad_T) - (var1[invl][0] / good_T))
    if draw:
        fig = plt.figure(figsize=(8, 5))
        ax1 = fig.add_subplot(111)
        ax1.bar(fea_label, sample_rate)
        ax1.set_xticklabels(labels=fea_label, rotation=30)
        ax1.set_ylabel('箱内样本数量占比')
        ax2 = ax1.twinx()
        ax2.plot(fea_label, bad_rate, linewidth=2, color='red')
        ax2.scatter(fea_label, bad_rate, color='red')
        ax2.set_ylabel('箱内坏样本占比')
        plt.title('特征%s箱线图iv%.2f' % (fea, iv))
        plt.legend(['箱内样本数量占比', '箱内坏样本占比'])
        plt.savefig('箱线图%d份_%s_划分%s.jpg' % (len(fea_label), fea, draw))
        # plt.show()
    # print("this feature: %s's iv = "%fea, iv)
    woe_encode_fea = pd.cut(df[fea], fea_bins, labels=woe_inv, ordered=True, right=False)
    return iv, woe_encode_fea


def format_print_list(out):
    for line in out:
        line = [str(e) for e in line]
        print(', '.join(line))


# 实现一个向前逐步选择
def cal_aic(model, x, y, y_hat):
    num_params = len(model.coef_) + 1
    mse = mean_squared_error(y, y_hat)
    aic = len(y) * log(mse) + 2 * num_params
    return aic


def bi_select(data, fea, target):
    """
    向前逐步回归
    :param data: 数据
    :param target:目标值
    :return:
    """
    variate = list(set(fea))
    # 参数
    selected = []  # 储存挑选的变量
    # 初始化
    # 初始化决定系数auc, 越大越好
    best_score, score_h = 0, 0
    # 循环删选变量,直至对所有变量进行了选择
    while variate:
        variate_r2 = []

        for var in variate:
            selected.append(var)
            # if len(selected) == 1:
            #     model = net.fit(train_data[selected[0]].values.reshape(-1, 1), y_train)
            #     y_pred = model.predict(test_data[selected[0]].values.reshape(-1, 1))
            # else:
            # y_pred = model.predict(test_data[selected])
            aic = cal_auc(data, selected, target)
            variate_r2.append((aic, var))
            selected.remove(var)
            # 逐个添加变量，看看添加哪个变量可以使aic最小
        variate_r2.sort()
        print(variate_r2)
        score_f, var_f = variate_r2.pop()  # pop用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值
        if best_score < score_f:  # 说明了加了该变量更好了就不移除了,否则就移除
            selected.append(var_f)
            best_score = score_f
            variate.remove(var_f)  # 在纯前向里，删掉避免了重复计算，关键看后向用不用到
            # 在后向里会判断移除它好不好，所以暂时不能remove
            print("R2_f={},continue!".format(best_score))
        else:
            break
    print(selected)
    return selected


# 目前获得了100个iv值较大的特征，现在需要用forward selection选择最好的20个，并送入线性预测模型预测
# 计算模型的psi，衡量预测的稳定性，选择psi小的
def cal_auc(data, fea, target, draw=False):
    X_train, X_test, y_train, y_test = train_test_split(data[fea], data[target], test_size=0.3, random_state=1)
    net = LogisticRegression(penalty="l2", C=1.5, solver='liblinear', max_iter=30000)
    net.fit(X_train, y_train)
    y_pred = net.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 1])
    roc_auc_0 = auc(fpr, tpr)
    if draw:
        plt.plot(fpr, tpr)
    return roc_auc_0


def get_good_fea_list(df, fea_list=None, draw=False):
    if fea_list is None:
        fea_list = '''fea_155
    stage_3_ret_rate
    month21_ret
    fea_1
    fea_6
    ratal_rate
    fea_341
    fea_45
    stage_7_ret_rate
    month31_ret
    fea_144
    fea_133
    fea_56
    month28_ret
    stage_5_ret_rate
    month20_ret
    fea_308
    stage_0_ret_rate
    fea_264
    month29_ret
    fea_187
    fea_319
    stage_6_ret_rate
    fea_198
    month17_ret
    fea_286
    month27_ret
    fea_297
    fea_330
    month30_ret
    stage_4_ret_rate
    month0_ret
    month1_ret
    month18_ret
    month25_ret
    fea_100
    month13_ret
    fea_89
    stage_2_ret_rate
    stage_2_pay_rate
    month19_ret
    sum_ratal_32month
    sum_tax_ret
    month24_ret
    month12_ret
    stage_1_pay_rate
    sum_sales
    stage_4_pay_rate
    month16_ret
    fea_209
    stage_3_pay_rate
    max_tax_pay
    fea_176
    pay_non0_month
    month15_ret
    fea_220
    month14_ret
    item0_sum
    month2_ret
    stage_1_ret_rate
    fea_165
    stage_0_pay_rate
    month9_ret
    month8_ret
    fea_143
    ret_non0_month
    sum_tax_pay
    fea_154
    month23_ret
    sum_ratal
    month7_ret
    month3_ret
    item5_sum
    month16_pay
    fea_22
    fea_253
    month6_ret
    fea_181
    month5_ret
    month4_ret
    lgb_pred
    fea_77
    fea_44
    month10_ret
    fea_242
    fea_110
    month15_pay
    fea_33
    month2_pay
    month12_pay
    fea_99
    fea_275
    month11_pay
    month10_pay
    fea_170
    fea_137
    month7_pay
    fea_126
    month9_pay
    fea_11
    month3_pay
    fea_115
    fea_55
    fea_231
    fea_5
    fea_104
    month6_pay
    fea_27
    fea_66
    month5_pay
    month13_pay
    fea_71
    fea_88
    fea_82
    fea_148
    month14_pay
    month8_pay
    fea_93
    fea_159
    fea_38
    fea_0
    fea_60
    fea_16
    month4_pay
    month1_pay
    month0_pay
    fea_49
    month11_ret
    fea_132
    fea_121
    rf_pred
    '''.split('\n')
    # print(fea_list)
    useful_fea = []
    now = time.time()
    for fea in fea_list:
        if fea not in list(df):
            continue
        totalG_B = df.groupby(['label'])['label'].count()  # 计算正负样本多少个
        good_T, bad_T = totalG_B[0], totalG_B[1]
        bin_gini = get_cart_bincut(df, fea, 'label')
        label_gini = ['%.3e' % (bin_gini[i]) for i in range(1, len(bin_gini))]
        print("gini use time: ", time.time() - now)
        now = time.time()
        bin_iv = optimal_binning_boundary(df[fea], df['label'])
        label_iv = ['%.3e' % (bin_iv[i]) for i in range(1, len(bin_iv))]
        print("iv 用时：", time.time() - now)
        now = time.time()
        bin_kf = get_chimerge_bincut(df, fea, 'label', 10)
        label_kf = ['%.5e' % (bin_kf[i]) for i in range(1, len(bin_kf))]
        print("kf use time: ", time.time() - now)
        now = time.time()

        iv1, _ = woe_iv(df, fea, bin_gini, label_gini, draw='gini' if draw else None)
        iv2, _ = woe_iv(df, fea, bin_iv, label_iv, draw='iv' if draw else None)
        iv3, _ = woe_iv(df, fea, bin_kf, label_kf, draw='kf' if draw else None)
        # if iv1>0.3 or iv2>0.3 or iv3>0.3:
        print("fea: %s\tiv_gini: %.3f\t iv_iv: %.3f\t iv_kf: %.3f\n" % (fea, iv1, iv2, iv3))
        if iv1 > 0.5 or iv1 < 0.02:
            iv1 = 0
        if iv2 > 0.5 or iv2 < 0.02:
            iv2 = 0
        if iv3 > 0.5 or iv3 < 0.02:
            iv3 = 0
        if iv1>=iv2 and iv1>=iv3 and iv1>0:
            useful_fea.append([fea, iv1, bin_gini])
        elif iv2>=iv1 and iv2>=iv3 and iv2>0:
            useful_fea.append([fea, iv2, bin_iv])
        elif iv3>=iv2 and iv3>=iv1 and iv3>0:
            useful_fea.append([fea, iv3, bin_kf])

        # if iv > 0.02 and iv<0.6:
        #     good_fea.append([fea, iv])
    print("iv0.02-0.5 fea:", useful_fea)
    # format_print_list(useful_fea)
    # print("0.02-0.6 fea:", good_fea)
    return useful_fea


def draw_and_conclu(df):
    fea_list = fea_list_gini = fea_list_kf = '''fea_275
    month18_ret
    month11_pay
    month20_ret
    month2_pay
    fea_126
    month10_pay
    stage_2_ret_rate
    fea_242
    month28_ret
    fea_264
    month29_ret
    fea_170
    fea_308
    fea_319
    sum_tax_ret
    stage_4_ret_rate
    sum_tax_pay
    fea_187
    fea_38
    fea_148
    pay_non0_month
    month7_pay
    month9_pay
    fea_104
    fea_297
    month27_ret
    fea_115
    fea_82
    fea_286
    fea_198
    month1_pay
    fea_71
    month0_ret
    sum_ratal
    month13_pay
    month17_ret
    month3_pay
    month30_ret
    fea_93
    fea_231
    fea_330
    month16_ret
    month14_pay
    month13_ret
    month0_pay
    item5_sum
    month8_pay
    stage_4_pay_rate
    fea_16
    stage_2_pay_rate
    fea_60
    fea_159
    month1_ret
    sum_ratal_32month
    fea_220
    fea_165
    stage_3_pay_rate
    month19_ret
    fea_49
    sum_sales
    fea_209
    fea_176
    month12_ret
    month5_pay
    stage_1_pay_rate
    month8_ret
    month2_ret
    stage_0_pay_rate
    month9_ret
    month4_pay
    month15_ret
    ret_non0_month
    stage_1_ret_rate
    item0_sum
    month14_ret
    month11_ret
    fea_143
    month10_ret
    fea_132
    fea_154
    month3_ret
    fea_121
    month7_ret
    month4_ret
    month6_ret
    fea_22
    fea_110
    month5_ret
    fea_77
    fea_33
    fea_44
    fea_99
    fea_66
    fea_55
    fea_88
    fea_11
    fea_0
    rf_pred
    lgb_pred'''.split('\n')
    # fea_list = list(df) 'lgb_pred', 'rf_pred',
    form_raw_fea = ['fea_88', 'fea_16', 'fea_176', 'fea_23', 'fea_72', 'fea_331', 'fea_175', 'fea_151', 'fea_3',
                    'fea_167', 'fea_174',
                    'fea_192', 'fea_80', 'fea_18', 'fea_59', 'fea_62', 'fea_75', 'fea_48', 'fea_208', 'fea_230']
    first_select = ['stage_2_ret_rate', 'stage_0_pay_rate', 'stage_3_pay_rate',
                    'fea_88', 'fea_16', 'fea_176', 'stage_1_ret_rate']
    second_select = ['fea_99', 'month1_pay', 'fea_77', 'fea_0', 'fea_55', 'month6_ret', 'fea_11',
                     'fea_38', 'fea_165', 'month2_ret', 'fea_132', 'ret_non0_month']
    fea_yty = ['fea_0', 'fea_66', 'fea_69', 'fea_142', 'fea_11']
    # for i in first_select:
    #     fea_list.remove(i)
    df.fillna(0.0, inplace=True)
    # model_train(df, fea_list, 'label')
    # bi_select(df, fea_list, 'label')
    final_fea = first_select
    dir_sele = ['fea_88', 'fea_16', 'fea_176', 'fea_23', 'ret_non0_month', 'fea_86', 'fea_174', 'fea_7', 'fea_187',
                'fea_59',
                'fea_225', 'month20_pay', 'fea_163', 'stage_2_ret_rate', 'stage_0_ret_rate', 'stage_5_ret_rate']
    final_fea.extend(second_select)
    # 画出箱线图
    # get_good_fea_list(df, final_fea, draw=True)
    # plt.cla()
    # plt.clf()
    # print("几个特征的箱线图画完了\n\n")
    # 用特征预测
    roc0 = cal_auc(df, first_select, 'label', True)
    roc1 = cal_auc(df, ['lgb_pred', 'rf_pred'], 'label', True)
    ori_fea = ['fea_%d' % i for i in range(352)]
    ori_fea.extend(['sum_inv', 'sum_sales', 'sum_ratal', 'inv_mean', 'inv_n'])
    roc2 = cal_auc(df, form_raw_fea, 'label', True)
    roc3 = cal_auc(df, second_select, 'label', True)
    roc4 = cal_auc(df, dir_sele, 'label', True)
    roc5 = cal_auc(df, fea_yty, 'label', True)
    plt.legend(['first_fea%.3f' % roc0, 'lgb+rf_pred%.3f' % roc1, 'from_raw_fea%.3f' % roc2,
                'sec_fea%.3f' % roc3, 'dir_sele%.3f' % roc4, 'yty%.3f' % roc5])
    plt.title('ROC')
    plt.savefig('roc_using_diff_fea.jpg')
    # plt.show()


if __name__ == '__main__':
    df = pd.read_csv('../data_final/all_data.csv')
    df.pop('id')
    df.fillna(0.0, inplace=True)
    if False:
        # 第一步，选出iv值在0.02-0.5之间的特征，并记录分类标准bins和label
        full_fea = list(df)
        unconsider_fea = ['lgb_pred', 'rf_pred', 'label']
        for i in unconsider_fea:
            full_fea.remove(i)
        iv_ok_fea = get_good_fea_list(df, full_fea, False)
        iv_ok_fea_name = [i[0] for i in iv_ok_fea]
        iv_ok_fea_dir = {i[0]: [i[1], i[2]] for i in iv_ok_fea}
        # 第二步，通过前向选择选出较优的20个特征
        for_sel_fea = bi_select(df, iv_ok_fea_name, 'label')
        print("选出来了%d个特征" % len(for_sel_fea))
        # 第三步，编码转换与逻辑回归预测
        woe_fea = []
        for i in for_sel_fea:
            iv, woe = woe_iv(df, i, fea_bins=iv_ok_fea_dir[i][1], fea_label=iv_ok_fea_dir[i][1][1:])
            woe_fea.append(woe)
        woe_fea.append(df['label'])
        for_sel_fea.append('label')
        new_data = pd.concat(woe_fea, axis=1)
        new_data.columns = for_sel_fea
        print(new_data)
        new_data_fea = list(new_data).remove('label')
        cal_auc(new_data, new_data_fea, 'label')

    # unconsider_fea = ['lgb_pred', 'rf_pred', 'label',
    #                   'fea_88', 'fea_16', 'fea_176', 'fea_23', 'ret_non0_month', 'fea_86', 'fea_174', 'fea_7', 'fea_187', 'fea_59',
    #                    'fea_225', 'month20_pay', 'fea_163', 'stage_2_ret_rate', 'stage_0_ret_rate', 'stage_5_ret_rate']
    # ori_fea = ['fea_%d' % i for i in range(352)]
    # ori_fea.extend(['sum_inv', 'sum_sales', 'sum_ratal', 'inv_mean', 'inv_n'])

    # bi_select(df, ori_fea, 'label')
    # draw_and_conclu(df)
