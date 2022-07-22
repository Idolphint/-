import re
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, StratifiedKFold
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import gc
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from data import SVMDataSet, Company

train_f = "../data_jk/train_dataR_yty085.pkl"
eval_f = "../data_jk/eval_dataR_yty015.pkl"
test_f = "../data_jk/test_yty.pkl"
train_data = SVMDataSet(train_f)
test_data = SVMDataSet(test_f)
eval_data = SVMDataSet(eval_f)
svm_train_x, svm_train_y = train_data.svm_train_x, train_data.svm_train_y
svm_eval_x, svm_eval_y = eval_data.svm_train_x, eval_data.svm_train_y
svm_test_x, test_id = test_data.svm_train_x, test_data.svm_id


def fiterDataModel(data_, test_, y_, folds_):
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    train_roc = []
    eval_roc = []
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_)):
        trn_x, trn_y = data_[trn_idx], y_[trn_idx]
        val_x, val_y = data_[val_idx], y_[val_idx]
        clf = LGBMClassifier(
            n_estimators=4000,
            # learning_rate=0.08,
            learning_rate=0.06,
            num_leaves=2 ** 5,
            colsample_bytree=.65,
            subsample=.9,
            max_depth=5,
            reg_alpha=.3,
            reg_lambda=.3,
            min_split_gain=.01,
            min_child_weight=2,
            # silent=-1,
            # verbose=-1,
        )
        clf.fit(trn_x, trn_y,
                eval_set=[(trn_x, trn_y.astype(int)), (val_x, val_y.astype(int))],
                eval_metric='auc', verbose=100, early_stopping_rounds=40  # 40
                )

        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        # train_preds = clf.predict_proba(trn_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_, num_iteration=clf.best_iteration_)[:, 1] / folds_.n_splits
        fpr, tpr, thresholds = roc_curve(val_y, oof_preds[val_idx])
        eval_roc.append(auc(fpr, tpr))
        plt.plot(fpr, tpr)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(y_, oof_preds))

    plt.legend( eval_roc )
    plt.title('ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.savefig('showlgb.jpg')
    plt.show()
    # test_['isDefault'] = sub_preds
    return sub_preds


# tr_cols = set(train_data.columns)
# same_col = list(tr_cols.intersection(set(train_inte.columns)))
# train_inteSame = train_inte[same_col].copy()
# Inte_add_cos = list(tr_cols.difference(set(same_col)))
# for col in Inte_add_cos:
#     train_inteSame[col] = np.nan
# y = train_data['isDefault']
y = svm_train_y
folds = KFold(n_splits=5, shuffle=True, random_state=546789)
IntePre = fiterDataModel(svm_train_x, svm_eval_x, y, folds)
# IntePre['isDef'] = train_inte['isDefault']
print("auc lgb", roc_auc_score(svm_eval_y, IntePre))


## 选择阈值0.05，从internet表中提取预测小于该概率的样本，并对不同来源的样本赋予来源值
# InteId = IntePre[IntePre < 0.5, 'loan_id'].tolist()
# train_inteSame['isDefault'] = train_inte['isDefault']
# use_te = train_inteSame[train_inteSame.loan_id.isin(InteId)].copy()
# data = pd.concat([train_data, test_public, use_te]).reset_index(drop=True)
# print('dataShape:', len(data))


###############开始最后一步的训练
def XGBModel(data_, test_, y_, folds_):
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    train_roc = []
    eval_roc = []
    plt.figure()
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_)):
        trn_x, trn_y = data_[trn_idx], y_[trn_idx]
        val_x, val_y = data_[val_idx], y_[val_idx]
        clf = XGBClassifier(eval_metric='auc', max_depth=5, alpha=0.3, reg_lambda=0.3, subsample=0.8,
                            colsample_bylevel=0.867, objective='binary:logistic', use_label_encoder=False,
                            learning_rate=0.08, n_estimators=4000, min_child_weight=2, tree_method='hist',
                            n_jobs=-1)
        clf.fit(trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],
                eval_metric='auc', verbose=100, early_stopping_rounds=40  # 40
                )
        print("in one fold, train size, test size", len(trn_idx), len(val_idx), len(data_))
        oof_preds[val_idx] = clf.predict_proba(val_x, ntree_limit=clf.best_ntree_limit)[:, 1]
        # train_preds = clf.predict_proba(trn_x, ntree_limit=clf.best_ntree_limit)[:, 1]
        sub_preds += clf.predict_proba(test_, ntree_limit=clf.best_ntree_limit)[:, 1] / folds_.n_splits
        fpr, tpr, thresholds = roc_curve(val_y, oof_preds[val_idx])
        eval_roc.append(auc(fpr, tpr))
        plt.plot(fpr, tpr)
        # fpr, tpr, thresholds = roc_curve(trn_y, train_preds)
        # train_roc.append(auc(fpr, tpr))
        # plt.plot(fpr, tpr)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(y_, oof_preds))
    plt.legend(eval_roc)
    plt.title('ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.savefig('showxgb.jpg')
    plt.show()
    # test_['isDefault'] = sub_preds
    return sub_preds


def LGBModel(data_, test_, y_, folds_):
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_)):
        trn_x, trn_y = data_[trn_idx], y_[trn_idx]
        val_x, val_y = data_[val_idx], y_[val_idx]
        clf = LGBMClassifier(
            n_estimators=4000,
            # learning_rate=0.08,
            learning_rate=0.06,
            num_leaves=2 ** 5,
            colsample_bytree=.65,
            subsample=.9,
            max_depth=5,
            reg_alpha=.3,
            reg_lambda=.3,
            min_split_gain=.01,
            min_child_weight=2,
            # silent=-1,
            # verbose=-1,
        )
        clf.fit(trn_x, trn_y,
                eval_set=[(trn_x, trn_y.astype(int)), (val_x, val_y.astype(int))],
                eval_metric='auc', verbose=100, early_stopping_rounds=40  # 40
                )

        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_, num_iteration=clf.best_iteration_)[:, 1] / folds_.n_splits
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(y_, oof_preds))

    # test_['isDefault'] = sub_preds
    return sub_preds


train = svm_train_x
y = svm_train_y
folds = KFold(n_splits=5, shuffle=True, random_state=546789)
test_preds = XGBModel(train, svm_eval_x, y, folds)
# print(test_preds)
print("auc xgb", roc_auc_score(svm_eval_y, test_preds))
print("avg eval auc", roc_auc_score(svm_eval_y, (test_preds+IntePre)/2))
# res = (test_preds+IntePre)/2
# dataframe = pd.DataFrame({'SHXYDM': test_id, 'predict_prob': res})
# dataframe.to_csv("boost_linshi.csv", index=False, sep=',')

