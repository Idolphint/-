# coding=utf-8
from sklearn.metrics import roc_curve, auc, roc_auc_score
from data import SVMDataSet, Company
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
import pandas as pd
import sklearn.ensemble as se
from imblearn.over_sampling import RandomOverSampler

train_f = "../data_jk/train_dataR_yty099.pkl"
eval_f = "../data_jk/eval_dataR_yty025.pkl"
test_f = "../data_final/all_data.pkl"
train_data = SVMDataSet(train_f)
test_data = SVMDataSet(test_f)
eval_data = SVMDataSet(eval_f)
svm_train_x, svm_train_y = train_data.svm_train_x, train_data.svm_train_y
svm_eval_x, svm_eval_y, eval_id = eval_data.svm_train_x, eval_data.svm_train_y, eval_data.svm_id
svm_test_x, test_id = test_data.svm_train_x, test_data.svm_id

depth, n_estimators, random_state = 3, 200, 100
print( 'depth, n_estimators, random_state:', depth, n_estimators, random_state )
clf = se.RandomForestClassifier(
    max_depth=depth, n_estimators=n_estimators, random_state=random_state)
clf.fit( svm_train_x, svm_train_y )
# pred = clf.predict( traind )
predp = clf.predict_proba( svm_train_x )

fpr, tpr, thresholds = roc_curve( svm_train_y, predp[:,1] )
roc_auc1 = auc(fpr, tpr)
# plt.subplot(1,2,1)
plt.plot( fpr, tpr )
print("train roc", roc_auc1)


clf_lgb = LGBMClassifier(
            n_estimators=16,
            # learning_rate=0.08,
            learning_rate=0.06,
            num_leaves=2 ** 3,
            colsample_bytree=.65,
            subsample=.9,
            max_depth=3,
            reg_alpha=.5,
            reg_lambda=.4,
            min_split_gain=.01,
            min_child_weight=2,
            # silent=-1,
            # verbose=-1,
        )
clf_lgb.fit(svm_train_x, svm_train_y)
predp_lgb = clf_lgb.predict_proba( svm_train_x )

fpr, tpr, thresholds = roc_curve( svm_train_y, predp_lgb[:,1] )
roc_auc_0 = auc(fpr, tpr)
# plt.subplot(1,2,1)
plt.plot( fpr, tpr )
print("train lgb roc", roc_auc_0)
smo = RandomOverSampler( random_state = 35, sampling_strategy='auto' )
# smo = SMOTETomek(random_state=0)


x_smo, y_smo = smo.fit_resample( svm_train_x, svm_train_y )
print( x_smo.shape, y_smo.shape )
print( y_smo, np.sum(y_smo) )
print( svm_train_x.shape, svm_train_y.shape )
svm_train_y = svm_train_y.reshape( (-1,1) )
y_smo = y_smo.reshape( (-1,1) )

straind1 = np.vstack( (svm_train_x, x_smo) )
straintar1 = np.vstack( (svm_train_y, y_smo) )[:,0]

clf1 = se.RandomForestClassifier(
    max_depth=depth, n_estimators=n_estimators, random_state=random_state)
clf1.fit( straind1, straintar1 )
predp1 = clf1.predict_proba( svm_eval_x )

pred = clf.predict( svm_eval_x )
predp = clf.predict_proba( svm_eval_x )

fpr, tpr, thresholds = roc_curve( svm_eval_y, predp[:,1] )
# print( 'test acc', '%.3f'%(np.sum( pred == testtar ) / testtar.shape[0] ) )
roc_auc2 = auc(fpr, tpr)
print( 'test auc:', '%.3f'%roc_auc2 )
plt.plot( fpr, tpr )
#
# fpr, tpr, thresholds = roc_curve( svm_eval_y, predp1[:,1] )
# roc_auc3 = auc(fpr, tpr)
#
# print( 'sub test auc:', '%.3f'%roc_auc3 )
# plt.plot( fpr, tpr )
#
# subpred = predp + predp1
#
# # print( subpred[:,1] )
#
# fpr, tpr, thresholds = roc_curve( svm_eval_y, subpred[:,1] )
# roc_auc4 = auc(fpr, tpr)
# print( 'aver test auc:', '%.3f'%roc_auc4 )
# plt.plot( fpr, tpr )

predp_lgb = clf_lgb.predict_proba(svm_eval_x)

fpr, tpr, thresholds = roc_curve( svm_eval_y, predp_lgb[:,1] )
# print( 'test acc', '%.3f'%(np.sum( pred == testtar ) / testtar.shape[0] ) )
roc_auc5 = auc(fpr, tpr)
print( 'test auc:', '%.3f'%roc_auc5 )
plt.plot( fpr, tpr )

avg_pred = (predp_lgb + predp) / 2
fpr, tpr, thresholds = roc_curve( svm_eval_y, avg_pred[:,1] )
roc_auc6 = auc(fpr, tpr)
print( 'test auc:', '%.3f'%roc_auc6 )

alln = svm_eval_y.shape[0]
posn = int( np.sum( svm_eval_y ) )
negn = alln - posn
print( "test ", alln, posn, negn )

plt.legend( [ 'train:'+'%.3f'%roc_auc1,'train_lgb:%.3f'%roc_auc_0,
              'test:'+'%.3f'%roc_auc2, 'test_lgb:%.3f'%roc_auc5] )
plt.title('ROC')
plt.savefig('rand_forest.jpg')
plt.show()


predp1 = clf.predict_proba( svm_test_x )[:,1]
predp2 = clf_lgb.predict_proba(svm_test_x)[:,1]
dataframe = pd.DataFrame({'SHXYDM': test_id, 'predict_prob': predp2+predp1})
dataframe.to_csv("rf_linshi.csv", index=False, sep=',')

df_o = pd.read_csv('../data_final/all_data.csv')
df_o.insert(df_o.shape[1], 'rf_pred', predp1)
df_o.insert(df_o.shape[1], 'lgb_pred', predp2)
df_o.to_csv('../data_final/all_data2.csv')

