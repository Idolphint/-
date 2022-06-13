from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.io as scio
from sklearn import svm
import numpy as np
import codecs
import xlwt
import xlrd
import sys
import os
import pandas as pd
import sklearn.ensemble as se
from sklearn import tree
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KDTree,BallTree
def ReadXlsx(Fileroot, sheet_num):
    workbook = xlrd.open_workbook(Fileroot)
    sheet = workbook.sheet_by_index(sheet_num)
    return sheet

r = '../data_jk/selfd.xlsx'
trainsheet = ReadXlsx( r, 0 )
testsheet = ReadXlsx( r, 1 )

train0, train1, traintar = trainsheet.col_values(1), trainsheet.col_values(3), trainsheet.col_values(4)
test0, test1, testtar = testsheet.col_values(1), testsheet.col_values(3), testsheet.col_values(4)


lamda = 0.001
train2 = np.array(train1)/(np.array(train0)+lamda)
test2 = np.array(test1)/(np.array(test0)+lamda)

# traind = np.vstack( ( train0, train1 ) ).transpose( 1, 0 )
# testd = np.vstack( ( test0, test1 ) ).transpose( 1, 0 )


traind = np.vstack( ( train0, train1, train2 ) ).transpose( 1, 0 )
testd = np.vstack( ( test0, test1, test2 ) ).transpose( 1, 0 )
# print( traind.shape )
# print( testd.shape )
# fpr, tpr, thresholds = roc_curve( testtar, testd )
# roc_auc = auc(fpr, tpr)
# print( 'test auc:', roc_auc )

# sys.exit(1)

# traind = traind.reshape( (-1,1) )
# testd = testd.reshape( (-1,1) )

traintar = np.array(traintar)
testtar = np.array(testtar)

print( traind.shape )
print( testd.shape )
print( traintar.shape )
print( testtar.shape )

# L2 = np.hstack((l1,l2))
# clf = KDTree(X, leaf_size=30, metric='euclidean')
# clf = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')


clf = se.RandomForestClassifier(
    max_depth=6, n_estimators=200, random_state=7)
# clf = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='entropy'),
#                          n_estimators=100,algorithm="SAMME", learning_rate=0.2)
# clf = tree.DecisionTreeClassifier(criterion='gini', 
#                                          max_depth=None,
#                                          min_samples_leaf=1,
#                                          ccp_alpha=0.0)
# clf = MultinomialNB()
# clf = GaussianNB()
# clf = svm.SVC( kernel='rbf', probability=True )
clf.fit( traind, traintar )
pred = clf.predict( traind )
predp = clf.predict_proba( traind )

fpr, tpr, thresholds = roc_curve( traintar, predp[:,0] )
print( 'train acc','%.3f'%( np.sum( pred == traintar ) / traintar.shape[0] ) )
roc_auc1 = auc(fpr, tpr)
if( roc_auc1<0.5 ):
    fpr, tpr, thresholds = roc_curve( traintar, predp[:,1] )
    roc_auc1 = auc(fpr, tpr)
print( 'train auc:', '%.3f'%roc_auc1 )

plt.plot( fpr, tpr )


predp = clf.predict_proba( testd )

fpr, tpr, thresholds = roc_curve( testtar, predp[:,0] )
print( 'test acc', '%.3f'%(np.sum( pred == testtar ) / testtar.shape[0] ) )
roc_auc2 = auc(fpr, tpr)
if( roc_auc2 < 0.5 ):
    fpr, tpr, thresholds = roc_curve( testtar, predp[:,1] )
    roc_auc2 = auc(fpr, tpr)
print( 'test auc:', '%.3f'%roc_auc2 )


plt.plot( fpr, tpr )
plt.legend( [ 'train:'+'%.3f'%roc_auc1, 'test:'+'%.3f'%roc_auc2 ] )
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')

plt.show()



