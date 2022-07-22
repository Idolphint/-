from sklearn.svm import SVC
import numpy as np
import pandas as pd
from data import Company, svm_data_pre, SVMDataSet
from functions import draw_roc
import pickle

train_f = "../data_jk/train_dataR05.pkl"
eval_f = "../data_jk/train_dataR099.pkl"
test_f = "../data_jk/test.pkl"


def svm_train():
    train_data = SVMDataSet(train_f)
    test_data = SVMDataSet(test_f)
    eval_data = SVMDataSet(eval_f)
    svm_train_x, svm_train_y = train_data.svm_train_x, train_data.svm_train_y
    svm_eval_x, svm_eval_y = eval_data.svm_train_x, eval_data.svm_train_y
    svm_test_x, test_id = test_data.svm_train_x, test_data.svm_id
    # print(svm_train_x.shape, svm_eval_y.shape, len(test_id))
    clf = SVC(probability=True, C=1.5, kernel='rbf')
    clf.fit(svm_train_x, svm_train_y)
    pred = clf.predict( svm_eval_x )
    eval_acc = ( np.sum( pred == svm_eval_y ) / len(svm_eval_y))
    pred = clf.predict( svm_train_x )
    train_acc = ( np.sum( pred == svm_train_y ) / len(svm_train_y))
    print("eval_acc: %.3f, train acc: %.3f"%(eval_acc, train_acc))
    predp = clf.predict_proba( svm_eval_x )
    eval_roc = draw_roc("./vis/svm_eval.jpg", svm_eval_y, predp[:,1], "eval")
    predp = clf.predict_proba(svm_train_x)
    train_roc = draw_roc("./vis/svm_train.jpg", svm_train_y, predp[:, 1], "train")
    print("roc:", min(eval_roc, train_roc))
    test_pred = clf.predict_proba(svm_test_x)[:,1]
    dataframe = pd.DataFrame({'SHXYDM': test_id, 'predict_prob': test_pred})
    dataframe.to_csv("svm_linshi.csv", index=False, sep=',')


if __name__ == '__main__':
    svm_train()
