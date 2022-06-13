# coding=utf-8
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
from imblearn.over_sampling import SMOTE

def GetCsvD( names, data ):
    datalen = len(names)
    idk = data['id'].values
    rows = len(idk)
    res = np.zeros( ( rows, datalen-1 ) )
    for i in range( 1, datalen ):
        subname = names[i]
        subd = data[ subname ].values
        res[ :, i-1 ] = subd
    return idk, res
def GetD( r ):
    data = pd.read_csv( r )
    names = data.columns.values
    idk, d = GetCsvD( names, data )
    return idk, d

def GetNormd( d, maxd ):
    for i in range( 0, d.shape[1] ):
        d[:,i] = d[:,i] / maxd[i]
    return d
def GetModelD( dataid, datad ):
    trainx, trainy = datad[:,:-1], datad[:,-1]
    train_data = trainx
    tar = trainy.copy()
    return dataid, train_data, trainy, tar

def GetMatchD( dataid, tar, Nid, Nd ):
    resall, restrain = [], []
    for i in range( 0, dataid.shape[0] ):
        subid = dataid[i]
        
        selsubd = np.where( Nid == subid ) [0]
        subseltr = []
        if( len(selsubd)>0 ):    
            subindex = selsubd[0]
            subNd = Nd[ subindex ]
            selindex = np.where( subNd==1 )[0]
            resall.append( selindex )
            
            for mk in range( 0, len(selindex) ):
                subsel = selindex[mk]
                subselid = Nid[subsel]
                subselindex = np.where( dataid == subselid ) [0]
                if( len(subselindex)>0  ):
                    subseltr.append( subselindex[0] )
                else:
                    subseltr.append( -1 )
        else:
            subseltr.append( -1 )
        restrain.append( subseltr )

    print( len(restrain) )
    return resall, restrain

def GetMatchD2( dataid, tar, trainid, Nid, Nd ):
    resall, restrain = [], []
    for i in range( 0, dataid.shape[0] ):
        subid = dataid[i]
        subindex = np.where( Nid == subid ) [0][0]
        subNd = Nd[ subindex ]
        selindex = np.where( subNd==1 )[0]
        resall.append( selindex )
        subseltr = []
        for mk in range( 0, len(selindex) ):
            subsel = selindex[mk]
            subselid = Nid[subsel]
            subselindex = np.where( trainid == subselid ) [0]
            if( len(subselindex)>0  ):
                subseltr.append( subselindex[0] )
            else:
                subseltr.append( -1 )
        restrain.append( subseltr )

    print( len(restrain) )
    return resall, restrain


def CalAuc( out, tar ):
    pred = torch.argmax( out, dim = 1 )
    predp = out.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    fpr, tpr, thresholds = roc_curve( tar, predp[:,1] )
    roc_auc = auc(fpr, tpr)
    fpr2, tpr2, thresholds2 = roc_curve( tar, pred )
    roc_auc2 = auc(fpr2, tpr2)
    acc = np.sum( pred == tar ) / tar.shape[0]
    return roc_auc, roc_auc2


