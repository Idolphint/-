# coding=utf-8
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2, size_average=True):
        super(FocalLoss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)
            print("alpha", self.alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log_softmax(preds, dim=1)
        # focal_loss func, Loss = -α(1-yi)**γ *ce_loss(xi,yi)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))

        alpha = self.alpha.gather(0, labels)
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


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


