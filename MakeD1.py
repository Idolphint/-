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
# from functions import *

def GetCsvD( names, data ):
    datalen = len(names)
    idk = data['ID'].values
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
def GetLabelD( r ):
    data = pd.read_csv( r )
    names = data.columns.values
    labeld = data[ names[0] ].values
    Id = data[ names[1] ].values
    return Id, labeld

def SetCatchD( sellist, saleid, saled ):
    namelist, valuelist = [], []
    for i in range( len(sellist) ):
        subname = sellist[i]
        namelist.append( subname )
        count = 0
        for j in range( len(saleid) ):
            if( subname == saleid[j] ):
                count = count + saled[j]
        valuelist.append( count )
    return namelist, valuelist
r0 = '.../data_jk/data_jk/train_basic_info'
r1= '../data_jk/train_tax_return_.csv'
r2 = '../data_jk/train_tax_payment_.csv'
r3 = '../data_jk/train_investor_info.csv'
r4 = '../data_jk/train_label.csv'

# labelid, labeld = GetD( r1 )
# print( labelid[0], labeld[0] )

# data = pd.read_csv( r )
# names = data.columns.values
# print( names )

Id, labeld = GetLabelD( r4 )


print( Id[0], labeld[0] )

allnum = len(Id)
posinum = np.sum(labeld)
neginum = allnum - posinum

print( 'allnum, posinum, neginum', allnum, posinum, neginum )

saledata = pd.read_csv( r1 )
# print( saledata )
saleid, saled = saledata['ID'].values, saledata['sales'].values
retaldata = pd.read_csv( r2 )
retalid, retald = retaldata['ID'].values, retaldata['ratal'].values

posc, negc = 0, 0
setp, setn = 228, 2572
trainselplist, trainselnlist = [], []
testselplist, testselnlist = [], []

for i in range( 0, allnum ):
    if( labeld[i] == 1 ):
        if( posc < setp ):
            trainselplist.append( Id[i] )
            posc = posc + 1
        else:
            testselplist.append( Id[i] )
            posc = posc + 1
    else:
        if( negc < setn ):
            trainselnlist.append( Id[i] )
            negc = negc + 1
        else:
            testselnlist.append( Id[i] )
            negc = negc + 1
print('------------split dataset-------------')
print( 'P:', len( trainselplist ), len(testselplist) )
print( 'N:', len( trainselnlist ), len(testselnlist) )
print('------------ sale && retal-------------')
print( 'len sale', len(saled) )
print( 'len retal', len(retald) )
print('------------ split sale -------------')

trainpname_sale, trainpd_sale = SetCatchD( trainselplist, saleid, saled )
trainnname_sale, trainnd_sale = SetCatchD( trainselnlist, saleid, saled )
testpname_sale, testpd_sale = SetCatchD( testselplist, saleid, saled )
testnname_sale, testnd_sale = SetCatchD( testselnlist, saleid, saled )

print( 'P list:', len( trainpname_sale ), len(testpname_sale) )
print( 'n list:', len( trainnname_sale ), len(testnname_sale) )

print('------------ split retal -------------')
trainpname_retal, trainpd_retal = SetCatchD( trainselplist, retalid, retald )
trainnname_retal, trainnd_retal = SetCatchD( trainselnlist, retalid, retald )
testpname_retal, testpd_retal = SetCatchD( testselplist, retalid, retald )
testnname_retal, testnd_retal = SetCatchD( testselnlist, retalid, retald )

print( 'P list:', len( trainpname_retal ), len(testpname_retal) )
print( 'n list:', len( trainnname_retal ), len(testnname_retal) )

print('\n\n\n------------ train -------------')

for i in range( len(trainselplist) ):
    print( trainpname_sale[i], trainpd_sale[i], trainpname_retal[i], trainpd_retal[i], 1 )
for i in range( len(trainselnlist) ):
    print( trainnname_sale[i], trainnd_sale[i], trainnname_retal[i], trainnd_retal[i], 0 )

print( '\n\n\ntest---------------\n\n' )
for i in range( len(testselplist) ):
    print( testpname_sale[i], testpd_sale[i], testpname_retal[i], testpd_retal[i], 1 )
for i in range( len(testselnlist) ):
    print( testnname_sale[i], testnd_sale[i], testnname_retal[i], testnd_retal[i], 0 )

# outv = np.array(outv)
# dataframe = pd.DataFrame({'id':subdn,'label':outv})
# dataframe.to_csv( 'Newpredict.csv', index=False, sep=',' )

# retalid, retald = 

# ...
# train_tax_return  # 申报信息
# Data columns (total 8 columns):
# #	Column	               Dtype	        COMMENT
# --  ------	               ------           -------		
# 0	ID	                   object	        '样本ID'
# 1	tax_return_date	       datetime64[ns]	'实际申报日期'
# 2	tax_return_deadline	   datetime64[ns]	'申报最后期限'
# 3	code_account	       object	        '项目代码'
# 4	code_item	           object	        '品目代码'
# 5	tax_return_begin	   datetime64[ns]	'所属月_起'
# 6	tax_return_end	       datetime64[ns]	'所属月_止'
# 7	sales	               float64	        '销售额'
# dtypes: datetime64[ns](4), float64(2), object(3)


# train_tax_payment # 缴税信息
# Data columns (total 9 columns):
# #	Column	               Dtype             COMMENT
# --  ------	               -----	         -------
# 0	ID	                   object            '样本ID'
# 1	tax_payment_deadline   datetime64[ns]    '缴款最后期限'
# 2	tax_payment_date	   datetime64[ns]    '实际缴款日期'
# 3	tax_payment_begin	   datetime64[ns]    '所属月_起'
# 4	tax_payment_end	       datetime64[ns]    '所属月_止'
# 5	code_account	       object            '项目代码'
# 6	code_item	           object            '品目代码'
# 7	code_taxclasses	       object            '税款种类代码'
# 8	ratal	               float64           '税额' 
# dtypes: datetime64[ns](4), float64(1), object(4)
# ...
