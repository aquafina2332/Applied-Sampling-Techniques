# -*- coding: utf-8 -*-
"""
Created on Sun May  8 11:50:33 2022

@author: aquaf
"""


import pandas as pd
import numpy as np
import math

data = pd.read_csv("5.8.csv")
X1i = np.array(data.iloc[:6,0])  #辅助变量-平原-样本数据
Y1i = np.array(data.iloc[:6,1])   #抽样变量-平原-样本数据
X2i = np.array(data.iloc[:,2])  #辅助变量-山区-样本数据
Y2i = np.array(data.iloc[:,3])  #抽样变量-山区-样本数据

Nh = np.array([120,180])  #各层总量
nh = np.array([6,9])   #各层样本量
Xh = np.array([24500,21200])  #辅助变量总量
Xh_bar = Xh/Nh  #辅助变量均值
Wh = Nh/sum(Nh)  #层权
fh = nh/Nh  #抽样比
sh2 = nh/(nh-1)*np.array([np.var(Y1i),np.var(Y2i)])
sxh2 = nh/(nh-1)*np.array([np.var(X1i),np.var(X2i)])
syxh = 1/(nh-1)*np.array([sum((Y1i-np.mean(Y1i))*(X1i-np.mean(X1i))),
                          sum((Y2i-np.mean(Y2i))*(X2i-np.mean(X2i)))])

#分层随机抽样
yh_bar = np.array([np.mean(Y1i),np.mean(Y2i)])  #抽样变量均值
yst_bar = sum(Wh*yh_bar)  #总体均值估计
Yhat1 = sum(Nh)*yst_bar  #总产量的估计

vystbar = sum((Wh**2)*(1-fh)*sh2/nh)  #总体均值方差估计
stdhat1 = math.sqrt((sum(Nh)**2)*vystbar)  #抽样标准误的估计

#分别比率估计
Rhhat = np.array([np.mean(Y1i)/np.mean(X1i),np.mean(Y2i)/np.mean(X2i)]) #各层比率
yRs_bar = sum(Wh*Rhhat*Xh_bar)
Yhat2 = sum(Nh)*yRs_bar

vyRs_bar = sum(((Wh**2)*(1-fh)/nh)*(sh2+Rhhat**2*sxh2-2*Rhhat*syxh))
stdhat2 = math.sqrt((sum(Nh)**2)*vyRs_bar)

#联合比率估计
xh_bar = np.array([np.mean(X1i),np.mean(X2i)])
xst_bar = sum(Wh*xh_bar)
Rchat = yst_bar/xst_bar  #比率
X_bar = sum(Wh*Xh_bar)
yRc_bar = Rchat*X_bar
Yhat3 = sum(Nh)*yRc_bar

vyRc_bar = sum(((Wh**2)*(1-fh)/nh)*(sh2+Rchat**2*sxh2-2*Rchat*syxh))
stdhat3 = math.sqrt((sum(Nh)**2)*vyRc_bar)

#分别回归估计
bh = syxh/sxh2  #回归系数
ylrh_bar = yh_bar+bh*(Xh_bar-xh_bar)
ylrs_bar = sum(Wh*ylrh_bar)
Yhat4 = sum(Nh)*ylrs_bar

vylrs_bar = sum((Wh**2*(1-fh)/nh)*(1/(nh-2))*((nh-1)*sh2-bh**2*(nh-1)*sxh2))
stdhat4 = math.sqrt((sum(Nh)**2)*vylrs_bar)

#联合回归估计
bc = sum((Wh**2*(1-fh)/nh)*syxh)/sum((Wh**2*(1-fh)/nh)*sxh2)  #回归系数
ylrc_bar = yst_bar+bc*(X_bar-xst_bar)
Yhat5 = sum(Nh)*ylrc_bar

vylrc_bar = sum(((Wh**2)*(1-fh)/nh)*(sh2+bc**2*sxh2-2*bc*syxh))
stdhat5 = math.sqrt((sum(Nh)**2)*vylrc_bar)

#对比解答
Yhat = [Yhat1,Yhat2,Yhat3,Yhat4,Yhat5]
stdhat = [stdhat1,stdhat2,stdhat3,stdhat4,stdhat5] 

#可视化
import matplotlib.pyplot as plt
x_axis_data = [i for i in range(5)]
plt.plot(x_axis_data, Yhat)
plt.show()

plt.plot(x_axis_data, stdhat)
plt.show()


'''
分别比估计与联合比估计：
如果各层的样本量都较大，同时各层的比率之间差异较大，则应采用分别比估计
如果各层的样本量不大，或者各层的比率之间差异较小，则应采用联合比估计

分别回归估计和联合回归估计：
当回归系数需要由样本进行估计时，
如果各层样本量较大，而且各层的回归系数差异较大，则采用分别回归估计
如果各层样本量不大，而且各层的回归系数大致相同，则采用联合回归估计

比估计与回归估计：
（1）分别估计要求各层的样本量都比较大，所以当某些层的样本量不够大时，采用联合估计
（2）当回归系数需要由样本进行估计时，回归估计量时有偏的。样本量越小，偏倚越大，MSE越大，这时采用联合比估计更保险
（3）如果各层样本量都比较大，而且各层的比率或回归系数差异大，则分别估计精度更高；反之反是

在本题中,各层的样本量不大，而且各层的回归系数大致相同，因此精度最高的是联合回归估计
'''




