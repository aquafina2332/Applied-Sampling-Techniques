# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 11:11:33 2022

@author: aquaf
"""

import numpy as np
import pandas as pd
from scipy.special import comb
import itertools as ite
import math
from scipy.stats import norm

data = np.array([1 , 3 , 5 , 7 , 9 , 6 , 12 , 21]) 
n = 3
N = 8

#(1)样本均值的抽样分布
samps = pd.DataFrame(list(ite.combinations(data , n)))
xbars = np.apply_along_axis(np.mean , 1 , samps)
xbarpdf = pd.value_counts(xbars)/comb(N , n)
print('(1) 样本均值抽样分布为:\n' , xbarpdf)

#(2)样本均值抽样本分布的期望与方差
Ex = np.mean(xbars)
Dx = np.var(xbars , ddof = 1)
print('(2) 期望：%.8f ; 方差：%.8f' % (Ex , Dx))  #精度为八位小数

#(3)抽样标准误
##抽样标准误就是抽样分布或抽样估计量的标准差
SE = math.sqrt(np.var(xbars , ddof = 1))
print('(3) 标准误：%.8f ' % SE)  #精度为八位小数

#(4)概率保证程度为95%时的抽样极限误差
##在标准正态分布中,t=1.96时,1-a=95%  deta=t×SE
t = 1.96
deta = t*SE
print('(4) 极限误差：%.8f ' % deta)  #精度为八位小数

#(5)抽中1、7、9，求95%概率保证的总体均值的置信区间
samps2 = [1 , 7 , 9]
mean = np.mean(samps2)  #样本均值
d = math.sqrt(np.var(data , ddof = 1)/N)  #总体增量因子
confidence = 0.95  #置信度
a , b = norm.interval(confidence)  #得到上下分位点
down = mean+a*d
up = mean+b*d
print('(5) 置信区间: (%.8f , %.8f)' % (down , up))  #保留八位小数

