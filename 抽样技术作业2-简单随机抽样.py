# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:44:46 2022

@author: aquaf
"""

quit()

#本题为不考虑顺序的无放回简单随机抽样

import pandas as pd
import numpy as np
import math
from scipy.stats import norm

#(1)95%置信度下估计平均值
data = pd.read_excel(r"D:\桌面\作业\抽样技术\作业数据3.3.xlsx") #读数据
N = 1750
n = len(data)
f = n/N
ybar_data = data.mean() #期望
s_data = data.std() #标准差

##假定样本均值服从标准正态分布
confidence = 0.95  
a , b = norm.interval(confidence) #求上下分位点
v_ybar1 = (1-f)*((s_data)**2)/n #均值的方差
se1 = math.sqrt(v_ybar1) #标准误 
con1 = pd.concat([ybar_data+a*se1 , ybar_data+b*se1]) #区间估计
est1 = np.round(con1.tolist(),2)  #结果保留两位小数


#(2)估计支出超出70元的人数
p = np.mean(data>70) #样本比例
dianguji = p*N
se2 = math.sqrt((1-f)*p*(1-p)/(n-1))
con2 = pd.concat([p+a*se2-1/(2*n) , p+b*se2+1/(2*n)])
est2 = np.round((con2*N).tolist(),2)  #结果保留两位小数


#(3)相对误差限不超过10%，以95%置信度估计支出超过70元人数比例，求样本量
r = 0.1
n0 = (b**2)*(1-p)/((r**2)*p)
n0/N<0.05  #判断如何确定近似样本量，返回False,则用n
n = n0/(1+(n0-1)/N)
est3 = int(n)+1 #结果应该为整数


#(4)绝对误差限不超过3%，以95%置信度估计支出超过70元人数比例，求样本量
deta = 0.03 
n02 = (b**2)*p*(1-p)/(deta**2)
n02/N<0.05  #判断如何确定近似样本量，返回False,则用n
n2 = n02/(1+(n02-1)/N)
est4 = int(n2)+1 #结果应该为整数
