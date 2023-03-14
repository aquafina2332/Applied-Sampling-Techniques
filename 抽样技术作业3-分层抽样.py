# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:13:35 2022

@author: aquaf
"""


import pandas as pd
import numpy as np
import math

data = pd.read_csv('第4章例题.csv',encoding = 'gb2312') #读数据,中文需要转码

#(1)估计总体的平均支出，并给出估计的标准差
Wh = data.iloc[:,1]/sum(data.iloc[:,1])  #返回一个列表 ,层权
ybh = np.apply_along_axis(np.mean, 1, data.iloc[:, 3:])   #每层的均值
Nh = data.iloc[:,1] #第h层单位数
N = sum(Nh)  #总体数
ybst = sum(Wh*ybh)  #带入公式计算得到总体均值的简单估计量

vbh = np.apply_along_axis(np.var, 1, data.iloc[:, 3:])  
nh = np.array([10]*3)  #计算每层的样本数
sbh = nh/(nh-1)*vbh  #计算每层的样本均值
fn = nh/Nh  #计算每层的抽样比
vbst = sum((Wh**2)*(1-fn)*sbh/nh)


#(2)置信度95%，相对误差不超过10%，求总样本量和各层样本量
Ws2 = sum(Wh*sbh)
sh = list(map(math.sqrt,sbh))
Ws = sum(Wh*sh)
t = 1.96
r = 0.1
v = (r*ybst/t)**2

##比例分配
n0 = Ws2/v
n1 = n0/(1+n0/N)   #修正后的总样本量
nh1 =  round(Wh*n1)  #比例分层
nh1
##尼曼分配
n2 = (Ws**2)/(v+Ws2/N)
nh2 = round(n2*Wh*sh/Ws)

##最优分配,费用为简单线性费用函数
ch = data.iloc[:,2]  #各层另外的费用
ch2 = list(map(math.sqrt,ch))  #费用的开方
n3 = (sum(Wh*sh*ch2)*sum(Wh*sh/ch2))/(v+Ws2/N)
nh3 = round((n3*Wh*sh/ch2)/(sum(Wh*sh/ch2)))

n = list(map(round,(n1,n2,n3)))  #总样本量
fr = pd.DataFrame(np.array([nh1,nh2,nh3]).T,\
             columns = ['prop','neyman','opt'])  #分层样本量

'''
iloc 用法

loc 主要通过行标签索引行数据,标签!前闭后闭
df.loc[0:1] 是取第一和第二行;df.loc[0,1]是按坐标取值  
如果列标签是个字符，比如'a'，loc['a']是不行的，必须为loc[:，'a']
但如果行标签是'a',选取这一行，用loc['a']是可以的

iloc 主要是通过行号获取行数据,序号！前闭后开
'''