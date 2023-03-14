# -*- coding: utf-8 -*-
"""
Created on Sun May 22 16:25:05 2022

@author: aquaf
"""
#两个函数可以返回抽取样本的编号
#布鲁尔方法

import numpy as np  #全局宏包

def BLE(Mi,n):

    if n<1 or n>len(Mi) :
        return "Error"  #如果n不是大于等于1的，则返回错误
    else:
        C = []  #创建列表用于储存每次抽样的编号
        P = []  #创建列表用于储存每次抽样的概率
        r = 1
        M0 = sum(Mi)
        Zi = Mi/M0
        while(r<=n):
            Zi_ = Zi*(1-Zi)/(1-(n-r+1)*Zi)
            if r>1:
                Zi_[C[r-2]-1] = 0  #使已抽中的样本再次被抽中的概率是0，列表是从0开始算起，因此需要-1
            P.append(Zi_/sum(Zi_))  #保存一下中间数据
            rd = np.random.rand(1)*sum(Zi_)  #生成0，1之间的随机数并与Zi的和相乘
            C.append(sum(rd>np.cumsum(Zi_))+1)  #标志被抽中的样本
            r = r+1
        return  C   #返回抽中第几个样本


#拉奥-桑福特方法
        
def LS(Mi,n):
    
    if n<1 or n>len(Mi) :
        return "Error"  #如果n不是大于等于1的，则返回错误
    else:
        flag = 1  #用于判断是否有单位被重复抽中
        while(flag == 1) :
            C = []  #创建列表用于储存每次抽样的编号
            P = []  #创建列表用于储存每次抽样的概率
            M0 = sum(Mi)
            Zi = Mi/M0
            P.append(Zi/sum(Zi))
            rd = np.random.rand(1)*sum(Zi)  #生成0，1之间的随机数并与Zi的和相乘
            C.append(sum(rd>np.cumsum(Zi))+1)  #标志被抽中的样本
            Zi_ = Zi/(1-n*Zi)
            Zi_[C[0]-1] = 0
            P.append(Zi_/sum(Zi_)) #保留中间数据
            r = 2
            while(r<=n) :
                rd = np.random.rand(1)*sum(Zi_)  #生成0，1之间的随机数并与Zi的和相乘
                C.append(sum(rd>np.cumsum(Zi_))+1)  #标志被抽中的样本
                r = r+1
            if len(set(C)) == len(C): #判断是否有重复抽取
                flag = 0
                return  C   #返回抽中第几个样本
            else:
                continue
    
#实证分析  
Mi = np.array([8,4,11,9,6,12,3,20,7])
n = 3
BLE(Mi,n)
LS(Mi,n)
