import typing
import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import comb
  


dat=np.array([3,9,6,12,6,15,21])
n=3
N=comb(7, 3)##不放回组合数函数，7个中选3个
samps=pd.DataFrame(list(itertools.combinations(dat,n)))#样本——将所有的组合以列表形式组成dataframe
xbars=np.apply_along_axis(np.mean, 1, samps)#样本——计算每种组合的均值

xbarpdf=pd.value_counts(xbars)/comb(7, n) # 样本——频数计算/总数=概率分布大小

# pip show pandas
# dat = np.array([3,9,6,12,6,15,21])
# n = 3
# N = comb(7,3)
# samps = pd.DataFrame(list(itertools.combinations(dat,n)))
# xbars = np.apply_along_axis(np.mean,1,samps)
# xbarpdf = pd.value_counts(xbars)/comb(7,n)


E_xbar=sum(xbarpdf.index*xbarpdf)#样本——（每种的概率*对应分布值大小）的求和=均值的期望
np.mean(xbars)#样本均值的期望，在不确定样本情况下，其实是很有用的，因为上一步计算其实应该是个统计量（随机变量）    但是这里没用
D_xbar=sum((xbars-E_xbar)**2)/(N-1)##样本——方差
np.var(xbars, ddof=1)#样本方差在/n情况下计算（有偏的）

theta=np.mean(dat) #总体——均值

# E_xbar = sum(xbarpdf.index*xbarpdf)
# np.mean(xbars)
# D_xbar=sum((xbars))

## 中位数 估计量
xmedians=np.apply_along_axis(np.median, 1, samps)##样本——计算每种组合中位数，并以numpy.ndarray的数据结构展示(不属于基本数据结构)

np.mean(xmedians)##样本——中位数的均值

np.var(xmedians, ddof=1)#样本——中位数方差




theta=np.mean(dat) #总体均值
'''1'''
### ddof表示自由度为n- ddof N
theta1=xbars #估计量1，样本均值
print(E_xbar==theta)##估计量1为无偏的
MSEthetahat1=np.var(xbars, ddof=1) # 样本均值不同组合的方差

'''2'''
thetahat2= xmedians # 估计量2，样本中位数
print(np.mean(thetahat2)==theta) ##为有偏估计量

V_thetahat2=np.var(thetahat2, ddof=0) # 样本中位数的方差，除以N
MSE_thetahat2=np.mean((thetahat2-theta)**2) # 样本中位数的方差，除以N

B=np.mean(thetahat2)-theta #计算样本中位数与真实均值之差

print(V_thetahat2+B**2) # 按照公式表示


### 极限分布
N=31
np.random.seed(112)
Heights=np.ceil(np.random.normal(170, 8, N))
print(Heights)
n=3

samps=pd.DataFrame(list(itertools.combinations(Heights, n)))#同上，dataframe展示
xbars=np.apply_along_axis(np.mean, 1, samps)
print(xbars) #均值

plt.hist(xbars) #频数分布图

'''三种误差
1.实际误差
'''
dat=np.array([3,9,6,12,6,15,21])
n=3
N=comb(28, 20)
samps=pd.DataFrame(list(itertools.combinations(dat,n)))
xbars=np.apply_along_axis(np.mean, 1, samps)

np.random.seed(113)
samp1=np.random.choice(dat, n, replace=False)
xbar1=np.mean(samp1)
Xbar=np.mean(dat)

print(xbar1-Xbar)

sum(xbars-Xbar)


[xbar1-1.96*np.std(xbars),xbar1+1.96*np.std(xbars)]#置信区间计算

'''第三章'''
import  numpy as np

pop=np.array(60)+1#生成总体

np.random.seed(123)#设置随机数种子，种子不变，下语句在每次运行时，能够不断接续随机数种子后面的数字组合，保证随机性
np.random.choice(pop, 10, replace=False)#不放回抽样




# 第三章‘
import pandas as pd
tele=pd.read_clipboard()
N=15230

n=len(tele)
f=n/N

ybar_tele=tele.mean()
s_tele=tele.std()

[ybar_tele.values-1.96*((1-f)/n)**0.5*s_tele,ybar_tele.values+1.96*((1-f)/n)**0.5*s_tele]

[ybar_tele.values-1.96*((1-f)/n)**0.5*s_tele,ybar_tele.values+1.96*((1-f)/n)**0.5*s_tele]

up=ybar_tele.values+1.96*((1-f)/n)**0.5*s_tele
down=ybar_tele.values-1.96*((1-f)/n)**0.5*s_tele
interval=pd.concat([down, up])

interval.tolist()

### 总值

(interval*N).tolist()

import numpy as np
## 比例
p=np.mean(tele>80)
low=p-1.96*((1-f)/n)**0.5*n/(n-1)*p*(1-p)
up=p+1.96*((1-f)/n)**0.5*n/(n-1)*p*(1-p)

interval=pd.concat([low, up])
interval.tolist()

np.round((interval*N).tolist())

### 样本容量

r=0.1
t=1.96
s2=n/(n-1)*p*(1-p)

n0=t**2*s2/(r**2*p**2)

n=np.ceil(n0/(1+n0/N))


## 分层抽样

import pandas as pd
import numpy as np

dat=pd.read_clipboard()

Wh=dat.iloc[:,1]/sum(dat.iloc[:], 1)#提取矩阵的某些行列

dat.iloc[:,3:]

np.apply_along_axis(np.mean,dat.iloc[:,3:])#重复进行行列

nh=np.array([10*4])
fh=nh/Nh
s2h=np.apply_along_axis(np.var,1,dat.iloc[:,3:])#每一层的样本方差

v_ybarst=sum(Wh**2*(1-fh)/nh*s2h)

[ybarst-1.96*v_ybarst**0.5, ybarst+1.96*v_ybarst**0.5]

#总体总量的估计
Ybarinterbal=np.array([ybarst-1.96*v_ybarst**0.5, ybarst+1.96*v_ybarst**0.5])

Yhat=N*ybarst
Yhatinterbal=N*Ybarinterbal


##prop
wh_prop=Wh

## Neyman
sh=s2h**0.5
wh_neyman=Wh*sh/sum(Wh*sh)

## Opt
ch=dat.ch
wh_opt=(Wh*sh/ch**0.5)/sum(Wh*sh/ch**0.5)


pd.DataFrame(np.array[wh_prop,wh_neyman, wh_neyman,wh_opr]).T,columns=['prop','neyman','opt']


##样本量的确定
gam=0.1
V=(gam*ybarst/1.96)**2

def samp_size_st(V,wh,Wh,sh,N):
   n=np.ceil(sum(((Wh*sh))**2/wh)/(V+sum(Wh*sh**2)/N))
   return (n,np.seil(n*wh))



#各层分配到的样本量
n_prop=samp_size_st(V,wh_prop,Wh,sh,N)
n_neyman=samp_size_st(V,wh_neyman,Wh,sh,N)
n_opt=samp_size_st(V,wh_opt,Wh,sh,N)









'''5章'''
import pandas as pd
import itertools
import numpy as np
from scipy.special import comb

yi=np.array([6,7,9,10,13,16,17,20])
xi=np.array([3,4,6,8,9,12,13,15])

N=len(yi)
n=3

xbar=np.mean(xi)

Nsamps=comb(N,n)

X=list(itertools.combination(xi,n))
Y=list(itertools.combination(yi,n))

xbars=np.apply_along_axis(np.mean,1,X)
ybars=np.apply_along_axis(np.mean,1,Y)

Rhats=ybars/xbars

ybars_R=Rhats*Xbar


## 验证ybar无偏性
np.mean(ybars)==Ybar

##验证ybar_R无偏性
np.mean(ybars_R)==Ybar


sum((ybars_R-Ybar)**2)(Nsamps-1)
np.var(ybars_R)


#例5.3
dat=pd.read_clipboard()

xi=dat[0::3]
yi=dat[1::3]

N=56
x=86436
Nn=len(yi)


Yhat_R=np.mean(yi)/np.mean(xi)
Yhat_R=Rhat*x

sx2=np.var(xi)
xy2=np.var(yi)
sxy=pd.DataFrame(np.hstack([xi.values,yi.values])).cov().iloc[0,1]


f=n/N
se_Yhat_R=N**2*(1-f)/n*(sy2+Rhat**2*sx2-2*Rhat*sxy)**0.5




'''第五章'''


Yhat=N*ybar

Yhat_se=N*((1-f)/n*sy2)**0.5



## 比率估计量
Rhat=ybar/xbar
YhatR=Rhat*X
YhatR_se=N*((1-f)/n*(sy2+Rhat**2*sx2-2*Rhat*sxy))**0.5


## 回归估计量
b=sxy/sx2
Yhat_lr=N*ybar-b*(N*xbar-X)
r=dat.corr().iloc[0,1]

Yhat_lr_se=N*((1-f)/n*sy2*(1-r**2)*(n-1)/(n-2))**0.5

## 比较
pd.DataFrame([[Yhat, YhatR, Yhat_lr],[Yhat_se, YhatR_se, Yhat_lr_se]],columns=['simple','ratio','lr'])

wheat=pd.read_clipboard(header=None)#读入的为数据框类型，参数设置header为空
Nh=wheat.iloc[:,0]#总数
Wh=wheat.iloc[:,1]#层权
nh=wheat.iloc[:,2]#样本
ybarh=wheat.iloc[:,3]#样本y均值
xbarh=wheat.iloc[:,4]#样本x均值
Xbarh=wheat.iloc[:,5]#总体X均值
s2yh=wheat.iloc[:,6]#样本方差y
s2xh=wheat.iloc[:,7]#样本方差x
sxyh=wheat.iloc[:,8]#样本协方差xy


fh=nh/Nh#抽样比（每一层的）
#简单估计量
ybarst=sum(Wh*ybarh)
v_ybarst=sum(Wh**2*(1-fh)/nh*s2yh)


'''比率估计'''
#分别比率估计
Rhath=ybarh/xbarh
ybarhR=Rhath*Xbarh
ybar_Rs=sum(Wh*ybarhR)
v_ybarRh=(1-fh)/nh*(s2yh+Rhath**2*s2xh**2-2*Rhath*sxy)

v_ybarRs=sum(wh**2*v_ybarRh)


#联合比率估计
xbarst=sum(Wh*xbarh)
Rhatc=ybarst/xbarst
Xbar=sum(Wh*Xbarh)
ybarR=Rhatc*Xbar
v_ybarRh=(1-fh)/nh*(s2yh+Rhath**2*s2xh-2*Rhatc*sxyh)
v_ybarRc=sum(Wh**2*v_ybarRh)


33.203..3

0
'''回归估计'''
#分别回归估计量
bh=sxyh/s2xh
ybarh_lr=ybarh-bh*(xbarh-Xbarh)
ybar_lrs=sum(Wh*ybarh_lr)

v_ybarlrh=(1-fh)/nh*(s2yh+bh**2*s2xh-2*bh*sxyh)
v_ybarlrs=sum(Wh**2*v_ybarlrh)

#联合回归估计量
bc=sum(Wh**2*(1-fh)/nh*sxyh)/sum(Wh**2*(1-fh)/nh*s2xh)
ybar_lrc=ybarst-bc*(xbarst-Xbar)

v_ybarlrh=(1-fh)/nh*(s2yh+bc**2*s2xh-2*bc*sxyh)
v_ybarlrc=sum(Wh**2*v_ybarlrh)

'''比较'''
pd.DataFrame([[ybarst,ybarRc,ybar_Rs,ybar_Rs,ybar_lrs,ybar_lrc]
[v_ybarst,v_ybarRc,v_ybarRs,v_ybarlrs,v_ybarlrc]],index=['统计量','方差'],columns=['simple','Rs','Rc','lrs','lrc']).T




Mi=np.array([88,34,69,15,24,47,36,56,260,51])
def Lahiri(Mi):
    N=len(Mi)
    Mstar=max(Mi)
    np.random.seed(12)
    samp=[]
    while len(samp)<4:
        i=np.random.choice(np.arrange(N))#设置随机数序列
        m=np.random.choice(range(Mstar))
        if Mi[i]>=m:
        samp.append(i)
    return samp



import pandas as pd
import numpy as np
pig=pd.read_csv("pig.csv")



M0=9546
zi=pig.m/M0

Yhat_HH=np.round(pig.y/zi)

v_Yhat_HH=1/n/(n-1)*sum((pig.y/zi-Yhat_HH)**2)

## n=2布鲁尔方法

Mi=np.array([8,4,11,9,6,12,3,20,7])
Zi=Mi


## n>2布鲁尔方法
samp=[]
Mi=np.array([8,4,11,9,6,12,3,20,7])
Zi=Mi/sum(Mi)
N=9
n=4

np.random.seed(123)

samp=[]
#step1
Zi1=Zi*(1-Zi)/(1-4*Zi)/(1-4*Zi+0.0000001)
P1=Zi1/sum(Zi1)

rndnum=np.random.random()
samp.append(sum(rndnum>np.cumsum(P1))+1)

#step2
Zi2=np.delete(Zi,7)
P1=Zi2/sum(Zi2)

rndnum=np.random.random()
samp.append(sum(rndnum>np.cumsum(P2))+1)


###耶茨—格伦迪方法 实施
def pps_code(Mi):
     Zi=Mi/sum(Mi)
     rnum=np.random.rand(1)
     return sum(rnum>np.cumsum(Zi))

import numpy as np
 
# num = 0
# while (num < 5):
#     np.random.seed(0)
#     print(np.random.rand(1,5)) # 得到一个范围从0到1的 1行5列的随机数
#     num += 1
 
# print('-------------------------')
'''放到外面'''
# import numpy as np
 
# num = 0
# np.random.seed(0)
# while (num < 5):
 
#     print(np.random.rand(1,5))
#     num += 1
 
# print('-------------------------')

# 所以我总结就是，通过随机种子，通过一些复杂的数学算法，你可以得到一组有规律的随机数，而随机种子就是这个随机数的初始值。随机种子相同，得到的随机数一定也相同。

import numpy as np
Mi=np.array([8,4,11,9,6,12,3,20,7]) #规模向量
No=np.arange(9)
n=3
np.random.seed(23)

##step1代码法进行抽样
Mi=Mi+np.random.rand(9)/1000000
s1=pps_code(Mi)

##step2
Mi2=np.delete(Mi,s1)
s2temp=pps_code(Mi2)
#到底是第几号个体，需进行判断
s2=np.delete(No,s1)[s2temp]
# s2=Mi.tolist().index(Mi2[s2])

##step3
Mi3=np.delete(Mi,[s1,s2])
s3temp=pps_code(Mi3)
s3=np.delete(No,[s1,s2])[s3temp]
# s3=Mi.tolist().index(Mi2[s3])

np.array([s1,s2,s3])+1

def YG(Mi,n,seed=123):
     """
     method可以选择五种
     YG:
     """
     No=np.arange(len(Mi))
     np.random.seed(seed)
     samp=[]
     for i in range(n):
          print(i)
          sitemp=pps_code(Mi)
          si=np.delete(No,samp)[sitemp]
          samp.append(si)
          Mi=np.delete(Mi,sitemp)
          print(samp)
     return np.array(samp)+1


samps=YG(Mi,3,23)
outputs=np.array([23,72,52])
zi=Mi/sum(Mi)
yicheck=outputs/zi[samp]



##n>2 布鲁尔方法
samp=[]
Mi=np.array([8,4,11,9,6,12,3,20,7])
Zi=Mi/sum(Mi)
N=9;n=4
np.random.seed(123)

samp=[]
#step=1
Zi1=Zi*(1-Zi)/(1-4*Zi+0.000001) #第一步实验的时候，每个个体被抽中的概率
P1=Zi1/sum(Zi1)  #归一化

rndnum=np.random.random()
samp.append(sum(rndnum>np.cumsum(P1))+1) #生成0到1之间的随机数,判断在哪个区间，实施代码法

# step=2
Zi2=np.delete(Zi,samp[-1]-1)
P2=Zi2/sum(Zi2)  #归一化

rndnum=np.random.random()
samp.append(sum(rndnum>np.cumsum(P2))+1) #生成0到1之间的随机数,判断在哪个区间，实施代码法


import pandas as pd
wheat = pd.read_excel("./例7.2.xls",header = None)
ybar = wheat.iloc[:,2].mean()
A, a = 614, 15
f = a/A
Mbar = 408256/A
Ybarbarhat = ybar/Mbar
V_Ybarbarhat = (1-f)/(a*Mbar**2)*wheat.iloc[:,2].var()
 