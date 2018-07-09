# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 16:09:34 2018

@author: JMP
"""
import scipy.optimize as optimization
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss
from scipy.optimize import least_squares
#import sys
import time as e_t
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.simplefilter('ignore', np.RankWarning)
def fun1(t,*par):
    yf=np.poly1d(par[0][:-2])
    y1=yf(t)
    y2=par[0][-1]*np.heaviside(t-par[0][-2],0)
    yt=y1+y2
    return yt
def fun2(t,*par):
    yf=np.poly1d(par[:-2])
    y1=yf(t)
    y2=par[-1]*np.heaviside(t-par[-2],0)
    yt=y1+y2
    return yt
def fun(par,t,ye):
#    s=np.shape(par)
    yf=np.poly1d(par[:-2])
    y1=yf(t)
    y2=par[-1]*np.heaviside(t-par[-2],0)
    yt=y1+y2
    return abs(ye**2-yt**2)
def differ(x,num):
    y=[]
    t1=[]
    print(np.shape(x))
    for i in range(0,np.shape(x)[0]-num-1):
        y.append(x[i+num]-x[i])
        t1.append(i)
    return t1,y
t=np.linspace(0,20,100)
y=fun1(t,(2,1,0,10,100))
x0=np.ones(5)
x0[-2]=10
#x0[-1]=1
#rr=least_squares(fun,x0,args=(t,y))
tt=optimization.curve_fit(fun2,t,y,x0)
y1=fun1(t,tt[0])
print(tt[0])
#tyspe(rr)
tx,dd=differ(y1,4)
ind=np.argmax(dd)
print(ind)
#print(rr)
fig1 = plt.figure(20, clear = True)
ax1 = fig1.add_subplot(211)
ax1.set_ylabel('T/Tc')
ax1.set_xlabel('time [sec]')
ax1.set_title('T vs time for both forks')
#ax1.scatter(A.time, A.T, color='blue',s=0.5)
ax1.scatter(t, y,color='green',lw=1)
ax1.scatter(t[ind],y[ind],color='green',lw=2)
ax1.scatter(t, y1,color='red',lw=0.1)
ax2 = fig1.add_subplot(212)
ax2.scatter(tx, dd,color='green',lw=1)

plt.grid()
plt.show()

