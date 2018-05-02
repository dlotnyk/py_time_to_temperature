# -*- coding: utf-8 -*-
"""
Created on Wed May  2 12:09:15 2018

@author: Dima
"""
import matplotlib.pyplot as plt
import numpy as np
def import_fun(path):
    '''import data from .dat files and concentrate matricies'''
    time=[]
    F=[]
    Q=[]
    T=[]
    #print(path)
    for p in path:
        #print(p)
        num_exp=10 # number of point around pulse to remove
        data=np.genfromtxt(p, unpack=True, skip_header=1, usecols = (2, 5, 6, 13, 7))
        #print(np.shape(data)[1])
        
        a=np.where(abs(data[2])>1500)[0] # pulse removal
        b=[]
        for i in a:
            for j in range(-num_exp,num_exp):
                b.append(i+j)
        d=np.in1d(range(0,len(data[0])),b,assume_unique=True,invert = True)
        t1=[]
        f1=[]
        q1=[]
        temp1=[]
        for k in range(np.shape(data)[1]):
            if d[k]==True:
                t1.append(data[0][k])                
                f1.append(data[1][k])
                q1.append(data[2][k])
                temp1.append(data[3][k])
        time=time+t1
        F=F+f1
        Q=Q+q1
        T=T+temp1
    print(np.shape(F))
    time=time-time[0] 
    return time,Q,T
# Fork 1
dir="d:\\therm_transport\\data\\0bar\\2018FEB\\"
path1=[dir+"20180208\\CF0p6mK.dat",dir+"20180209\\CF0p4mK.dat",dir+"20180210\\CF0p8mK.dat"]
# Fork 2
path2=[dir+"20180208\\FF0p6mK.dat",dir+"20180209\\FF0p4mK.dat",dir+"20180210\\FF0p8mK.dat"]
time,Q,T=import_fun(path2)

# plotting
fig1 = plt.figure(1, clear = True)
ax1 = fig1.add_subplot(111)
ax1.set_ylabel('Q')
ax1.set_xlabel('time')
ax1.set_title('Q vs time')
line, = ax1.plot(time, Q, color='blue', lw=2)
plt.show()

fig2 = plt.figure(2, clear = True)
ax2 = fig2.add_subplot(111)
ax2.set_ylabel('Q')
ax2.set_xlabel('temperature')
ax2.set_title('Q vs temperature')
line, = ax2.plot(time, T, color='blue', lw=2)
plt.show()