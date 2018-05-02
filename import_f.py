# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 14:36:55 2018

@author: Dima
    0 - date nan;     1 - time nan;    2 - Universal time;    3 - X(V);    4 - Y(V)
    5 - Drive Frequency; 6- Q;    7 - Inferred frequency;    8 - X - feedthrough;    9 - Y - feedthrough
    10 - Drive voltage;    11 - k_eff;    12 - R_dale;    13 - T_mct;     14 - c_mct
    15  - T_pa;    16 - SP;    17 - BP;    18 - WT
"""
import matplotlib.pyplot as plt
import numpy as np
#import sys

path = "d:\\therm_transport\\thermal_cond\\FF2mKheat.dat"
num_exp=10 # number of point around pulse to remove

data=np.genfromtxt(path, unpack=True, skip_header=1, usecols = (2, 5, 6, 13, 7))
""" 0 - time; 1 - frequency; 2 - Q; 3 - Tmct"""
data[0] = data[0]-data[0][0] 

a=np.where(abs(data[2])>2000)[0]
b=[]

for i in a:
    for j in range(-num_exp,num_exp):
        b.append(i+j)
d=np.in1d(range(0,len(data[0])),b,assume_unique=True,invert = True)

Q=data[2][d]
time=data[0][d]
F=data[4][d]
T=data[3][d]
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
line, = ax2.plot(T, Q, color='blue', lw=2)
plt.show()