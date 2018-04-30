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

path = "d:\\therm_transport\\thermal_cond\\FF2mKheat.dat"

data=np.genfromtxt(path, unpack=True, skip_header=1, usecols = (2, 5, 6, 13))
""" 0 - time; 1 - frequency; 2 - Q; 3 - Tmct"""
data[0] = data[0]-data[0][0] 

# plotting
plt.plot(data[0], data[2])
plt.xlabel('time (s)')
plt.ylabel('Q')
plt.title('Q vs time')
plt.grid(True)
#plt.savefig("test.png")
plt.show()


#print(data[3])