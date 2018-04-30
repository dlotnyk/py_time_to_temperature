# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 14:36:55 2018

@author: Dima
"""

import numpy as np

path = "d:\\therm_transport\\thermal_cond\\FF2mKheat.dat"

data=np.genfromtxt(path, unpack=True, skip_header=1)

print(data[0][0])