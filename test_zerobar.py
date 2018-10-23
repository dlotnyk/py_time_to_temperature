# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 20:05:09 2018

@author: Dima
"""
#import matplotlib.pyplot as plt
#import scipy.signal as ss
#import shutil
##import sys
#import time as e_t
#from mpl_toolkits.mplot3d import Axes3D
import os
import unittest
import numpy as np
from zero_bar import timetotemp
import warnings
warnings.simplefilter('ignore', np.RankWarning)

class TestTimeToTemp(unittest.TestCase):
    '''testing timetotemp'''
    def setUp(self):
        self.currdir1=os.getcwd()
        self.bar0_f1=self.currdir1+"\\CF_0bar_01.dat"
        self.bar0_f2=self.currdir1+"\\FF_0bar_01.dat"
        self.bar0_data1=np.genfromtxt(self.bar0_f1, unpack=True, skip_header=1, usecols = (2, 6, 13, 7))
        self.bar0_data2=np.genfromtxt(self.bar0_f2, unpack=True, skip_header=1, usecols = (2, 6, 13, 7))
        self.dir=self.currdir1+"\\zerobar_data1\\"
        self.A=timetotemp(0,20,9200,47000,1800) # zero bar
        
    def test_importf(self):
        '''test of import function'''
        self.assertEqual(self.bar0_data1[1][9200],self.A.rawdata1[1][0],'HEC data import not done')
        self.assertEqual(self.bar0_data2[1][9200],self.A.rawdata2[1][0],'IC data import not done')
    
if __name__ == '__main__':
    unittest.main()