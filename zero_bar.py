# -*- coding: utf-8 -*-
"""
Created on Wed May  2 12:09:15 2018

@author: Dima
"""
import matplotlib.pyplot as plt
import numpy as np

class timetotemp:
    '''obtain T1(Q1) for F1; using this obtain T2(Q1) for F2'''
    def __init__(self,*nums):
        
        self.num1=nums[2]
        self.num2=nums[3]
        self.num_exp=nums[1]
        self.fork=nums[0]
        #self.dir="d:\\therm_transport\\data\\0bar\\2018FEB\\" # home dir
        self.dir="c:\\Users\\JMP\\Documents\\Thermal Conductivity\\Backup\\2018FEB\\" # work dir
        # Fork 1
        self.path1=[self.dir+"20180208\\CF0p6mK.dat",self.dir+"20180209\\CF0p4mK.dat",self.dir+"20180210\\CF0p8mK.dat"]
        # Fork 2
        self.path2=[self.dir+"20180208\\FF0p6mK.dat",self.dir+"20180209\\FF0p4mK.dat",self.dir+"20180210\\FF0p8mK.dat"]  
        
        if self.fork==1:
            self.time,self.Q,self.T=self.import_fun(self.path1)
        else:
            self.time,self.Q,self.T=self.import_fun(self.path2)
     
    def import_fun(self,path):
        '''import data from .dat files and concentrate matricies'''
        time=[]
        F=[]
        Q=[]
        T=[]
        #self.num1=8885 # skip point from begin of files
        #num2=43930 # skip all after
        for p in path:
            #num_exp=10 # number of point around pulse to remove
            data=np.genfromtxt(p, unpack=True, skip_header=1, usecols = (2, 5, 6, 13, 7))
            a=np.where(abs(data[2])>1500)[0] # pulse removal
            
            b=[]
            for i in a:
                for j in range(-self.num_exp,self.num_exp):
                    b.append(i+j)
            na=np.argwhere(np.isnan(data[3]))
#            for ii in na:
#                b.append(ii)
            print("ind=",na[0])
            print("T =", data[3][4])
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
        #print(np.shape(F))
        #print(np.shape(time))
        time=time[self.num1:self.num2]
        Q=Q[self.num1:self.num2]
        T=T[self.num1:self.num2]
        time=time-time[0] 
        return time,Q,T        
    
    def plotting(self,*args):
        '''simplfied, i hope, func for plotting'''
        # 0 - number of fig; 1 - X; 2 - Y; 3 - label X; 4 - label Y
        fig2 = plt.figure(args[0], clear = True)
        ax2 = fig2.add_subplot(111)
        ax2.set_ylabel(args[4])
        ax2.set_xlabel(args[3])
        ax2.set_title(args[4]+' vs '+args[3])
        if args[1]==0:
            X = self.time
        elif args[1]==1:
            X = self.Q
        elif args[1]==2:
            X = self.T
        elif args[1]==3:
            X=range(np.shape(self.time)[0])
            
        if args[2]==0:
            Y = self.time
        elif args[2]==1:
            Y = self.Q
        elif args[2]==2:
            Y = self.T
        elif args[2]==3:
            Y=range(np.shape(self.time)[0])
            
        ax2.scatter(X, Y, color='blue',s=0.5)
        plt.grid()
        plt.show()

# main program statrs here
A=timetotemp(2,10,8885,47000) 
A.plotting(1,3,1,'time','Q')
A.plotting(2,0,2,'time','T')
a=np.argwhere([A.T,np.isnan(A.T)])
#print(a)
#idx = np.isfinite(A.time) & np.isfinite(A.T)
##ab = np.polyfit(x[idx], y[idx], 1)
#fit = np.polyfit(A.time[idx],A.T[idx],1)
#fit_fn = np.poly1d(fit) 
#
##plt.plot(x,y, 'yo', x, fit_fn(x), '--k')
##time,Q,T=import_fun(path2)
## plotting
#fig1 = plt.figure(3, clear = True)
#ax1 = fig1.add_subplot(111)
#ax1.set_ylabel('Q')
#ax1.set_xlabel('time')
#ax1.set_title('Q vs time')
#sc1 = ax1.scatter(A.time, A.T, color='blue',s=0.5)
#ln1=ax1.plot(A.time, fit_fn(A.time))
#plt.grid()
#plt.show()

#ind1=range(np.shape(A.time)[0])
## plotting
#fig1 = plt.figure(1, clear = True)
#ax1 = fig1.add_subplot(111)
#ax1.set_ylabel('Q')
#ax1.set_xlabel('time')
#ax1.set_title('Q vs time')
#line = ax1.plot(ind1, A.Q, color='blue', lw=1)
#plt.show()
#
#fig2 = plt.figure(2, clear = True)
#ax2 = fig2.add_subplot(111)
#ax2.set_ylabel('T')
#ax2.set_xlabel('time')
#ax2.set_title('Q vs temperature')
#scatter = ax2.scatter(A.time, A.T, color='blue')
#plt.show()