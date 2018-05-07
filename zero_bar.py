# -*- coding: utf-8 -*-
"""
Created on Wed May  2 12:09:15 2018

@author: Dima
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss

class timetotemp:
    '''obtain T1(Q1) for F1; using this obtain T2(Q1) for F2
    Tc(0bar) = 0.929 mK
    Tc(22) = 2.293 mK
    Tc(9psi) = 1.013 mK'''
    def __init__(self,*nums):
        self.tc=[0.929,1.013,2.293] # list of Tc for experiments
        self.num_exp=nums[0]
        self.num1=nums[1]
        self.num2=nums[2]
        self.offset=nums[3]
        #self.dir="d:\\therm_transport\\data\\0bar\\2018FEB\\" # home dir
        self.dir="c:\\Users\\JMP\\Documents\\Thermal Conductivity\\Backup\\2018FEB\\" # work dir
        # Fork 1
        self.path1=[self.dir+"20180208\\CF0p6mK.dat",self.dir+"20180209\\CF0p4mK.dat",self.dir+"20180210\\CF0p8mK.dat"]
        # Fork 2
        self.path2=[self.dir+"20180208\\FF0p6mK.dat",self.dir+"20180209\\FF0p4mK.dat",self.dir+"20180210\\FF0p8mK.dat"] 
        self.dtime=[]
        self.time,self.Q,self.T=self.import_fun(self.path1) # import fork1
        self.t_fit=self.temp_fit() # linear fir of T vs times. remove nan
        self.TQ=self.QtoTF1() # convert Q into T
        self.time2,self.Q2,self.T2=self.import_fun(self.path2) # import fork 2
        self.time2=self.time2[0:len(self.time2)-self.offset] # cut temperature offset to SF state
        self.Q2=self.Q2[0:len(self.Q2)-self.offset]
        self.T2=self.T2[0:len(self.T2)-self.offset]
        self.TQ2=self.TQ
        tf=np.poly1d(self.TQ2)
        dt2=self.tc[0]-tf(self.Q2[-1])
        self.TQ2[-1]+=dt2
     
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
            time += t1
            F += f1
            Q += q1
            T += temp1
        time=time[self.num1:self.num2]
        Q=Q[self.num1:self.num2]
        T=T[self.num1:self.num2]
        if not self.dtime:
            self.dtime.append(time[0])
            time -= time[0]
        else:
            time -= self.dtime[0]
        return time,Q,T   
     
    def temp_fit(self):
        '''linear regression fit of temperature data, removing nan first'''
        t1=[]
        temp1=[]
        w=[]
        na=np.argwhere(np.isnan(self.T))
        num_del=0
        #print(len(self.T))
        for ii in range(len(self.T)):
            if num_del < len(na):
                if ii == int(na[num_del]): 
                    num_del+=1
                else:
                    t1.append(self.time[ii])
                    temp1.append(self.T[ii])
                    if ii < 0.6*len(self.T):
                        w.append(1)
                    else:
                        w.append(2)
            else:
                t1.append(self.time[ii])
                temp1.append(self.T[ii])
                if ii < 0.6*len(self.T):
                    w.append(1)
                else:
                    w.append(2)
        fit = np.polyfit(t1,temp1,1,w=w)
        fit_fn = np.poly1d(fit) 
        dt=self.tc[0]-fit_fn(t1[-1]) #correction to tc
        fit[1]+=dt
        return fit
    
    def QtoTF1(self):
        '''Transformation of Q into Temperature based on Fork1'''
        #filt=ss.medfilt(A.Q,151) #filtering
        filt1=ss.savgol_filter(self.Q,111,7)
        filt=ss.medfilt(filt1,151) #filtering
        fit = np.polyfit(self.time,filt,6)
        fit_fn = np.poly1d(fit) # Q
        Q=fit_fn(self.time)
        tx=np.poly1d(self.t_fit)
        T=tx(self.time)
        fit_qt=np.polyfit(Q,T,14)
#        fit_revqt=np.poly1d(fit_qt)
#        fig1 = plt.figure(5, clear = True)
#        ax1 = fig1.add_subplot(111)
#        ax1.set_ylabel('T')
#        ax1.set_xlabel('Q')
#        ax1.set_title('T vs Q')
#        ax1.scatter(self.time, fit_revqt(Q), color='blue',s=2)
#        plt.grid()
#        plt.show() 
        return fit_qt
        
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
A=timetotemp(10,9000,47000,3200) 
Q_f=np.poly1d(A.t_fit)
T_f=np.poly1d(A.TQ2)
T_f1=np.poly1d(A.TQ)
filt=ss.medfilt(T_f1(A.Q),11) #filtering
## plotting
fig1 = plt.figure(4, clear = True)
ax1 = fig1.add_subplot(111)
ax1.set_ylabel('Temperature')
ax1.set_xlabel('time')
ax1.set_title('Temp vs time for Fork 1')
ax1.scatter(A.time, T_f1(A.Q), color='blue',s=0.5)
ax1.plot(A.time, filt,color='red',lw=2)
plt.grid()
plt.show()

fig1 = plt.figure(2, clear = True)
ax1 = fig1.add_subplot(111)
ax1.set_ylabel('T')
ax1.set_xlabel('time')
ax1.set_title('T vs time for fork2')
#ax1.scatter(A.time, A.T, color='blue',s=0.5)
ax1.plot(A.time2, T_f(A.Q2),color='red',lw=2)
plt.grid()
plt.show()

fig1 = plt.figure(1, clear = True)
ax1 = fig1.add_subplot(111)
ax1.set_ylabel('Temperature')
ax1.set_xlabel('time')
ax1.set_title('Temperature vs time for both forks')
#ax1.scatter(A.time, A.T, color='blue',s=0.5)
f2=ax1.plot(A.time, filt,color='blue',lw=1 )
f1=ax1.plot(A.time2, T_f(A.Q2),color='red',lw=1)
ax1.legend(['Fork 2', 'Fork1'])
plt.grid()
plt.show()

fig1 = plt.figure(3, clear = True)
ax1 = fig1.add_subplot(111)
ax1.set_ylabel('T/T_c')
ax1.set_xlabel('time')
ax1.set_title('Reduced Temperature vs time for both forks')
#ax1.scatter(A.time, A.T, color='blue',s=0.5)
f2=ax1.plot(A.time, filt/A.tc[0],color='blue',lw=1 )
f1=ax1.plot(A.time2, T_f(A.Q2)/A.tc[0],color='red',lw=1)
ax1.legend(['Fork 2', 'Fork1'])
plt.grid()
plt.show()