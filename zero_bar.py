# -*- coding: utf-8 -*-
"""
Created on Wed May  2 12:09:15 2018

@author: Dima
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss
#import sys
import time as e_t
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.simplefilter('ignore', np.RankWarning)

class timetotemp:
    '''obtain T1(Q1) for F1; using this obtain T2(Q1) for F2
    Tc(0bar) = 0.929 mK
    Tc(22) = 2.293 mK
    Tc(9psi) = 1.013 mK'''
    def __init__(self,*nums):
        self.tc=[0.929,1.013,2.293] # list of Tc for experiments
        self.set=nums[0]
        self.num_exp=nums[1]
        self.num1=nums[2]
        self.num2=nums[3]
        self.offset=nums[4]
        if self.set is 0: # zero bar
#            self.dir="f:\\data\\old_hdd_dima\\d\\therm_transport\\data\\0bar\\2018FEB\\" # home dir 0 Bar
            self.dir="c:\\Users\\JMP\\Documents\\Thermal Conductivity\\Backup\\2018FEB\\" # work dir
            # Fork 1
            self.path1=[self.dir+"20180208\\CF0p6mK.dat",self.dir+"20180209\\CF0p4mK.dat",self.dir+"20180210\\CF0p8mK.dat"]
            # Fork 2
            self.path2=[self.dir+"20180208\\FF0p6mK.dat",self.dir+"20180209\\FF0p4mK.dat",self.dir+"20180210\\FF0p8mK.dat"]
            self.rawdata1,self.rawdata2=self.import_fun(self.path1,self.path2) # import fork1, fork 2
            self.pulseID=self.pulse_indicies(2) # find indicies of pulses
        elif self.set is 2: # 22 bar
            #self.dir="d:\\therm_transport\\data\\0bar\\2018FEB\\" # home dir 0 Bar
#            self.dir="d:\\therm_transport\\data\\22bar\\" # home dir 0 Bar
#            self.dir="f:\\data\\old_hdd_dima\\d\\therm_transport\\data\\22bar\\" # home dir 22 Bar
            self.dir="c:\\Users\\JMP\\Documents\\Thermal Conductivity\\Backup\\2017DEC\\" # work dir
            # Fork 1
            self.path1=[self.dir+"20171226\\CF0p62mK.dat",self.dir+"20171227\\CF1p2mK.dat"]#,self.dir+"20171216\\CF1p7mK.dat",self.dir+"20171217\\CF2p3mk.dat"]
            # Fork 2
            self.path2=[self.dir+"20171226\\FF0p62mK.dat",self.dir+"20171227\\FF1p2mK.dat"]#,self.dir+"20171216\\FF1p7mK.dat",self.dir+"20171217\\FF2p3mk.dat"]
            self.rawdata1,self.rawdata2=self.import_fun(self.path1,self.path2) # import fork1, fork 2
            self.pulseID=self.pulse_indicies(1) # find indicies of pulses
        elif self.set is 1: # 9psi
#            self.dir="f:\\data\\old_hdd_dima\\d\\therm_transport\\data\\9psi\\2018mar\\" # home dir 22 Bar
            self.dir="c:\\Users\\JMP\\Documents\\Thermal Conductivity\\Backup\\2018MAR\\" # work dir
            # Fork 1
            self.path1=[self.dir+"20180315\\CF0p7mK.dat",self.dir+"20180316\\CF0p8mK.dat",self.dir+"20180317\\CF0p9mK.dat"]
            # Fork 2
            self.path2=[self.dir+"20180315\\FF0p7mK.dat",self.dir+"20180316\\CF0p8mK.dat",self.dir+"20180317\\FF0p9mK.dat"]
            self.rawdata1,self.rawdata2=self.import_fun(self.path1,self.path2) # import fork1, fork 2
            self.pulseID=self.pulse_indicies(2) # find indicies of pulses
#        self.calibration()
        
        
    def calibration(self):
        '''The sequence of commands to calibrate temperature according to Q's'''
        
        self.rawdata1,self.rawdata2=self.import_fun(self.path1,self.path2) # import fork1, fork 2
        self.pulseID=self.pulse_indicies(2) # find indicies of pulses
        self.nopulse1,self.nopulse2=self.pulse_remove(20,2) # remove pulse and its surroundings
        self.t_fit,self.linTemp=self.temp_fit() # linear fit of T vs time Fork 1. remove nan
        self.TQ=self.QtoTF1(13) # convert Q into T. Fork 1
        TQ21=np.asarray(self.TQ)
        tf=np.poly1d(TQ21) # convert Q into T Fork 2
        Q21=self.rawdata2[1][self.nopulse2]
        dt2=self.tc[self.set]-tf(Q21[-1])
        TQ21[-1]+=dt2 # count an offset
        self.TQ2=tuple(TQ21)
        self.timeT2=self.QtoTF2() # time to a new temperature for Fork 2
       
    def import_fun(self,path,path1):
        '''import data from .dat files and concentrate matricies'''
        start_time=e_t.time()
        counter=0
        for p in path:
            data=np.genfromtxt(p, unpack=True, skip_header=1, usecols = (2, 6, 13, 7))
            if counter == 0:
                data1=data.copy()
                counter += 1
            else:
                data1=np.concatenate((data1,data),axis=1)
        counter2=0
        for p1 in path1:
            dataF2=np.genfromtxt(p1, unpack=True, skip_header=1, usecols = (2, 6, 13, 7))
            if counter2 == 0:
                data11=dataF2.copy()
                counter2 += 1
            else:
                data11=np.concatenate((data11,dataF2),axis=1)
        print(np.shape(data1))
        data2=data1[0:,self.num1:self.num2]
        data3=data11[0:,self.num1:self.num2-self.offset]
        
        t0=data2[0][0]
        data2[0]=data2[0]-t0
        data3[0]=data3[0]-t0
        print("import_fun time: {}".format(e_t.time()-start_time))
        return data2,data3 
    
    def pulse_indicies(self,f1):
        '''Find pulses in fork 2'''
        start_time=e_t.time()
        if f1 is 1:
            a=np.where(np.abs(self.rawdata1[1])>2000)
        else:
            a=np.where(np.abs(self.rawdata2[1])>1500)
        pulse=[]
        pulse.append(a[0][0])
        ite=a[0][0]
        for x in range(1,np.shape(a)[1]):
            if (a[0][x] > ite+100):
                pulse.append(a[0][x])
                ite=a[0][x]
            else:
                ite += 1
        pul=np.asarray(pulse)       
        print("pulse_index time: {}".format(e_t.time()-start_time))
        return pul
    
    def pulse_remove(self,n1,n2):
        '''Remove pulse and n-surroundings'''
        start_time=e_t.time()
        a=range(-n1,n1*n2)
        s=[]
        for p in np.nditer(self.pulseID):
            for ad in np.nditer(a):
                s.append(ad+p)
        pulse_rem=np.asarray(s)
        d1=np.in1d(range(0,len(self.rawdata1[1])),pulse_rem,assume_unique=True,invert = True)
        d2=np.in1d(range(0,len(self.rawdata2[1])),pulse_rem,assume_unique=True,invert = True)
        print("pulse_remove time: {}".format(e_t.time()-start_time))
        return d1,d2
     
    def temp_fit(self,nump):
        '''linear regression fit of temperature data, removing nan first'''
        start_time=e_t.time()
        na=np.where(np.isnan(self.rawdata1[2]))
        d1=np.in1d(range(0,len(self.rawdata1[2])),na,invert = True)
        w=np.ones(len(self.rawdata1[2]))
        w[int(len(w)/2):]=2
        fit = np.polyfit(self.rawdata1[0][d1],self.rawdata1[2][d1],nump,w=w[d1])
        fit_fn = np.poly1d(fit) 
        temp2=fit_fn(self.rawdata1[0][d1])
        dt=self.tc[self.set]-np.mean(temp2[-30:-1]) #correction to tc
        fit[-1]+=dt
        temp2=fit_fn(self.rawdata1[0][d1])
        fit_rev=np.polyfit(temp2,self.rawdata1[0][d1],nump)
        timeRev=np.poly1d(fit_rev)
        fig1 = plt.figure(7, clear = True)
        ax1 = fig1.add_subplot(111)
        ax1.set_ylabel('T')
        ax1.set_xlabel('time')
        ax1.set_title('T and time')
        ax1.plot(self.rawdata1[0][d1], self.rawdata1[2][d1], color='green',lw=1)
        ax1.plot(self.rawdata1[0][d1], fit_fn(self.rawdata1[0][d1]), color='blue',lw=1)
        plt.grid()
        plt.show()
        
        fig1 = plt.figure(8, clear = True)
        ax1 = fig1.add_subplot(111)
        ax1.set_ylabel('time')
        ax1.set_xlabel('T')
        ax1.set_title('T and time reverse')
        ax1.plot(self.rawdata1[2][d1], self.rawdata1[0][d1], color='green',lw=1)
        ax1.plot(temp2, timeRev(temp2), color='blue',lw=1)
        plt.grid()
        plt.show()
        fit1=tuple(fit)
        fit_rev1=tuple(fit_rev)
        print("temp_fit time: {}".format(e_t.time()-start_time))
        return fit1,fit_rev1
    
    def optim_polies(self,t,q,t_t,npol):
        '''optimization of QtoT'''
        start_time=e_t.time()
        w=np.ones(len(t))
        w[0:200]=5
        w[-200:-1]=5
        tx1=np.poly1d(t_t)
        tx=tx1(t)
#        y1=[]
        fig1 = plt.figure(20, clear = True)
        ax1 = fig1.add_subplot(111,projection='3d')
        er=[]
        num=1
        for i in range(1,npol):
            fittQ = np.polyfit(t,q,i,w=w)
            Q1 = np.poly1d(fittQ)
            Q=Q1(t)
            for j in range(2,npol):
                fitQT=np.polyfit(Q,tx,j)
                temp1=np.poly1d(fitQT)
                temp=temp1(Q)
                if num is 1:
                    err1=np.sum((temp-tx)**2)
                    # should be done further
                    
                er.append(np.sum((temp-tx)**2))
                ax1.scatter(i, j, er[-1])
        ax1.set_ylabel('ttoT_poly')
        ax1.set_xlabel('QtoT_poly')
        ax1.set_zlabel('error')
        ax1.set_zlim(0,10)
        plt.grid()
        plt.show()
        print("Opti polies: {}".format(e_t.time()-start_time))
        
    def optim_poly(self,x,y,npol):
        '''optimization for degree of polyfit'''
        start_time=e_t.time()
        w=np.ones(len(x))
        w[0:200]=5
        w[-200:-1]=5
        y1=[]
        er=[]
        for i in range(1,npol):
            fit = np.polyfit(x,y,i,w=w)
            fit_fn = np.poly1d(fit) # Q
            y1.append(np.sum((fit_fn(x)-y)**2))
            if i is 1:
                err1=np.sum((fit_fn(x)-y)**2)
            else:
                err2=np.sum((fit_fn(x)-y)**2)
                er.append((err1-err2)/err1)
                err1=err2
            
        fig1 = plt.figure(21, clear = True)
        ax1 = fig1.add_subplot(211)
        ax1.set_ylabel('err')
        ax1.set_xlabel('p_num')
        ax1.set_title('error in polynomial')
        ax1.scatter(range(1,npol), y1, color='blue',s=3)
        plt.grid()
        ax2 = fig1.add_subplot(212)
        ax2.set_ylabel('err')
        ax2.set_xlabel('p_num')
        ax2.set_title('error in polynomial')
        ax2.scatter(range(2,npol), er, color='blue',s=3)
#        ax1.plot(Q, fit_revqt(Q), color='red',lw=1)
        plt.grid()
        plt.show() 
        print("Opti poly: {}".format(e_t.time()-start_time))
    
    def QtoTF1(self,npol1,npol2):
        '''Transformation of Q into Temperature based on Fork1'''
        start_time=e_t.time()
        
        fit = np.polyfit(self.rawdata1[0][self.nopulse1],self.rawdata1[1][self.nopulse1],npol1)
        fit_fn = np.poly1d(fit) # Q
        Q=fit_fn(self.rawdata1[0][self.nopulse1])
        w=np.ones(len(Q))
        w[0:200]=5
        w[-200:-1]=5
        tx=np.poly1d(self.t_fit)
        T=tx(self.rawdata1[0][self.nopulse1])
        fit_qt=np.polyfit(Q,T,npol2, w=w)
#        print(fit_qt)
        fit_revqt=np.poly1d(fit_qt)
        print(np.sum((fit_revqt(Q) - T)**2)) #residue
#        tm=fit_revqt(Q)
#        dt=self.tc[self.set]-tm
#        fit_qt[-1] += dt
        
        fig1 = plt.figure(19, clear = True)
        ax1 = fig1.add_subplot(111)
        ax1.set_ylabel('T')
        ax1.set_xlabel('Q')
        ax1.set_title('T vs Q (QtoTemp prat)')
        ax1.scatter(Q, T, color='blue',s=0.5)
        ax1.plot(Q, fit_revqt(Q), color='red',lw=1)
        plt.grid()
        plt.show() 
              
        fit_qt1=tuple(fit_qt)
        print("QtoTF1 time: {}".format(e_t.time()-start_time))
        return fit_qt1
    
    def QtoTF2(self):
        '''Transformation of time into real temperature of Fork 2'''
        start_time=e_t.time()
        T_f=np.poly1d(self.TQ2)
        tfq=T_f(self.rawdata2[1][self.nopulse2])
        fit=np.polyfit(self.rawdata2[0][self.nopulse2],tfq,8)
        fit_f=np.poly1d(fit)
        
#        fig1 = plt.figure(20, clear = True)
#        ax1 = fig1.add_subplot(111)
#        ax1.set_ylabel('T/Tc')
#        ax1.set_xlabel('time [sec]')
#        ax1.set_title('T vs time for both forks')
#        #ax1.scatter(A.time, A.T, color='blue',s=0.5)
#        ax1.plot(self.time, filt/self.tc[self.set],color='red',lw=1)
#        ax1.plot(self.time2,T_f(self.rawdata2[self.nopulse2])/self.tc[self.set],color='green', lw=1)
#        plt.grid()
#        plt.show()
        print("QtoTF2: {}".format(e_t.time()-start_time))
        return fit
    
    def savetofile(self):
        '''Write a pulses Temp(time) of true temperature into two .dat files'''
        start_time=e_t.time()
        Q1=self.rawdata1[1][self.nopulse1]
        time1=self.rawdata1[0][self.nopulse1]
        Q2=self.rawdata2[1][self.nopulse2]
        time2=self.rawdata2[0][self.nopulse2]
        path1=self.dir+"Fork13n.dat"
        tf1=np.poly1d(self.TQ)
        filt=ss.medfilt(tf1(Q1),11) #filtering fork 1        
        path2=self.dir+"Fork23n.dat"
        tf2=np.poly1d(self.TQ2)
        temp2=tf2(Q2)
        list1=[]
        for i in range(len(time1)):
            list1.append("{0}\t{1}\t{2}\n".format(time1[i],filt[i],filt[i]/self.tc[self.set]))
        str1 = ''.join(list1)
        with open(path1,'w') as file1:
            file1.write(str1)
        list2=[]
        for j in range(len(time2)):
            list2.append("{0}\t{1}\t{2}\n".format(time2[j],temp2[j],temp2[j]/self.tc[self.set]))
        str2 = ''.join(list2)
        with open(path2,'w') as file2:
            file2.write(str2)
        fig1 = plt.figure(12, clear = True)
        ax1 = fig1.add_subplot(111)
        ax1.set_ylabel('T/Tc')
        ax1.set_xlabel('time [sec]')
        ax1.set_title('T vs time for both forks (save to file part)')
        ax1.plot(time1, filt/self.tc[self.set],color='red',lw=1) #/self.tc[self.set]
        ax1.plot(time2,tf2(Q2)/self.tc[self.set],color='green', lw=1)
        ax1.legend(['Fork 1','Fork 2'])
        plt.grid()
        plt.show()
        print("savetofile time: {}".format(e_t.time()-start_time))
        
    def importtaus(self):
        '''import data file with taus vs old temperature and convert into a real temperature'''
        start_time=e_t.time()
        path=self.dir+"bar0tau.dat"
        path1=self.dir+"bar0tau_new.dat"
        data=np.genfromtxt(path, unpack=True, skip_header=3)
        Ttotime=np.poly1d(self.linTemp)
        newTime=Ttotime(data[0])
        TimetoT=np.poly1d(self.timeT2)
        newT=TimetoT(newTime)
        open(path1, 'w').write(''.join("{0}\t{1}\t{2}\n".format(newT[i],data[1][i],newT[i]/self.tc[self.set]) for i in range(len(data[0]))))
        fig1 = plt.figure(2, clear = True)
        ax1 = fig1.add_subplot(111)
        ax1.set_ylabel('tau [sec]')
        ax1.set_xlabel('temperature [mK]')
        ax1.set_title('tau vs Temperature (import taus)')
        ax1.scatter(data[0], data[1], color='blue',s=2)
        ax1.scatter(newT,data[1], color='red',s=2)
        plt.grid()
        plt.show()
        print("importtaus time: {}".format(e_t.time()-start_time))
        
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
start_time1=e_t.time()
#C=timetotemp(1,10,10000,53000,250)
#i1,i2=C.pulse_remove(10,5)
#C.nopulse1,C.nopulse2=C.pulse_remove(20,4) # remove pulse and its surroundings
#C.t_fit,C.linTemp=C.temp_fit(3) # linear fit of T vs time Fork 1. remove nan
##C.optim_poly(C.rawdata1[0][C.nopulse1],C.rawdata1[1][C.nopulse1],20)
#C.TQ=C.QtoTF1(7,25) # convert Q into T. Fork 1
#TQ23=np.asarray(C.TQ) # coeff for a fork 2
#tf2f=np.poly1d(TQ23) # convert Q into T Fork 2
#Q32=C.rawdata2[1][C.nopulse2]
#dt2=C.tc[C.set]-tf2f(Q32[-1])
#TQ23[-1]+=dt2 # count an offset
#C.TQ2=tuple(TQ23)
#C.timeT2=C.QtoTF2() # time to a new temperature for Fork 2
#C.savetofile()
#fig1 = plt.figure(11, clear = True)
#ax1 = fig1.add_subplot(211)
#ax1.set_ylabel('Q')
#ax1.set_xlabel('time [sec]')
#ax1.set_title('Q vs time for both forks')
#ax1.scatter(C.rawdata1[0][C.nopulse1],C.rawdata1[1][C.nopulse1],color='green', s=0.5)
#ax2 = fig1.add_subplot(212)
#ax2.set_ylabel('T')
#ax2.set_xlabel('time [sec]')
#ax2.set_title('T vs time for both forks')
#ax2.scatter(C.rawdata1[0][C.nopulse1],C.rawdata1[2][C.nopulse1],color='green', s=0.5)
##ax1.scatter(A.time, A.T, color='blue',s=0.5)
##ax1.plot(A.rawdata2[0][f2],temp/A.tc[A.set],color='green', lw=1)
##ax1.scatter(A.rawdata1[3][f1]-A.rawdata1[3][-1],1/A.rawdata1[1][f1],color='blue', s=0.5)
#plt.grid()
#plt.show()
##fig1 = plt.figure(9, clear = True)
##ax1 = fig1.add_subplot(111)
##ax1.set_ylabel('Q [sec]')
##ax1.set_xlabel('time [mK]')
##ax1.set_title('tau vs Temperature (import taus)')
##ax1.scatter(C.rawdata1[0], C.rawdata1[1], color='blue',s=2)
###ax1.scatter(newT,data[1], color='red',s=2)
##plt.grid()
##plt.show()
#
#del C
B=timetotemp(2,10,2000,30050,250)
i1,i2=B.pulse_remove(10,5)
B.nopulse1,B.nopulse2=B.pulse_remove(20,2) # remove pulse and its surroundings
B.t_fit,B.linTemp=B.temp_fit(1) # linear fit of T vs time Fork 1. remove nan
#B.optim_poly(B.rawdata1[0][B.nopulse1],B.rawdata1[1][B.nopulse1],20)
#B.optim_polies(B.rawdata1[0][B.nopulse1],B.rawdata1[1][B.nopulse1],B.t_fit,20)
B.TQ=B.QtoTF1(7,25) # convert Q into T. Fork 1

TQ21=np.asarray(B.TQ)
tf=np.poly1d(TQ21) # convert Q into T Fork 2
print(tf(B.rawdata2[1][-1]))
Q21=B.rawdata2[1][B.nopulse2]
dt2=B.tc[B.set]-tf(Q21[-1])
TQ21[-1]+=dt2 # count an offset
B.TQ2=tuple(TQ21)
B.timeT2=B.QtoTF2() # time to a new temperature for Fork 2
B.savetofile()
#B.importtaus()
fig1 = plt.figure(90, clear = True)
ax1 = fig1.add_subplot(211)
ax1.set_ylabel('Q')
ax1.set_xlabel('time [sec]')
ax1.set_title('Q vs time for both forks')
ax1.plot(B.rawdata1[0][i1],B.rawdata1[1][i1],color='green', lw=1)
#ax1.plot(A.rawdata1[0][f1],filt/A.tc[A.set],color='blue', lw=1)
plt.grid()
ax2 = fig1.add_subplot(212)
ax2.set_ylabel('Q')
ax2.set_xlabel('time [sec]')
ax2.set_title('Q vs time for both forks')
ax2.plot(B.rawdata2[0][i2],B.rawdata2[1][i2],color='blue', lw=1)
#ax1.plot(A.rawdata1[0][f1],filt/A.tc[A.set],color='blue', lw=1)
plt.grid()
plt.show()
del B
#A=timetotemp(0,20,9200,47000,1800)
#A.nopulse1,A.nopulse2=A.pulse_remove(20,2) # remove pulse and its surroundings
#A.t_fit,A.linTemp=A.temp_fit(1) # linear fit of T vs time Fork 1. remove nan
#A.TQ=A.QtoTF1(6,13) # convert Q into T. Fork 1
#TQ21A=np.asarray(A.TQ)
#tfA=np.poly1d(TQ21A) # convert Q into T Fork 2
#Q21A=A.rawdata2[1][A.nopulse2]
#dt2A=A.tc[A.set]-tfA(Q21A[-1])
#TQ21A[-1]+=dt2A # count an offset
#A.TQ2=tuple(TQ21A)
#A.timeT2=A.QtoTF2() # time to a new temperature for Fork 2
#f1,f2=A.pulse_remove(10,3)
#A.savetofile()
#A.importtaus()
#tf=np.poly1d(A.TQ2)
#tf1=np.poly1d(A.TQ)
#
#temp1=tf1(A.rawdata1[1][f1])
#filt=ss.medfilt(temp1,11)
#temp=tf(A.rawdata2[1][f2])
#fig1 = plt.figure(1, clear = True)
#ax1 = fig1.add_subplot(111)
#ax1.set_ylabel('T/Tc')
#ax1.set_xlabel('time [sec]')
#ax1.set_title('T vs time for both forks')
##ax1.scatter(A.time, A.T, color='blue',s=0.5)
##ax1.plot(self.time, filt/self.tc[self.set],color='red',lw=1)
#ax1.plot(A.rawdata2[0][f2],temp/A.tc[A.set],color='green', lw=1)
#ax1.plot(A.rawdata1[0][f1],filt/A.tc[A.set],color='blue', lw=1)
#plt.grid()
#plt.show()
#
#fig1 = plt.figure(11, clear = True)
#ax1 = fig1.add_subplot(111)
#ax1.set_ylabel('1/Q')
#ax1.set_xlabel('Df [sec]')
#ax1.set_title('1/Q vs df for both forks')
##ax1.scatter(A.time, A.T, color='blue',s=0.5)
##ax1.plot(A.rawdata2[0][f2],temp/A.tc[A.set],color='green', lw=1)
#ax1.scatter(A.rawdata1[3][f1]-A.rawdata1[3][-1],1/A.rawdata1[1][f1],color='blue', s=0.5)
#plt.grid()
#plt.show()
#
#del A
print("Total time: {}".format(e_t.time()-start_time1))