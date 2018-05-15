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
        self.dir="d:\\therm_transport\\data\\0bar\\2018FEB\\" # home dir 0 Bar
        #self.dir="c:\\Users\\JMP\\Documents\\Thermal Conductivity\\Backup\\2018FEB\\" # work dir
        # Fork 1
        self.path1=[self.dir+"20180208\\CF0p6mK.dat",self.dir+"20180209\\CF0p4mK.dat",self.dir+"20180210\\CF0p8mK.dat"]
        # Fork 2
        self.path2=[self.dir+"20180208\\FF0p6mK.dat",self.dir+"20180209\\FF0p4mK.dat",self.dir+"20180210\\FF0p8mK.dat"]
        self.calibration()
        
    def calibration(self):
        '''The sequence of commands to calibrate temperature according to Q's'''
        self.dtime=[]
        self.rawdata1,self.rawdata2=self.import_fun(self.path1,self.path2) # import fork1, fork 2
        self.pulse=self.pulse_remove()
#        self.time,self.Q,self.T=self.import_fun(self.path1) # import fork1
#        self.time2,self.Q2,self.T2=self.import_fun(self.path2) # import fork 2
#        self.t_fit,self.linTemp=self.temp_fit() # linear fit of T vs time Fork 1. remove nan
#        self.TQ=self.QtoTF1() # convert Q into T. Fork 1
#        
#        self.time2=self.time2[0:len(self.time2)-self.offset] # cut temperature offset to SF state
#        self.Q2=self.Q2[0:len(self.Q2)-self.offset]
#        self.T2=self.T2[0:len(self.T2)-self.offset]
#        self.TQ2=self.TQ
#        tf=np.poly1d(self.TQ2) # convert Q into T Fork 2
#        dt2=self.tc[0]-tf(self.Q2[-1])
#        self.TQ2[-1]+=dt2 # count an offset
#        self.timeT2=self.QtoTF2() # time to a new temperature for Fork 2
        
    def import_fun(self,path,path1):
        '''import data from .dat files and concentrate matricies'''
        start_time=e_t.time()
        counter=0
        for p in path:
            #num_exp=10 # number of point around pulse to remove
            data=np.genfromtxt(p, unpack=True, skip_header=1, usecols = (2, 6, 13))
            if counter == 0:
                data1=data.copy()
                counter += 1
            else:
                data1=np.concatenate((data1,data),axis=1)
        counter2=0
        for p1 in path1:
            dataF2=np.genfromtxt(p1, unpack=True, skip_header=1, usecols = (2, 6, 13))
            if counter2 == 0:
                data11=dataF2.copy()
                counter2 += 1
            else:
                data11=np.concatenate((data11,dataF2),axis=1)
#            a=np.where(abs(data[2])>1500)[0] # pulse removal
            #print(np.shape(a))
#            b=[]
#            for i in a:
#                for j in range(-int(self.num_exp),int(3*self.num_exp)):
#                    b.append(i+j)
#            d=np.in1d(range(0,len(data[0])),b,assume_unique=True,invert = True)
#            t1=[]
#            f1=[]
#             q1=[]
#            temp1=[]
#            for k in range(np.shape(data)[1]):
#                if d[k]==True:
#                    t1.append(data[0][k])                
#                    f1.append(data[1][k])
#                    q1.append(data[2][k])
#                    temp1.append(data[3][k])
#            time += data[0]
##            F += f1
#            Q += data[1]
#            T += data[2]
#        time=time[self.num1:self.num2]
#        Q=Q[self.num1:self.num2]
#        T=T[self.num1:self.num2]
        #print(np.shape(Q))
#        if not self.dtime:
#            self.dtime.append(time[0])
#            time -= time[0]
#        else:
#            time -= self.dtime[0]
        #print(np.shape(data1))
        data2=data1[0:,self.num1:self.num2]
        data3=data11[0:,self.num1:self.num2-self.offset]
        t0=data2[0][0]
        data2[0]=data2[0]-t0
        data3[0]=data3[0]-t0
        print("import_fun time: {}".format(e_t.time()-start_time))
        return data2,data3 
    
    def pulse_remove(self):
        '''Remove pulses from fork 2'''
        start_time=e_t.time()
        a=np.where(np.abs(self.rawdata2[1])>1500)
        pulse=[]
        pulse.append(a[0][0])
        print(np.shape(a)[1])
        ite=a[0][0]
        for x in range(1,np.shape(a)[1]):
            if (a[0][x] > ite+100):
                pulse.append(a[0][x])
                ite=a[0][x]
            else:
                ite += 1
        pul=np.asarray(pulse)
        #d=np.in1d(range(0,len(self.rawdata2[1])),pulse)
#        d=np.in1d(range(0,len(self.rawdata2[1])),b,assume_unique=True,invert = True)        
        print("pulse_remove time: {}".format(e_t.time()-start_time))
        return pul
     
    def temp_fit(self):
        '''linear regression fit of temperature data, removing nan first'''
        start_time=e_t.time()
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
        dt=self.tc[self.set]-fit_fn(t1[-1]) #correction to tc
        fit[1]+=dt
        fit_fn = np.poly1d(fit)
        temp2=fit_fn(t1)
        fit_rev=np.polyfit(temp2,t1,1)
#        timeRev=np.poly1d(fit_rev)
        
#        fig1 = plt.figure(8, clear = True)
#        ax1 = fig1.add_subplot(111)
#        ax1.set_ylabel('T')
#        ax1.set_xlabel('time')
#        ax1.set_title('T and time')
#        ax1.plot(t1, temp2, color='blue',lw=1)
#        ax1.scatter(self.time,self.T)
#        plt.grid()
#        plt.show()
#        fig1 = plt.figure(3, clear = True)
#        ax1 = fig1.add_subplot(111)
#        ax1.set_ylabel('time')
#        ax1.set_xlabel('T')
#        ax1.set_title('T and time')
#        ax1.plot(temp2, timeRev(temp2), color='blue',lw=1)
#        plt.grid()
#        plt.show()
        print("temp_fit time: {}".format(e_t.time()-start_time))
        return fit,fit_rev
    
    def QtoTF1(self):
        '''Transformation of Q into Temperature based on Fork1'''
        #filt=ss.medfilt(A.Q,151) #filtering
        start_time=e_t.time()
        filt1=ss.savgol_filter(self.Q,111,7)
        filt=ss.medfilt(filt1,151) #filtering
        fit = np.polyfit(self.time,filt,6)
        fit_fn = np.poly1d(fit) # Q
        Q=fit_fn(self.time)
        tx=np.poly1d(self.t_fit)
        T=tx(self.time)
        fit_qt=np.polyfit(Q,T,14)
#        fit_revqt=np.poly1d(fit_qt)
#        fig1 = plt.figure(9, clear = True)
#        ax1 = fig1.add_subplot(111)
#        ax1.set_ylabel('T')
#        ax1.set_xlabel('Q')
#        ax1.set_title('T vs Q')
#        ax1.scatter(self.time, fit_revqt(Q), color='blue',s=2)
#        plt.grid()
#        plt.show() 
        print("QtoTF1 time: {}".format(e_t.time()-start_time))
        return fit_qt
    
    def QtoTF2(self):
        '''Transformation of time into real temperature of Fork 2'''
        start_time=e_t.time()
        T_f=np.poly1d(self.TQ2)
        tfq=T_f(self.Q2)
        filt1=ss.savgol_filter(tfq,63,5)
        filt2=ss.medfilt(filt1,61) #filtering
        fit=np.polyfit(self.time2,filt2,8)
        print("QtoTF2: {}".format(e_t.time()-start_time))
        return fit
    
    def savetofile(self):
        '''Write a pulses Temp(time) of true temperature into two .dat files'''
        start_time=e_t.time()
        path1=self.dir+"Fork1n.dat"
        tf1=np.poly1d(self.TQ)
        filt=ss.medfilt(tf1(self.Q),11) #filtering fork 1        
        path2=self.dir+"Fork2n.dat"
        tf2=np.poly1d(self.TQ2)
        temp2=tf2(self.Q2)
        list1=[]
        for i in range(len(self.time)):
            list1.append("{0}\t{1}\t{2}\n".format(self.time[i],filt[i],filt[i]/self.tc[self.set]))
        str1 = ''.join(list1)
        with open(path1,'w') as file1:
            file1.write(str1)
        list2=[]
        for j in range(len(self.time2)):
            list2.append("{0}\t{1}\t{2}\n".format(self.time2[j],temp2[j],temp2[j]/self.tc[self.set]))
        str2 = ''.join(list2)
        with open(path2,'w') as file2:
            file2.write(str2)
        fig1 = plt.figure(1, clear = True)
        ax1 = fig1.add_subplot(111)
        ax1.set_ylabel('T/Tc')
        ax1.set_xlabel('time [sec]')
        ax1.set_title('T vs time for both forks')
        #ax1.scatter(A.time, A.T, color='blue',s=0.5)
        ax1.plot(self.time, filt/self.tc[self.set],color='red',lw=1)
        ax1.plot(self.time2,tf2(self.Q2)/self.tc[self.set],color='green', lw=1)
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
        ax1.set_title('tau vs Temperature')
        ax1.scatter(data[0], data[1], color='blue',s=2)
        ax1.scatter(newT,data[1], color='red',s=2)
        #ax1.plot(data[0], data[1],color='blue',lw=1)
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
A=timetotemp(0,20,9200,47000,1800)
fig1 = plt.figure(3, clear = True)
ax1 = fig1.add_subplot(111)
ax1.set_ylabel('Q')
ax1.set_xlabel('time')
ax1.set_title('Q and time')
ax1.plot(A.rawdata1[0], A.rawdata1[1], color='blue',lw=1)
plt.grid()
plt.show()

fig1 = plt.figure(4, clear = True)
ax1 = fig1.add_subplot(111)
ax1.set_ylabel('Q')
ax1.set_xlabel('time')
ax1.set_title('Q and time')
#ax1.plot(A.rawdata2[0], A.rawdata2[1], color='blue',lw=1)
ax1.scatter(A.rawdata2[0][A.pulse],A.rawdata2[1][A.pulse],color='red')
#ax1.set_ylim(bottom=10, top=100)
plt.grid()
plt.show()  
#A.importtaus()
del A
#A=timetotemp(0,10,9000,47000,4200) 
#A.savetofile()
#print(sys.getsizeof(A))
#del A
print("Total time: {}".format(e_t.time()-start_time1))
#Q_f=np.poly1d(A.t_fit)
#T_f=np.poly1d(A.TQ2)
#T_f1=np.poly1d(A.TQ)
#filt=ss.medfilt(T_f1(A.Q),11) #filtering
##
#filt1=ss.savgol_filter(T_f(A.Q2),63,5)
#filt2=ss.medfilt(filt1,61) #filtering
#fit2=np.polyfit(A.time2,filt2,8)
#fit2_fn=np.poly1d(fit2)
#
### plotting
#fig1 = plt.figure(4, clear = True)
#ax1 = fig1.add_subplot(111)
#ax1.set_ylabel('Temperature')
#ax1.set_xlabel('time')
#ax1.set_title('Temp vs time for Fork 2')
#ax1.scatter(A.time2, T_f(A.Q2), color='blue',s=0.5)
#ax1.plot(A.time2, fit2_fn(A.time2),color='red',lw=2)
#plt.grid()
#plt.show()
##
#fig1 = plt.figure(5, clear = True)
#ax1 = fig1.add_subplot(111)
#ax1.set_ylabel('T')
#ax1.set_xlabel('time')
#ax1.set_title('T vs time for fork2')
##ax1.scatter(A.time, A.T, color='blue',s=0.5)
#ax1.plot(A.time2, T_f(A.Q2),color='red',lw=2)
#plt.grid()
#plt.show()
#
#fig1 = plt.figure(6, clear = True)
#ax1 = fig1.add_subplot(111)
#ax1.set_ylabel('Temperature')
#ax1.set_xlabel('time')
#ax1.set_title('Temperature vs time for both forks')
##ax1.scatter(A.time, A.T, color='blue',s=0.5)
#f2=ax1.plot(A.time, filt, color='blue', lw=1)
#f1=ax1.plot(A.time2, T_f(A.Q2), color='red', lw=1)
##f3=ax1.plot(A.time2, filt2, color='green', lw=1)
#ax1.legend(['filt','Fork 1', 'Fork 2'])
#plt.grid()
#plt.show()
#
#fig1 = plt.figure(7, clear = True)
#ax1 = fig1.add_subplot(111)
#ax1.set_ylabel('T/T_c')
#ax1.set_xlabel('time')
#ax1.set_title('Reduced Temperature vs time for both forks')
##ax1.scatter(A.time, A.T, color='blue',s=0.5)
#f2=ax1.plot(A.time, filt/A.tc[A.set],color='blue',lw=1)
#f1=ax1.plot(A.time2, T_f(A.Q2)/A.tc[A.set],color='red',lw=1)
#ax1.legend(['Fork 1', 'Fork 2'])
#plt.grid()
#plt.show()