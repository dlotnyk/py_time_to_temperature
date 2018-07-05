# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 11:01:42 2018

@author: John Wilson
"""
import numpy as np
import matplotlib.pyplot as plt

def tempC(Ppsi) :
    
    Pbar = Ppsi * 0.0689476
    
    a0 = 0.92938375
    a1 = 0.13867188
    a2 = -0.0069302185
    a3 = 0.00025685169
    a4 = -0.0000057248644
    a5 = 0.000000053010918
    
    tc = a0 + a1 * (Pbar) + a2 * (Pbar**2) + a3 * (Pbar**3) + a4 * (Pbar**4) + a5 * (Pbar**5)
    
    return(tc)

class pulse:
    
    def __init__(self, Ppsi, *nums):
        
        self.Ppsi = Ppsi
        
        self.Pbar = self.Ppsi * 0.0689475729
        self.Tc = tempC(self.Ppsi)
        
        self.ramp = nums[0]
        
        self.doneCutting = nums[1]
        
        self.start = nums[2]
        self.stop = nums[3]
        
        self.ne1 = nums[4]
        self.ne2 = nums[5]
        
        impDir="C:\\Users\\John Wilson\\Documents\\Cornell REU\\Data\\All Data\\"
        
                    
        # Fork 1
        self.path1 = [impDir+"20180619\\CF1p0mK.dat",impDir+"20180620\\CF2p1mK.dat"]
            
        # Fork 2
        self.path2 = [impDir+"20180619\\FF1p0mK.dat",impDir+"20180620\\FF2p1mK.dat"]
        
        
        self.rawData1,self.rawData2 = self.importData(self.path1),self.importData(self.path2) # import fork 1, fork 2 
        
        n1s = [np.float64( range( 1, len( self.rawData1[0] ) + 1 ) )]
        self.rawData1 = np.concatenate( (self.rawData1, n1s), 0 )

        n2s = [np.float64( range( 1, len( self.rawData2[0] ) + 1 ) )]
        self.rawData2 = np.concatenate( (self.rawData2, n2s), 0 )
        
        if self.doneCutting == 0 :
            
            self.rawData1 = self.cutFunction(self.rawData1)
            self.rawData2 = self.cutFunction(self.rawData2)
        
        elif self.doneCutting == 1:
            
            self.d, self.dtemp = self.filterPulses()
            
            # dtemp gives a logic string for removing NaN from temperature data
            # dtemp also removes points with pulsing near by
            
                        
        
    def importData(self,path):    
    
        fulldata = np.genfromtxt(path[0],
                                 skip_header=1,
                                 unpack=1,
                                 usecols=(2, 6, 13, 7)
                                 )

        for i in range( len(path) - 1):
            newdata = np.genfromtxt(path[i + 1],
                                    skip_header=1,
                                    unpack=1,
                                    usecols=(2, 6, 13, 7)
                                    )
            fulldata = np.concatenate( (fulldata, newdata), axis=1)
        
        data = fulldata[0:,self.start:self.stop]
        
        t0=data[0][0]
        
        data[0]=data[0]-t0
    
        return(data)
    
    def cutFunction(self,dataset):
        
        fig1 = plt.figure(1, clear = True)
        ax1 = fig1.add_subplot(111)
        ax1.set_ylabel('Temp')
        ax1.set_xlabel('point #')
        ax1.set_title('Temp vs point #')
        line, = ax1.plot(dataset[4] + self.start, dataset[1], color='blue', lw=2)
        
        plt.show()
    
    def filterPulses(self) :
                 
        a = np.where( abs( self.rawData2[1] ) > 1500 )[0]
        # np.where function creates a vector of points greater than 2000 in Q
        b = set()
        # this creates a set of numbers to use for b

        for i in a:
    
            for j in range( -self.ne1, self.ne2 ):
                
                b.add( i + j )
                
                # for a point in i, j points to either side are recorded in b
                # Using the add function instead of append skips double numbers
                
        b = list( b )
        # leaving b as a set messes up later code
                
        d = np.isin( range( 0, len(self.rawData2[0]) ), b, invert = True)
        
        na = np.where(np.isnan(self.rawData1[2]))
        d1 = np.isin(range(0,len(self.rawData1[2])),na,invert = True)
        
        dtemp = d & d1
        # creates a list of points that dont have pulsing near them,
        # and removes NaN
        # no need for the assume unique state anymore
          
        # Alternatively you could return "treated" data as a whole new data set,
        # but this would waste memory
        
#        time=self.rawData2[0][d]
#        Q=self.rawData2[1][d]
#        F=self.rawData2[3][d]
#        T=self.rawData2[2][d]
#        treatedData2 = np.float64([time,Q,F,T])
#
#        treatedData1=np.zeros((4,len(treatedData2[0])))
#
#        for i in range(4):
#    
#            treatedData1[i] = self.rawData1[i][d]
#        
#        
#        n1s = [np.float64( range( 1, len( treatedData1[0] ) + 1 ) )]
#        n2s = [np.float64( range( 1, len( treatedData2[0] ) + 1 ) )]
#        
#        treatedData1 = np.concatenate( ( treatedData1, n1s), 0 )
#        treatedData2 = np.concatenate( ( treatedData2, n2s), 0 )   
        
        fig1 = plt.figure(1, clear = True)
        ax1 = fig1.add_subplot(111)
        ax1.set_ylabel('Q')
        ax1.set_xlabel('time')
        ax1.set_title('Q vs time')
        line, = ax1.plot( self.rawData2[0][d], self.rawData2[1][d], color='blue', lw=2)
#         + self.start to look at point numbers ( rawdData2[4][d] ) 
        
        plt.show()         
        
        return d, dtemp
        
    
    def temp_fit(self,deg):
        
        w = np.ones(len(self.rawData1[2]))
#        w[int(len(w)/2):]=2
        
        # Adjust w to adjust the weighting of the fit
        
        fit = np.polyfit(self.rawData1[0][self.d],self.rawData1[2][self.d],deg,w=w[self.d])
        fit_fn = np.poly1d( fit )
        
        fitTemp = fit_fn(self.rawData1[0][self.d])
        
        if self.ramp == -1 :
            dt = tempC(self.Ppsi)-np.mean(fitTemp[1:30])
            
        elif self.ramp == 1:
            dt = tempC(self.Ppsi)-np.mean(fitTemp[-30:-1])
        
        elif self.ramp != -1 or 1 :
            raise Exception('Specify warming or cooling ramp, and make sure it is cut to Tc at one end')
        
        
        fit[-1] += dt # The last element of the fit (cons) is shifted by dt
        
        fitTemp = fit_fn(self.rawData1[0][self.d])
        
        fig2 = plt.figure(2, clear = True)
        ax2 = fig2.add_subplot(111)
        ax2.set_ylabel('T')
        ax2.set_xlabel('time')
        ax2.set_title('T vs time')
    
        ax2.plot(self.rawData1[0][self.d], self.rawData1[2][self.d], color='green',lw=1)
            
        ax2.plot(self.rawData1[0][self.d], fit_fn(self.rawData1[0][self.d]), color='blue',lw=1)
            
        plt.grid()
        plt.show()
            
        fit1=tuple(fit)
        
        return fit1

        
    def QtoTF1(self,npol1,npol2):
        '''Transformation of Q into Temperature based on Fork1'''
        
        qfit = np.polyfit(self.rawData1[0][self.d],self.rawData1[1][self.d],npol1)
        
        qfit_fn = np.poly1d(qfit) # Q
        
        Q = qfit_fn(self.rawData1[0][self.d])
        
#       Check if Q fit to time is good        
#        fig1 = plt.figure(5, clear = True)
#        ax1 = fig1.add_subplot(111)
#        ax1.set_ylabel('Q')
#        ax1.set_xlabel('time')
#        ax1.set_title('Q vs time (QtoTemp prat)')
#        ax1.scatter(self.rawData1[0][self.d], self.rawData1[1][self.d], color='blue',s=0.5)
#        ax1.plot(self.rawData1[0][self.d], Q, color='red',lw=1)
#        plt.grid()
#        plt.show() 
        
        w = np.ones(len(Q))
#        w[0:200]=5
#        w[-200:-1]=5
        
        tx=np.poly1d(self.temp_fit(1))
        
        T=tx(self.rawData1[0][self.dtemp]) #dtemp used here to remove NaN points
        
        fit_qt=np.polyfit(Q,T,npol2, w=w)
        
#        print(fit_qt)
        
        fit_revqt=np.poly1d(fit_qt)
        
#        print(np.sum((fit_revqt(Q) - T)**2)) #residue
#        tm=fit_revqt(Q)
#        dt=self.tc[self.set]-tm
#        fit_qt[-1] += dt
        
        fig1 = plt.figure(5, clear = True)
        ax1 = fig1.add_subplot(111)
        ax1.set_ylabel('T')
        ax1.set_xlabel('Q')
        ax1.set_title('T vs Q (QtoTemp prat)')
        ax1.scatter(Q, T, color='blue',s=0.5)
        ax1.plot(Q, fit_revqt(Q), color='red',lw=1)
        plt.grid()
        plt.show() 
              
        fit_qt1=tuple(fit_qt)
        
        return fit_qt1
        
        
end = 1000000
P = pulse(425,-1, 1, 15985, 28000, 10, 100)
# 15985
# psi, cooling (-1) or warming (1) ? ,done cutting?, start point, stop point, cut from start of pulse, cut from end of pulse
# If you don't know the end point, start with the variable end in its place, adjust from there as needed
del end

if P.doneCutting == 1 :
    
    T_fit = P.temp_fit(1)
    
    fit_QtoF1 = P.QtoTF1(8,9)

#    P.QtoTF2()