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
        
        self.date = nums[6]
        
        if nums[7] == 1:
            self.fin = False
        else:
            self.fin = True
        
        impDir="C:\\Users\\John Wilson\\Documents\\Cornell REU\\Data\\All Data\\"
        
        if self.date == 619 :            
            # Fork 1
            self.path1 = [impDir+"20180619\\CF1p0mK.dat",impDir+"20180620\\CF2p1mK.dat"]
            
            # Fork 2
            self.path2 = [impDir+"20180619\\FF1p0mK.dat",impDir+"20180620\\FF2p1mK.dat"]
        
        elif self.date == 705 :
            
            # Fork 1
            self.path1 = [impDir+"20180705\\CF1p7mK.dat"]
                
            # Fork 2
            self.path2 = [impDir+"20180705\\FF1p7mK.dat"]
            
        
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
        line, = ax1.plot(dataset[4] + self.start, dataset[2], color='blue', lw=2)
        
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

        
        fig1 = plt.figure(1, clear = True)
        ax1 = fig1.add_subplot(111)
        ax1.set_ylabel('Q')
        ax1.set_xlabel('Point #')
        ax1.set_title('Q vs Point #')
        ax1.scatter( self.rawData1[4][d]+self.start, self.rawData1[1][d], color='blue', s=0.5)
#line, = 
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
            dt = tempC(self.Ppsi) - np.mean(fitTemp[1:30])
            
        elif self.ramp == 1:
            dt = tempC(self.Ppsi) - np.mean(fitTemp[-30:-1])
        
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

        
    def QtotimeF1(self,npol1):
        '''Transformation of Q into Temperature based on Fork1'''
                
        tempData1 = self.rawData1[1]
                
        self.timeAB = 0
        self.step = 0
        
        for i in range( 40, len(tempData1) - 40 ):
            # It is entirely arbitrary what range you choose, as long as you know it won't lead
            # to you calling a point for averaging thats outside your index
            
            avg_1 = sum(tempData1[(i + 1):(i + 20)]) / np.float64(len(tempData1[(i + 1):(i + 20)]))
            
            avg_2 = sum(tempData1[(i - 20):i]) / np.float64(len(tempData1[(i - 20):i]))
            
            if abs(avg_1 - avg_2) > self.step :
                
                self.step = avg_1 - avg_2
                
                self.pointAB = self.rawData1[4][i]
                self.timeAB = self.rawData1[0][i]
            
        tempData1 = tempData1 - self.step * np.heaviside( self.rawData1[0] - self.timeAB, 0 )    
        # subtract off the heavisidetheta function and fit the remaining data with a polynomial
        
        qfit = np.polyfit(self.rawData1[0][self.d],tempData1[self.d],npol1)
        
        qfit = tuple(qfit)
        
        qfit_fn = np.poly1d(qfit) # Q
        
        Q = qfit_fn(self.rawData1[0][self.d])
        
        Qplot = Q + self.step * np.heaviside( self.rawData1[0] - self.timeAB, 0 )
        # Adding in the step function to be able to compare it to the true data
        
#       Check if Q fit to time is good        
        fig1 = plt.figure(5, clear = True)
        ax1 = fig1.add_subplot(111)
        ax1.set_ylabel('Q')
        ax1.set_xlabel('time')
        ax1.set_title('Q vs time (Qtotime prat)')
        ax1.scatter(self.rawData1[0][self.d], self.rawData1[1][self.d], color='blue',s=0.5)
        ax1.plot(self.rawData1[0][self.d], Qplot, color='red',lw=1)
        plt.grid()
        plt.show() 
        
        return(qfit)
        
    def reCallibrateF2(self,shift):
        
        qfit = self.fit_QtoF1
        qfit_fn = np.poly1d(qfit) # Q
        
        Q = qfit_fn(self.rawData1[0][self.d])
        
        Qplot = Q + self.step * np.heaviside( self.rawData1[0] - self.timeAB, 0 )
        
        if self.ramp == -1:
            
            t0 = self.rawData2[0][shift]
            self.p0 = shift + 1
            # plus one here to make sure we don't mess up any indexing later on
            
            q0 =  Qplot[0] - self.rawData2[1][0]
            
        elif self.ramp == 1 :
            # Recall ramp == 1 implies it is warming and therefore will change
            # where our Tc occurs in time
            
            L = len(self.rawData2[0]) - shift
            
            t0 = self.rawData2[0][L]
            self.p0 = L + 1
            # p0 is saved as an attribute to the class because its later used
            # to save time in indexing within a binary search
            
        self.rawData2[4] = self.rawData2[4] - self.p0
        # We can edit rawData2[4] because it is a list of numbers we generated anyway,
        # doing this keeps our point values consistent with time
        
        self.rawData2 = np.vstack((self.rawData2, self.rawData2[0] - t0, self.rawData2[1] + q0 ))
        # Add new rows for corrected time and q value
        
        fig1 = plt.figure(7, clear = True)
        ax1 = fig1.add_subplot(111)
        ax1.set_ylabel('Q')
        ax1.set_xlabel('point #')
        ax1.set_title('Fork 2 Q vs point # and Fork 1 fit')
        ax1.scatter(self.rawData2[4], self.rawData2[6], color='blue',s=0.5)
        line, = ax1.plot(self.rawData1[4], Qplot, color='red', lw=1)
        
        plt.show()    
    
    def k_NNsearch(self,tol):
    
        qfit = self.fit_QtoF1
        qfit_fn = np.poly1d(qfit) # Q
        
        Q = qfit_fn(self.rawData1[0][self.d])
        
        Qplot = Q + self.step * np.heaviside( self.rawData1[0] - self.timeAB, 9000 )
        
        # THe sceond argument of np.haeviside changed to let us skip over the point
        # where the AB transition occurs, otherwise this would raise an error
        
        newTemp = np.zeros(len(self.rawData2[6]))
        
        for i in range(int(self.p0), len(self.rawData2[6])):
            
            first = 0
            last = len( self.rawData2[6] ) - 1
            
        
        
            a = True
            
            if Qplot[i] > 9000:
                
                newTemp[i] = newTemp[i-1]
                
                a = False
                
            if self.rawData2[6][i] > Qplot[-1]:
                a = False
            
            while a == True:
                    
                m = (first + last) / 2
                m = int(m)
                    
                if abs( Qplot[m] - self.rawData2[6][i] ) < tol :
                        
                    newTemp[i] = self.rawData1[2][m]
                            
                    a = False
                    del m
                    
                    print(i, 'completed')
                        
                elif Qplot[m] > self.rawData2[6][i] :

                    last = m + 1
                            
                elif Qplot[m] < self.rawData2[6][i]:
                                                
                    first = m - 1

            
        self.rawData2 = np.vstack(( self.rawData2, newTemp ))
                    
#    def savetofile(self):
##    Work in progress to correctly write files to some directory    
#        '''Write a pulses Temp(time) of true temperature into two .dat files'''
#        Q1=self.rawData1[1][self.d]
#        time1=self.rawData1[0][self.d]
#        
#        Q2=self.rawData2[1][self.nopulse2]
#        time2=self.rawData2[0][self.nopulse2]
#        
#        path1=self.dir+"Fork13n.dat"
#        
#        tf1=np.poly1d(self.TQ)   
#        
#        path2=self.dir+"Fork23n.dat"
#        
#        tf2=np.poly1d(self.TQ2)
#        temp2=tf2(Q2)
#        with open(path1,'w') as file1:
#            file1.write(str1)
#        list2=[]
#        for j in range(len(time2)):
#            list2.append("{0}\t{1}\t{2}\n".format(time2[j],temp2[j],temp2[j]/self.tc[self.set]))
#        str2 = ''.join(list2)
#        with open(path2,'w') as file2:
#            file2.write(str2)
                              
#%% 6/19        
end = 1000000
P = pulse(425,-1, 1, 15266, 26500, 10, 100, 619, 0)
# 15985 to 26500
# psi, cooling (-1) or warming (1) ? ,done cutting?, start point, stop point, cut from start of pulse, cut from end of pulse
# 2nd to last number is date
# last number is done with data treatment
# If you don't know the end point, start with the variable end in its place, adjust from there as needed
del end

if P.doneCutting == 1  :
    
    P.T_fit = P.temp_fit(1)
    
    P.fit_QtoF1 = P.QtotimeF1(7)
    
    P.reCallibrateF2(750)
    
    P.k_NNsearch(0.05)
    
#    P.savetoFile
#%% RUN THIS SECTION ONLY TO VIEW FINAL RESULTS
    # Run via ctrl + Enter
            
fig1 = plt.figure(8, clear = True)
ax1 = fig1.add_subplot(111)
ax1.set_ylabel('New Temperature')
ax1.set_xlabel('time')
ax1.set_title('Fork 2 New Temperature')
ax1.scatter(P.rawData2[0][P.d],P.rawData2[7][P.d], color='blue',s=0.5)
        
plt.show()    

