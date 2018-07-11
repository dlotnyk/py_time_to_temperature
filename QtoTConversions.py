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
        
        self.impDir = "C:\\Users\\John Wilson\\Documents\\Cornell REU\\Data\\All Data\\"
        
        if self.date == 619 :            
            # Fork 1
            self.path1 = [self.impDir+"20180619\\CF1p0mK.dat",self.impDir+"20180620\\CF2p1mK.dat"]
            
            # Fork 2
            self.path2 = [self.impDir+"20180619\\FF1p0mK.dat",self.impDir+"20180620\\FF2p1mK.dat"]
        
        elif self.date == 613 :
            
            # Fork 1
            self.path1 = [self.impDir+"20180613\\CF2p0mK.dat",self.impDir+"20180614\\CF2p2mK.dat"]
                
            # Fork 2
            self.path2 = [self.impDir+"20180613\\FF2p0mK.dat",self.impDir+"20180614\\FF2p2mK.dat"]
            
        elif self.date == 612 :
            
            # Fork 1
            self.path1 = [self.impDir+"20180612\\CF2p1mK.dat"]
                
            # Fork 2
            self.path2 = [self.impDir+"20180612\\FF2p1mK.dat"]
        
        elif self.date == 602 :
            
            # Fork 1
            self.path1 = [self.impDir+"20180602\\CF2p3mK.dat"]
                
            # Fork 2
            self.path2 = [self.impDir+"20180602\\FF2p3mK.dat"]
            
            
        self.rawData1,self.rawData2 = self.importData(self.path1),self.importData(self.path2) # import fork 1, fork 2 
        
        n1s = [np.float64( range( 1, len( self.rawData1[0] ) + 1 ) )]
        self.rawData1 = np.concatenate( (self.rawData1, n1s), 0 )

        n2s = [np.float64( range( 1, len( self.rawData2[0] ) + 1 ) )]
        self.rawData2 = np.concatenate( (self.rawData2, n2s), 0 )
        
        if self.doneCutting == 0 :
            
            self.cutFunction()
        
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
    
    def cutFunction(self):
        
        fig1 = plt.figure(1, clear = True)
        ax1 = fig1.add_subplot(311)
        ax1.set_ylabel('Tmc (mK)')
        ax1.set_title('Tmc and Q(F1 & F2) vs Point # for cutting for '+ str(self.date))
        line, = ax1.plot(self.rawData1[4] + self.start, self.rawData1[2], color='blue', lw=2)
        
        
        ax1 = fig1.add_subplot(312)
        ax1.set_ylabel('Fork 2 Q')
        ax1.scatter( self.rawData1[4]+self.start, self.rawData1[1], color='blue', s=0.5)
        axes = plt.gca()
        axes.set_ylim([0,max(self.rawData1[1])])
        
        ax1 = fig1.add_subplot(313)
        ax1.set_ylabel('Fork 1 Q')
        ax1.set_xlabel('Point #')
        ax1.scatter( self.rawData1[4]+self.start, self.rawData2[1], color='blue', s=0.5)
        axes = plt.gca()
        axes.set_ylim([0,max(self.rawData1[1])])

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
        ax1 = fig1.add_subplot(211)
        ax1.set_ylabel('Fork 1 Q')
        ax1.set_title('Q(F1) and Q(F2) vs Time for Pulse Removal for '+ str(self.date))
        ax1.scatter( self.rawData1[0][d], self.rawData1[1][d], color='blue', s=0.5)
        
        
        ax1 = fig1.add_subplot(212)
        ax1.set_ylabel('Fork 2 Q')
        ax1.set_xlabel('Time (s)')
        ax1.scatter( self.rawData2[0][d], self.rawData2[1][d], color='blue', s=0.5)
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
            dt = self.Tc - np.mean(fitTemp[1:30])
            
        elif self.ramp == 1:
            dt = self.Tc - np.mean(fitTemp[-30:-1])
        
        elif self.ramp != -1 or 1 :
            raise Exception('Specify warming or cooling ramp, and make sure it is cut to Tc at one end')
        
        
        fit[-1] += dt # The last element of the fit (cons) is shifted by dt
        
        fitTemp = fit_fn(self.rawData1[0][self.d])
        
        fig1 = plt.figure(2, clear = True)
        ax1 = fig1.add_subplot(111)
        ax1.set_ylabel('T (mK)')
        ax1.set_xlabel('time (s)')
        ax1.set_title('Tmc and T F1 (corrected fit) vs time for '+ str(self.date))
    
        ax1.plot(self.rawData1[0][self.d], self.rawData1[2][self.d], color='green',lw=1)
        ax1.plot(self.rawData1[0][self.d], fit_fn(self.rawData1[0][self.d]), color='blue',lw=1)
        
        plt.grid()
        plt.show()
        
        self.rawData1 = np.vstack(( self.rawData1, fit_fn(self.rawData1[0][self.d]) ))
        # Row 5 (index 4) is now the corrected temperature for rawData1 (Fork 1)
        # Row 3 (index 3) is still the Tmc temperature (in both rawData1 and rawData2)
        
        fit1=tuple(fit)

        self.T_fit = fit1

        return fit1

        
    def QtotimeF1(self,npol1):
        ''' Fit Fork 1 with a polynomial + step function for Q(t) '''
                
        tempData1 = self.rawData1[1] 
        # Creates a temporary version of Q values
        # within fork 1 that we can adjust
        
        self.timeAB = 0
        self.step = 0
        if self.Pbar >= 21.22 :
            # If Pbar is below the pcp then there will be no AB transition to identify.
            
            # Identify the AB transition
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
        
        # Check if Q fit to time is good        
        fig1 = plt.figure(3, clear = True)
        ax1 = fig1.add_subplot(111)
        ax1.set_ylabel('Q')
        ax1.set_xlabel('time')
        ax1.set_title('Fork 1\'s Q and Q fitted vs time ( Both from Fork 1 ) for ' + str(self.date) )
        ax1.scatter(self.rawData1[0][self.d], self.rawData1[1][self.d], color='blue',s=0.5)
        ax1.plot(self.rawData1[0][self.d], Qplot, color='red',lw=1)
        plt.grid()
        plt.show() 
        
        self.fit_QtoF1 = qfit
        
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
        
        fig1 = plt.figure(4, clear = True)
        ax1 = fig1.add_subplot(111)
        ax1.set_ylabel('Q')
        ax1.set_xlabel('point #')
        ax1.set_title('Fork 2 Q vs point # and Fork 1 fit for '+ str(self.date))
        ax1.scatter(self.rawData2[4], self.rawData2[6], color='blue',s=0.5)
        line, = ax1.plot(self.rawData1[4], Qplot, color='red', lw=1)
        ax1.annotate('Tc points should match here', xy=(0, Qplot[0]), xytext=(self.rawData2[4][-1000], self.rawData2[6][1000]),
#                     arrowprops=dict(facecolor='black', shrink=0.05),
                     )
#        self.rawData2[4][-1000],self.rawData2[6][1000]
        
        
        plt.show()    
    
    def k_NNsearch(self,tol):
    
        qfit = self.fit_QtoF1
        qfit_fn = np.poly1d(qfit) # Q
        
        Q = qfit_fn(self.rawData1[0][self.d])
        u = 0
        if self.Pbar >= 21.22:
            u = self.step * np.heaviside( self.rawData1[0] - self.timeAB, 9000 )
            
        Qplot = Q + u
        # THe sceond argument of np.haeviside changed to let us skip over the point
        # where the AB transition occurs, otherwise this would raise an error
        
        newTemp = np.full(len(self.rawData2[6]), -0.5)
        
        for i in range(int(self.p0), len(self.rawData2[6])):
            
            first = 0
            last = len( self.rawData2[6] ) - 1
            
        
        
            a = True
            
            if Qplot[i] > 9000 or self.rawData2[6][i] < Qplot[0]:
                
                newTemp[i] = newTemp[i-1]
                
                a = False
                
            if self.rawData2[6][i] > Qplot[-1]:
                a = False
            
            while a == True:
                    
                m = (first + last) / 2
                m = int(m)
                    
                if abs( Qplot[m] - self.rawData2[6][i] ) < tol or abs(first - last) <= 3 :
                        
                    newTemp[i] = self.rawData1[2][m]
                            
                    a = False
                    del m
                    
                    print(i, 'completed')
                        
                elif Qplot[m] > self.rawData2[6][i] :

                    last = m + 1
                    
                    
                elif Qplot[m] < self.rawData2[6][i]:
                                                
                    first = m - 1
                     

            
        self.rawData2 = np.vstack(( self.rawData2, newTemp ))

    def savetofile(self):
        list1=[]
        for j in range( -1, len(self.rawData1[0]) ):
            if j == -1:
                list1.append("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format('#[ 1 ] Universal Time (s) ','#[ 2 ] T Local (mK)', '#[ 3 ] Tl / Tc', '#[ 4 ] Tmc / Tc', '#[ 5 ] Q', '#[ 6 ] Inferred Frequency (Hz)' ) )
            else:
                list1.append("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(self.rawData1[0][j],self.rawData1[5][j],self.rawData1[5][j]/self.Tc,self.rawData1[2][j]/self.Tc, self.rawData1[1][j],self.rawData1[3][j]))
            
        str1 = ''.join(list1)
        if self.ramp == -1:
            path1 = self.impDir + 'Python Analysis\\' + '0' + str(self.date) + 'CF ' + str(self.Ppsi) +' psi cooling.dat'
        if self.ramp == 1:
            path1 = self.impDir + 'Python Analysis\\' + '0' + str(self.date) + 'CF ' + str(self.Ppsi) +' psi warming.dat'
        with open(path1,'w') as file1:
            file1.write(str1)
        
        list2=[]
        for j in range(-1,len(self.rawData2[0]) ):
            if j == -1:
                list2.append("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format('#[ 1 ] Universal Time (s) ','#[ 2 ] T Local (mK)', '#[ 3 ] Tl / Tc', '#[ 4 ] Tmc / Tc', '#[ 5 ] Q', '#[ 6 ] Inferred Frequency (Hz)') )
            else:
                if self.rawData2[7][j] != -0.5 :
                    list2.append("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(self.rawData2[0][j],self.rawData2[7][j],self.rawData2[7][j]/self.Tc,self.rawData2[2][j]/self.Tc, self.rawData2[1][j], self.rawData2[3][j]))
            
        str2 = ''.join(list2)
        if self.ramp == -1:
            path2 = self.impDir + 'Python Analysis\\' + '0' + str(self.date) + 'FF '  + str(self.Ppsi) +' psi cooling.dat'
        if self.ramp == 1:
            path2 = self.impDir + 'Python Analysis\\' + '0' + str(self.date) + 'FF '  + str(self.Ppsi) +' psi warming.dat'
        with open(path2,'w') as file2:
            file2.write(str2)
        
#%% Choose a
a = 4

#%% 6/19    

if a == 1:
    end = 1000000
    P1 = pulse(425,-1, 1, 15266, 26500, 10, 100, 619)
    # 15985 to 26500
    # psi, cooling (-1) or warming (1) ? ,done cutting?, start point, stop point, cut from start of pulse, cut from end of pulse
    # 2nd to last number is date
    # last number is done with data treatment
    # If you don't know the end point, start with the variable end in its place, adjust from there as needed
    
    del end
    
    if P1.doneCutting == 1  :
        
        P1.temp_fit(1)
        print(P1.T_fit)
        P1.QtotimeF1(7)
        print(P1.fit_QtoF1)
        print(P1.step,P1.timeAB)
        
        P1.reCallibrateF2(750)
#        750
        P1.k_NNsearch(0.05)
        
        P1.savetofile()
        
        
#%% 613

if a == 2:
    end = 1000000
    P2 = pulse(363,-1, 1, 15487, 25500, 10, 100, 613)
    # 15487 to 25500
    # psi, cooling (-1) or warming (1) ? ,done cutting?, start point, stop point, cut from start of pulse, cut from end of pulse
    # 2nd to last number is date
    # last number is done with data treatment
    # If you don't know the end point, start with the variable end in its place, adjust from there as needed

    del end
    
    if P2.doneCutting == 1  :
    
        P2.temp_fit(1)
        print(P2.T_fit)
        P2.QtotimeF1(7)
        print(P2.fit_QtoF1)
        
        P2.reCallibrateF2(1080)
#        1080
        P2.k_NNsearch(0.05)
        P2.savetofile()
  

#%% 612

if a == 3:
    end = 1000000
    P3 = pulse(303.5,-1, 1, 13289, 14474, 10, 100, 612)
    #  13289 to 15000
    # psi, cooling (-1) or warming (1) ? ,done cutting?, start point, stop point, cut from start of pulse, cut from end of pulse
    # 2nd to last number is date
    # last number is done with data treatment
    # If you don't know the end point, start with the variable end in its place, adjust from there as needed

    del end
    
    if P3.doneCutting == 1  :
    
        P3.temp_fit(2)
        print(P3.T_fit)
        P3.QtotimeF1(7)
        print(P3.fit_QtoF1)
        
        P3.reCallibrateF2(272)
#        272
        P3.k_NNsearch(0.05)
        P3.savetofile()



#%% 602
if a == 4:
    end = 1000000
    P4 = pulse(306,-1, 0, 7540, 9400, 10, 100, 602)
    #  7540 to 9400
    # psi, cooling (-1) or warming (1) ? ,done cutting?, start point, stop point, cut from start of pulse, cut from end of pulse
    # 2nd to last number is date
    # last number is done with data treatment
    # If you don't know the end point, start with the variable end in its place, adjust from there as needed

    del end
    
    if P4.doneCutting == 1  :
    
        P4.temp_fit(2)
        print(P4.T_fit)
        P4.QtotimeF1(7)
        print(P4.fit_QtoF1)
        
        P4.reCallibrateF2(272)
#        272
        P4.k_NNsearch(0.05)
        P4.savetofile()

#%% RUN THIS SECTION ONLY TO VIEW FINAL RESULTS
        # Run via ctrl + Enter
                
        fig1 = plt.figure(100, clear = True)
        ax1 = fig1.add_subplot(111)
        ax1.set_ylabel('Tlocal / Tc')
        ax1.set_xlabel('Tmc / Tc')
        ax1.set_title('Fork 2 Local Temperature vs Melting Curve Temperature for 6/19')
        ax1.scatter(( P1.rawData2[2][P1.d] )/P1.Tc,(P1.rawData2[7][P1.d])/P1.Tc, color='blue',s=0.5)
        ax1.annotate('AB Transition in fork 1', xy=(0.756, 0.754), xytext=(0.85, 0.7),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     )
        axes = plt.gca()
        axes.set_ylim([0.6,1.1])
            
        plt.show()    
        
        fig1 = plt.figure(200, clear = True)
        ax1 = fig1.add_subplot(111)
        ax1.set_ylabel('Tfit / Tc')
        ax1.set_xlabel('Q')
        ax1.set_title('Fork 1 Q to Tfit / Tc for 6/19')
        ax1.scatter(P1.rawData1[1][P1.d],(P1.rawData1[5][P1.d])/P1.Tc, color='blue',s=0.5)
        
        plt.show()    
        

