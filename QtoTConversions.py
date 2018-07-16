# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 11:01:42 2018

@author: John Wilson

"""
import numpy as np
import matplotlib.pyplot as plt

def tempC(Ppsi) :
    ''' This function takes our direct reading of pressure (in psi) and gives us the expected Tc'''
    Pbar = Ppsi * 0.0689476
    # Greywall Paper Coeffecients
    a0 = 0.92938375
    a1 = 0.13867188
    a2 = -0.0069302185
    a3 = 0.00025685169
    a4 = -0.0000057248644
    a5 = 0.000000053010918
    
    tc = a0 + a1 * (Pbar) + a2 * (Pbar**2) + a3 * (Pbar**3) + a4 * (Pbar**4) + a5 * (Pbar**5)
    
    return(tc)

def finalPlot(Pulse):
    '''This function plots final data after it has been saved'''
    fig1 = plt.figure(100, clear = True)
    ax1 = fig1.add_subplot(111)
    ax1.set_ylabel('Tlocal / Tc')
    ax1.set_xlabel('Tmc / Tc')
    ax1.set_title('Fork 2 Local Temperature vs Melting Curve Temperature')
    ax1.scatter(( Pulse.rawData2[2][Pulse.d] )/Pulse.Tc,(Pulse.rawData2[6][Pulse.d])/Pulse.Tc, color='blue',s=0.5)
    axes = plt.gca()
    axes.set_ylim([0.6,1.1])
    plt.show()    
    
    fig1 = plt.figure(200, clear = True)
    ax1 = fig1.add_subplot(111)
    ax1.set_ylabel('Tlocal / Tc')
    ax1.set_xlabel('Q')
    ax1.set_title('Fork 2 Q to Tlocal / Tc')
    ax1.scatter(Pulse.rawData2[1][Pulse.d],(Pulse.rawData2[6][Pulse.d])/Pulse.Tc, color='blue',s=0.5)
    axes = plt.gca()
    axes.set_ylim([0.6,1.1])
    plt.show()    
    
    fig1 = plt.figure(300, clear = True)
    ax1 = fig1.add_subplot(211)
    ax1.set_ylabel('Q')
    ax1.set_xlabel('point #')
    ax1.set_title('Fork 1 Q and Fork 2 Q final to Point #')
    ax1.scatter(np.linspace(1,len(Pulse.rawData1[1]),len(Pulse.rawData1[1])),Pulse.rawData1[1], color='blue',s=0.5)
    ax1 = fig1.add_subplot(212)
    ax1.set_ylabel('Q')
    ax1.set_xlabel('point #')
    ax1.scatter(np.linspace(1,len(Pulse.rawData2[1]),len(Pulse.rawData2[1])) - 388 ,Pulse.rawData2[1], color='blue',s=0.5)
    plt.show()    
    
class pulse:
    def __init__(self, Ppsi, *nums):
        # Input arguments are defined here
        self.Ppsi = Ppsi
        self.Pbar = self.Ppsi * 0.0689475729
        self.Tc = tempC(self.Ppsi)
        self.ramp = nums[0]
        # Ramp is either 1 or -1 for warming or cooling.
        # This is used to reverse the order of data in warming so warming
        # and cooling ramps can be treated the same
        self.doneCutting = nums[1]
        # This is used to alternate between plots being displayed. 
        # Prior to data treatment, during the cutting stage this is 1 and everything
        # will be plotted against point #'s
        self.start = nums[2]
        self.stop = nums[3]
        # Initial arguments for cutting
        self.ne1 = nums[4]
        self.ne2 = nums[5]
        # Number of points to be removed before/after pulses
        self.date = nums[6]
        # Date used to choose which data file to import into the path
        self.impDir = "C:\\Users\\John Wilson\\Documents\\Cornell REU\\Data\\All Data\\"
        # This import directory should contain all the folders that contain the date(s)
        # you want to look at
        if self.date == 619 : # If loop to choose the date to load/analyze
            # Fork 1
            self.path1 = [self.impDir+"20180619\\CF1p0mK.dat",self.impDir+"20180620\\CF2p1mK.dat"]
            # Fork 2
            self.path2 = [self.impDir+"20180619\\FF1p0mK.dat",self.impDir+"20180620\\FF2p1mK.dat"]        
        elif self.date == 613 :            
            # Fork 1
            self.path1 = [self.impDir+"20180613\\CF2p0mK.dat",self.impDir+"20180614\\CF2p2mK.dat",self.impDir+"20180615\\CF2p2mK.dat"]                
            # Fork 2
            self.path2 = [self.impDir+"20180613\\FF2p0mK.dat",self.impDir+"20180614\\FF2p2mK.dat",self.impDir+"20180615\\FF2p2mK.dat"]            
        elif self.date == 612 :            
            # Fork 1
            self.path1 = [self.impDir+"20180612\\CF2p1mK.dat"]                
            # Fork 2
            self.path2 = [self.impDir+"20180612\\FF2p1mK.dat"]
        elif self.date == 602 :
            # Fork 1
            self.path1 = [self.impDir+"20180602\\CF2p3mK.dat",self.impDir+"20180603\\CF2p2mK.dat"]
            # Fork 2
            self.path2 = [self.impDir+"20180602\\FF2p3mK.dat",self.impDir+"20180603\\FF2p2mK.dat"]
        elif self.date == 428:
            # Fork 1
            self.path1 = [self.impDir+"20180427\\CF2p5mK.dat", self.impDir+"20180428\\CF2p3mK.dat",self.impDir+"20180429\\CF2p3mK.dat"]
            # Fork 2
            self.path2 = [self.impDir+"20180427\\FF2p5mK.dat", self.impDir+"20180428\\FF2p3mK.dat",self.impDir+"20180429\\FF2p3mK.dat"]
        elif self.date == 705:
            # Fork 1
            self.path1 = [self.impDir+"20180703\\CF153p3mK.dat",self.impDir+"20180704\\CF15p4mK.dat",self.impDir+"20180705\\CF1p7mK.dat"]
            # Fork 2
            self.path2 = [self.impDir+"20180703\\FF153p3mK.dat",self.impDir+"20180704\\FF15p4mK.dat",self.impDir+"20180705\\FF1p7mK.dat"]
        elif self.date == 504:
            # Fork 1
            self.path1 = [self.impDir+"20180504\\CF1p5mK.dat"]
            # Fork 2
            self.path2 = [self.impDir+"20180504\\FF1p5mK.dat"]
            
        self.rawData1,self.rawData2 = self.importData(self.path1),self.importData(self.path2) # import fork 1, fork 2 
        
        if self.doneCutting == 0 :    
            n1s = [np.float64( range( 1, len( self.rawData1[0] ) + 1 ) )]
            self.rawData1 = np.concatenate( (self.rawData1, n1s), 0 )
            n2s = [np.float64( range( 1, len( self.rawData2[0] ) + 1 ) )]
            self.rawData2 = np.concatenate( (self.rawData2, n2s), 0 )
            self.cutFunction()
        elif self.doneCutting == 1:            
            if self.ramp == 1:
                for i in range( 1, len( self.rawData1 ) - 1  ) :
                    self.rawData1[i] = self.rawData1[i][::-1]                
                for i in range( 1, len( self.rawData2 )  ) :
                    self.rawData2[i] = self.rawData2[i][::-1]
            n1s = [np.float64( range( 1, len( self.rawData1[0] ) + 1 ) )]
            self.rawData1 = np.concatenate( (self.rawData1, n1s), 0 )
            n2s = [np.float64( range( 1, len( self.rawData2[0] ) + 1 ) )]
            self.rawData2 = np.concatenate( (self.rawData2, n2s), 0 )            
            self.d, self.dtemp = self.filterPulses()            
            # dtemp gives a logic string for removing NaN from temperature data
            # dtemp also removes points with pulsing near by                                    
        
    def importData(self,path):        
        ''' Function for importing the data in a given path'''
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
        # self.start and stop defined by the initialization of the class.
        # These are meant to be used in junction with the self.doneCutting variable
        # to make the cutFunction display plots that will help cutting go faster
        t0=data[0][0]        
        data[0]=data[0]-t0    
        return(data)
    
    def cutFunction(self):        
        '''This function plots Temperature, Fork 1 Q, and Fork 2 Q vs Point #
         to help with cutting. start and stop points should be changed with 
         this function being used to plot inbetween changes '''
        fig1 = plt.figure(1, clear = True)
        ax1 = fig1.add_subplot(131)
        ax1.set_ylabel('Tmc (mK)')
        ax1.set_title('Tmc and Q(F1 & F2) vs Point # for cutting for '+ str(self.date))
        line, = ax1.plot(self.rawData1[4] + self.start, self.rawData1[2], color='blue', lw=2)
        ax1 = fig1.add_subplot(132)
        ax1.set_ylabel('Fork 1 Q')
        ax1.scatter( self.rawData1[4]+self.start, self.rawData1[1], color='blue', s=0.5)
        axes = plt.gca()
        axes.set_ylim([0,max(self.rawData1[1])])
        ax1 = fig1.add_subplot(133)
        ax1.set_ylabel('Fork 2 Q')
        ax1.set_xlabel('Point #')
        ax1.scatter( self.rawData1[4]+self.start, self.rawData2[1], color='blue', s=0.5)
        axes = plt.gca()
        axes.set_ylim([0,max(self.rawData1[1])])
        plt.show()
        
    def filterPulses(self) :
        ''''This function filters pulses when pulsing occurs in fork 2. It hasn't been vigurously tested yet as of 07/13/2018'''
        a = np.where( abs( self.rawData2[1] ) > 1500 )[0]
        # np.where function creates a vector of points greater than 2000 in Q
        b = set()
        # this creates a set of numbers to use for b, sets do not allow repeat entries
        for i in a:
            for j in range( -self.ne1, self.ne2 ):
                b.add( i + j )
                # for a point in i, j points to either side are recorded in b
                # Using the add function instead of append skips double numbers
        b = list( b )
        # converts b to a list so it works with np.isin function
        d = np.isin( range( 0, len(self.rawData2[0]) ), b, invert = True)
        na = np.where(np.isnan(self.rawData1[2]))
        # na is a list of NaN points in our Tmc.
        # This is especially useful under 1 mK data when labview is likely to generate NaN Temperatures
        d1 = np.isin(range(0,len(self.rawData1[2])),na,invert = True)
        dtemp = d & d1        
        # creates a list of points that dont have pulsing near them, and removes NaN
        # no need for the assume unique state anymore because of our use of b being a set
        fig1 = plt.figure(1, clear = True)
        # Creating a plot of both fork's Qs to check that pulsing has been removed
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
        '''This function fits melting curve temperature and adds an offset so it
         can be used as the local temperature for Fork 1'''
        w = np.ones(len(self.rawData1[2]))
#        w[int(len(w)/2):]=2
        # Adjust w to adjust the weighting of the fit, if left commented there will be no weighting
        fit = np.polyfit(self.rawData1[0][self.d],self.rawData1[2][self.d],deg,w=w[self.d])
        # Generates a fit from Tmc as a function of time
        fit_fn = np.poly1d( fit )
        fitTemp = fit_fn(self.rawData1[0][self.d])
        # Generates a set of temperatures from a time input
        dt = self.Tc - np.mean(fitTemp[1:30])        
        # Takes the difference between the theoretical Tc and the point we cut to (Which should be the real Tc)
        fit[-1] += dt # The last element of the fit (constant offset) is shifted by dt
        fitTemp = fit_fn(self.rawData1[0][self.d])
        # This generates a new set of temperatures, this time shifted to the real Tc
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
        # np.vstack is used to save the new (local) temperature for Fork 1 into the self.rawData1 array
        # Row 5 (index 4) is now the corrected temperature for rawData1 (Fork 1)
        # Row 3 (index 2) is still the Tmc temperature (in both rawData1 and rawData2)        
        fit1 = tuple(fit)
        # fit saved as tuple to avoid accidental changes to the fit
        self.T_fit = fit1 # saved as a self. variable to make it easier to see the fit later
        
    def QtotimeF1(self,npol1):
        ''' Fit Fork 1 with a polynomial + step function for Q(t) '''
        tempData1 = self.rawData1[1] 
        # Creates a temporary version of Q values within fork 1
        #  that we can adjust without fear of losing our original data
        self.timeAB = 0 # Initializes our variables for the step function because
        self.step = 0   # Python doesn't like to fit non-linear models (i.e. Stepfunctions)
        if self.Pbar >= 21.22 and self.date != 705 :
            # If Pbar is below the pcp then there will be no AB transition to identify in fork 1
            # Date 705 is also excluded because CF has no identifiable AB in that data range
            for i in range( 40, len(tempData1) - 40 ): # Identify the AB transition
                # Range doesn't actually matter, it won't lead to the function
                # calling a point for averaging thats outside the index
                avg_1 = sum(tempData1[(i + 1):(i + 21)]) / np.float64(len(tempData1[(i + 1):(i + 21)]))
                avg_2 = sum(tempData1[(i - 20):i]) / np.float64(len(tempData1[(i - 20):i]))
                # Averages are computed to avoid noise from being flagged as the AB transition
                if abs(avg_1 - avg_2) > abs(self.step) :
                    self.step = avg_1 - avg_2
                    # Step is set to the difference between these two
                    self.pointAB = self.rawData1[4][i] # Point and time are saved as self. variables to help
                    self.timeAB = self.rawData1[0][i]  # us identify AB later, outside of Python if needed
                    
        tempData1 = tempData1 - self.step * np.heaviside( self.rawData1[0] - self.timeAB, 0 )    
        # subtract off the heavisidetheta function and fit the remaining data with a polynomial
        # This (hopefully) keeps Python from trying to fit a discontinuous function
        w = np.ones(len(tempData1))
#        w[0:45] = 3
        # Turn on weighting here to givea higher weight to early points near Tc
        # Adjust w to adjust the weighting of the fit
        qfit = np.polyfit(self.rawData1[0][self.d],tempData1[self.d],npol1,w = w[self.d])
        # This generates fit of the temporary data to the original time, then the
        # step function can be added into that fit later so it works on our rawData1
        qfit = tuple(qfit)
        # Saving the fit as a tuple to avoid it accidentally being overwritten
        qfit_fn = np.poly1d(qfit) # Q
        # Generates a fit for the temporary data 
        Q = qfit_fn(self.rawData1[0][self.d])
        # This generates data based off our fit from the temporary data
        Qplot = Q + self.step * np.heaviside( self.rawData1[0] - self.timeAB, 0 )
        # Adding in the step function to be able to compare it to the true data
        # Plotting to check if Q fit to time is good        
        fig1 = plt.figure(3, clear = True)
        ax1 = fig1.add_subplot(111)
        ax1.set_ylabel('Q')
        ax1.set_xlabel('point #')
        ax1.set_title('Fork 1\'s Q and Q fitted vs point # ( Both from Fork 1 ) for ' + str(self.date) )
        ax1.scatter(self.rawData1[4][self.d], self.rawData1[1][self.d], color='blue',s=0.5)
        ax1.plot(self.rawData1[4][self.d], Qplot, color='red',lw=1)
        plt.grid()
        plt.show()
        self.fit_QtoF1 = qfit
        # Saves as a self. variable to make it easier to see and use the fit later
        
    def reCallibrateF2(self,shift):
        qfit = self.fit_QtoF1
        qfit_fn = np.poly1d(qfit) # Q
        Q = qfit_fn(self.rawData1[0][self.d])
        Qplot = Q + self.step * np.heaviside( self.rawData1[0] - self.timeAB, 0 )
        # Generates data to be plotted from the fit of our original Q in fork 1.
        self.p0 = shift + 1
        # plus one here to make sure we don't mess up any indexing later on
        # p0 is saved as an attribute to the class because its later used
        # to save time in indexing within a binary search
        q0 =  Qplot[0] - self.rawData2[1][shift]            
        # q0 created to show the shift entered into the command
        self.rawData2[4] = self.rawData2[4] - self.p0
        # We can edit rawData2[4] because it is a list of numbers we generated anyway
        self.rawData2 = np.vstack((self.rawData2, self.rawData2[1] + q0 ))
        # Add new rows for corrected q value to be used later in the k-NN search
        # Plots the shifted Fork 2 data to compare with the Fork 1 fit and match Tc points
        fig1 = plt.figure(4, clear = True)
        ax1 = fig1.add_subplot(111)
        ax1.set_ylabel('Q')
        ax1.set_xlabel('point #')
        ax1.set_title('Fork 2 Q vs point # and Fork 1 fit for '+ str(self.date))
        ax1.scatter(self.rawData2[4], self.rawData2[5], color='blue',s=0.5)
        line, = ax1.plot(self.rawData1[4], Qplot, color='red', lw=1)
        ax1.annotate('Tc points should match here', xy=(0, Qplot[0]), xytext=(self.rawData2[4][-500], self.rawData2[5][500]))
        # xytext point needs to be adjusted sometimes because the index location might be too big, or not big enough if it covers data
        plt.show()    
    
    def k_NNsearch(self,tol=0.05):
        '''Enter a tolerance and the search will carry out a closest nearest neighbor search,
         implimented via a binary search to convert Q in fork 2 to Temperature'''
        # Print functions are used heavily in this function because the
        # program is prone to getting stuck inside this function.
        print('Starting k-NN Search')
        qfit = self.fit_QtoF1
        qfit_fn = np.poly1d(qfit) # Q
        Q = qfit_fn(self.rawData1[0][self.d])
        u = 0
        # If an AB point occurs in CF we make sure to use the step function fit
        if self.Pbar >= 21.22:
            u = self.step * np.heaviside( self.rawData1[0] - self.timeAB, 9000 )
            # The sceond argument of np.haeviside changed to let us skip over the point
            # where the AB transition occurs, otherwise the function would get stuck there
            print('setting step function')
        Qplot = Q + u
        newTemp = np.full(len(self.rawData2[5]), -0.5)
        # Since temperature is never negative, intializing an array this way
        # will let us skip these points later on
        for i in range(int(self.p0), len(self.rawData2[5])):
            first = 0
            last = len( self.rawData2[5] ) - 1
            a = True
            if Qplot[i] > 9000 or self.rawData2[5][i] < Qplot[0]:
                # This skips the intial points that have no Q value to map to,
                # as well as the one point on the CF AB transition
                newTemp[i] = newTemp[i-1]
                a = False
                
            if self.rawData2[5][i] > Qplot[-1]:
                a = False
                #This skips points after the last Qplot point
            while a == True:
                m = (first + last) / 2
                m = int(m)
                # This sets the midpoint for the search
#                    Inclue this in the if in order to use a tolerance for the k-NN search
#                    abs( Qplot[m] - self.rawData2[5][i] ) < tol
                if abs(first - last) <= 3 :
                    newTemp[i] = self.rawData1[5][m]
                    a = False
                    del m
                    print(i, 'completed')
                elif Qplot[m] > self.rawData2[5][i] :
                    last = m + 1
                elif Qplot[m] < self.rawData2[5][i]:
                    first = m - 1
        
        self.rawData2 = np.vstack(( self.rawData2, newTemp ))

    def savetofile(self):
        if self.ramp == 1:
            for i in range( 1, len( self.rawData1 )  ) :
                self.rawData1[i] = self.rawData1[i][::-1]
                
            for i in range( 1, len( self.rawData2 )  ) :
                self.rawData2[i] = self.rawData2[i][::-1]
                
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
                if self.rawData2[6][j] != -0.5 :
                    list2.append("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(self.rawData2[0][j],self.rawData2[6][j],self.rawData2[6][j]/self.Tc,self.rawData2[2][j]/self.Tc, self.rawData2[1][j], self.rawData2[3][j]))
            
        str2 = ''.join(list2)
        if self.ramp == -1:
            path2 = self.impDir + 'Python Analysis\\' + '0' + str(self.date) + 'FF '  + str(self.Ppsi) +' psi cooling.dat'
        if self.ramp == 1:
            path2 = self.impDir + 'Python Analysis\\' + '0' + str(self.date) + 'FF '  + str(self.Ppsi) +' psi warming.dat'
        with open(path2,'w') as file2:
            file2.write(str2)
        
#%% Choose a
a = 3
# 1, 2, 3, 4, 5 or 6 for cooling at dates
# 12, 14, or 15 for warming at dates
#%% 6/19    

if a == 1:
    end = 1000000
    P1 = pulse(425,-1, 1, 15266, 26500, 10, 100, 619)
    # 15985 to 26500
    # psi, cooling (-1) or warming (1) ? ,done cutting?, start point, stop point, cut from start of pulse, cut from end of pulse
    # 2nd to last number is date, last number is a guess for AB point if QFit can't find it
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
    # 2nd to last number is date, last number is a guess for AB point if QFit can't find it
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

if a == 12:
    end = 1000000
    P12 = pulse(363, 1, 1, 30352, 43560, 10, 100, 613)
    # 30352 to 43560
    # psi, cooling (-1) or warming (1) ? ,done cutting?, start point, stop point, cut from start of pulse, cut from end of pulse
    # 2nd to last number is date, last number is a guess for AB point if QFit can't find it
    # last number is done with data treatment
    # If you don't know the end point, start with the variable end in its place, adjust from there as needed

    del end
    
    if P12.doneCutting == 1  :
    
        P12.temp_fit(2)
        print(P12.T_fit)
        P12.QtotimeF1(8)
        print(P12.fit_QtoF1)
        
        P12.reCallibrateF2(801)
#        801
        P12.k_NNsearch(0.05)
        P12.savetofile()
        finalPlot(P12)
#%% 612

if a == 3:
    end = 1000000
    P3 = pulse(303.5,-1, 1, 13289, 14474, 10, 100, 612)
    #  13289 to 15000
    # psi, cooling (-1) or warming (1) ? ,done cutting?, start point, stop point, cut from start of pulse, cut from end of pulse
    # 2nd to last number is date, last number is a guess for AB point if QFit can't find it
    # last number is qfit cant identify a range for AB
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
        finalPlot(P3)



#%% 602
if a == 4:
    end = 1000000
    P4 = pulse(306,-1, 1, 7540, 9400, 10, 100, 602)
    #  7540 to 9400
    # psi, cooling (-1) or warming (1) ? ,done cutting?, start point, stop point, cut from start of pulse, cut from end of pulse
    # 2nd to last number is date, last number is a guess for AB point if QFit can't find it
    # last number is done with data treatment
    # If you don't know the end point, start with the variable end in its place, adjust from there as needed

    del end
    
    if P4.doneCutting == 1  :
    
        P4.temp_fit(1)
        print(P4.T_fit)
        P4.QtotimeF1(11)
        print(P4.fit_QtoF1)
#        
        P4.reCallibrateF2(638)
#        638
        P4.k_NNsearch(0.05)
        P4.savetofile()

if a == 14:
    end = 1000000
    P14 = pulse(306,1, 1, 11000, 16949, 10, 100, 602)
    #  11000 to 16949
    # psi, cooling (-1) or warming (1) ? ,done cutting?, start point, stop point, cut from start of pulse, cut from end of pulse
    # 2nd to last number is date, last number is a guess for AB point if QFit can't find it
    # last number is done with data treatment
    # If you don't know the end point, start with the variable end in its place, adjust from there as needed

    del end
    
    if P14.doneCutting == 1  :
    
        P14.temp_fit(1)
        print(P14.T_fit)
        P14.QtotimeF1(11)
        print(P14.fit_QtoF1)
#        
        P14.reCallibrateF2(1615)
#        1615
        P14.k_NNsearch(0.05)
        P14.savetofile()
        finalPlot(P14)
#%% 428
if a == 5:
    end = 1000000
    P5 = pulse(320,-1, 1, 17490, 27990, 10, 100, 428)
    #  8900 to 28200
    # psi, cooling (-1) or warming (1) ? ,done cutting?, start point, stop point, cut from start of pulse, cut from end of pulse
    # 2nd to last number is date, last number is a guess for AB point if QFit can't find it
    # last number is done with data treatment
    # If you don't know the end point, start with the variable end in its place, adjust from there as needed

    del end
    
    if P5.doneCutting == 1  :
    
        P5.temp_fit(1)
        print(P5.T_fit)
        P5.QtotimeF1(7)
        print(P5.fit_QtoF1)
#        
        P5.reCallibrateF2(2974)
#        2974
        P5.k_NNsearch(0.05)
        P5.savetofile()
        finalPlot(P5)

if a == 15:
    end = 1000000
    P15 = pulse(320,1, 1, 31333, 39653, 10, 100, 428)
    #  8900 to 28200
    # psi, cooling (-1) or warming (1) ? ,done cutting?, start point, stop point, cut from start of pulse, cut from end of pulse
    # 2nd to last number is date, last number is a guess for AB point if QFit can't find it
    # last number is done with data treatment
    # If you don't know the end point, start with the variable end in its place, adjust from there as needed

    del end
    
    if P15.doneCutting == 1  :
    
        P15.temp_fit(1)
        print(P15.T_fit)
        P15.QtotimeF1(8)
        print(P15.fit_QtoF1)
        
        P15.reCallibrateF2(2109)
        2109
        P15.k_NNsearch(0.05)
        P15.savetofile()
        finalPlot(P15)

#%% 7/05

if a == 60:
    end = 1000000
    P6 = pulse(313.5, -1, 1, 51250, 52763, 10, 100, 705)
    # 32885 to 33390 for one ramp done
    # 48374 to 49490 for another possibility done
    # 51250 to 52763 for another possibility
    # psi, cooling (-1) or warming (1) ? ,done cutting?, start point, stop point, cut from start of pulse, cut from end of pulse
    # 2nd to last number is date, last number is a guess for AB point if QFit can't find it
    # last number is done with data treatment
    # If you don't know the end point, start with the variable end in its place, adjust from there as needed

    del end

if P6.doneCutting == 1  :

    P6.temp_fit(1)
#        print(P6.T_fit)
#        P6.QtotimeF1(4)
#        print(P6.fit_QtoF1)
    
#        P6.reCallibrateF2(410)
#        410
#        P6.k_NNsearch(0.05)
#        P6.savetofile()
#        finalPlot(P6)

#%% 5/01


if a == 70:
    end = 1000000
    P7 = pulse(320, -1, 0, 0, end, 10, 100, 504)
    #  8900 to 28200
    # psi, cooling (-1) or warming (1) ? ,done cutting?, start point, stop point, cut from start of pulse, cut from end of pulse
    # 2nd to last number is date, last number is a guess for AB point if QFit can't find it
    # last number is done with data treatment
    # If you don't know the end point, start with the variable end in its place, adjust from there as needed
    
    del end

    if P7.doneCutting == 1  :

        P7.temp_fit(1)
#        print(P7.T_fit)
#        P7.QtotimeF1(7)
#        print(P7.fit_QtoF1)
#        
#        P7.reCallibrateF2(0)
#        2974
#        P7.k_NNsearch(0.05)
#        P7.savetofile()
#        finalPlot(P7)
