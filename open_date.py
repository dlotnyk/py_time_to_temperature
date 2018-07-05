"""
Created on Mon Jun 18 14:50:28 2018

@author: John Wilson
"""
import numpy as np
import tkinter.filedialog as tkn
import matplotlib.pyplot as plt


def chooseFile():

    tkn.Tk().withdraw() # Close the root window
    in_path = tkn.askopenfilename()
    return(in_path)


def combineData(days):
    
    filename = chooseFile()
    fullData = np.genfromtxt(filename,
                            skip_header=1,
                            skip_footer=1,
                            unpack=1,
                            usecols=(2,6,13)
                            )

    c=1
    
    while c < days :
        filename = chooseFile()
        newdata = np.genfromtxt(filename,
                                skip_header=1,
                                skip_footer=1,
                                unpack=1,
                                usecols=(2,6,13)
                                )
    
        fullData=np.concatenate((fullData,newdata),axis=1)
        c=c+1
        
    else :
        return(fullData)
    
def datPlot(days):
    
    data=combineData(days)
    fig1 = plt.figure(13, clear = True)
    ax1 = fig1.add_subplot(211)
    ax1.set_ylabel('Q')
    ax1.set_xlabel('time')
    ax1.set_title('JW')
    ax1.plot(data[0],data[1])
    plt.grid()
    ax2 = fig1.add_subplot(212)
    ax2.set_ylabel('T')
    ax2.set_xlabel('time')
    ax2.plot(data[0],data[2])
    plt.grid()
    plt.show()

datPlot(3)