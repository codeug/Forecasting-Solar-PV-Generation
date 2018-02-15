import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime
import seaborn as sns
from scipy.stats import pearsonr
from matplotlib import cm as cm
from statsmodels.tsa.stattools import adfuller
import warnings

################Stationary check##################################################################################################################################
def plot_graph (timeseries, unit, title):    
    rolmean = pd.rolling_mean(timeseries, window=720)
    rolstd = pd.rolling_std(timeseries, window=12)

    timeseries.plot(color='blue',label='Original')
##    rolmean.plot(color='red', label='Rolling Mean')
##    plt.legend(loc='best')

    plt.ylabel('%s/ %s'%(title, unit), fontsize=16)
#    plt.xticks(size = 50)
    plt.yticks(size = 25)
    plt.title('%s variations recorded in the interval of every 1min'%(title))
    plt.show(block=False)
    plt.show()


################Stationary check##################################################################################################################################
def Dickey_Fuller_Test (timeseries):
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


