import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime
import seaborn as sns
from scipy.stats import pearsonr
from matplotlib import cm as cm
import calendar
import warnings
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf


######ARIMA model fitting parameters ##################################################################################################################################
###The ACF and PACF plots for the TS after differencing can be plotted as:
###ACF and PACF plots:

def plot_acf_pacf(ts_log_diff):
    lag_acf = acf(ts_log_diff, nlags=10)
    lag_pacf = pacf(ts_log_diff, nlags=10, method='ols')

    #Plot ACF:
    plt.subplot(121) 
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-7.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.axhline(y=7.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')

    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-7.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.axhline(y=7.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.show()

###################################################################################################################################


def arima_model_fit(ts_log, ts_log_diff,order=(1,1,1)):
    
    ### ARMA model
    model = ARIMA(ts_log, order)  
    results_ARMA = model.fit(disp=-1)  
    ts_log_diff.plot(label='Original')
    results_ARMA.fittedvalues.plot(color='red',label='AR model fitted')

    plt.title('RSS: %.4f'% sum((results_ARMA.fittedvalues.values-ts_log_diff.values)**2))
    plt.legend(loc='best')
    plt.show()

##    # AR model
##    model = ARIMA(ts_log, order=(2, 1, 0))  
##    results_AR = model.fit(disp=-1)  
##    ts_log_diff_series.plot(label='Original')
##    results_AR.fittedvalues.plot(color='red',label='AR model fitted')
##
##    plt.title('RSS: %.4f'% sum((results_AR.fittedvalues.values-ts_log_diff_series.values)**2))
##    plt.legend(loc='best')
##    plt.show()


##    # MA model
##    model = ARIMA(ts_log_series, order=(0, 1, 2))
##    results_MA = model.fit(disp=-1)  
##    ts_log_diff_series.plot(label='Original')
##    results_MA.fittedvalues.plot(color='red',label='MA model fitted')
##
##    plt.title('RSS: %.4f'% sum((results_MA.fittedvalues.values-ts_log_diff_series.values)**2))
##    plt.legend(loc='best')
##    plt.show()

def arima_model_predict(ts_log,ts_log_diff,order=(2,1,4)):
    model = ARIMA(ts_log, order=(2, 1, 4))  
    results_ARMA = model.fit(disp=-1)  

    predictions_ARMA_diff = pd.Series(results_ARMA.fittedvalues, copy=True)
    print (predictions_ARMA_diff.head())

    predictions_ARMA_diff_cumsum = predictions_ARMA_diff.cumsum()
    print (predictions_ARMA_diff_cumsum.head())

    predictions_ARMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
    predictions_ARMA_log = predictions_ARMA_log.add(predictions_ARMA_diff_cumsum,fill_value=0)
    print(predictions_ARMA_log.head())

    predictions_ARMA = np.exp(predictions_ARMA_log)
    ts.plot(label='Original')
    predictions_ARMA.plot(color='red',label='AR model fitted')
    plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARMA.values-ts.values)**2)/len(ts.values)))
    plt.show()

#################################################################################################################################
## ARIMA testing
#mod = sm.tsa.statespace.SARIMAX(new_data['dhi'], order=(2,1,2), enforce_stationarity=False, enforce_invertibility=False)
#results = mod.fit()
#print(results.summary().tables[1])

