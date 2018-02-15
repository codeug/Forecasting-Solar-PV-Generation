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
from statsmodels.tsa.stattools import adfuller
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from sklearn.metrics import mean_squared_error

import seaborn as sb 
from statsmodels.tsa.seasonal import seasonal_decompose
import data_plot as dp
import arima_model_fit as ar

###########Loading the data into dataframes############################################################################################################################
y = 2015
new_data = pd.DataFrame()
sample_times = []

# From Jan to July
for y in range(2016,2017,1):
    print (y)
    for m in range(10,13,1):
        no_of_days = calendar.monthrange(2014,m)[1]

        for d in range (1,no_of_days+1,1):
##        for d in range (1,2,1):

#            data = pd.read_csv("C:\\Users\\ahilan\\Dropbox\\Research\\Solar Forecast\\Solar Asia 2018\\Data\\Year %d\\D120318_%d%02d%02d_0000.csv"%(y,y,m, d));
            data = pd.read_csv("C:\\Users\\kahil\\Documents\\Dropbox\\Dropbox\\Research\\Solar Forecast\\Solar Asia 2018\\Data\\Year %d\\D120318_%d%02d%02d_0000.csv"%(y,y,m, d));
            
            if (pd.to_datetime(data['Date/time'][2]) -pd.to_datetime(data['Date/time'][1])).seconds ==600:
                new_data_temp = data[['Date/time','Anemometer;wind_speed;Avg','Wind Vane;wind_direction;Avg','Hygro/Thermo;humidity;Avg', 'Hygro/Thermo;temperature;Avg','Barometer;air_pressure;Avg','Pyranometer-Diffused;solar_irradiance;Avg', 'Pyranometer-Global;solar_irradiance;Avg', 'Silicon;voltage;Avg']][0:144].copy()
                new_data = new_data.append(new_data_temp)

                for i in range(len(new_data_temp)):
                    sample_times.append(datetime.datetime(y, m, d, 6, 00, 0)+ i*datetime.timedelta(minutes=10))
        
            elif (pd.to_datetime(data['Date/time'][2]) -pd.to_datetime(data['Date/time'][1])).seconds ==60:
                new_data_temp = data[['Date/time','Anemometer;wind_speed;Avg','Wind Vane;wind_direction;Avg','Hygro/Thermo;humidity;Avg', 'Hygro/Thermo;temperature;Avg','Barometer;air_pressure;Avg','Pyranometer-Diffused;solar_irradiance;Avg', 'Pyranometer-Global;solar_irradiance;Avg', 'Silicon;voltage;Avg']][0:1440].copy()
                new_data = new_data.append(new_data_temp)

                for i in range(len(new_data_temp)):
                    sample_times.append(datetime.datetime(y, m, d, 6, 00, 0)+ i*datetime.timedelta(minutes=1))



for y in range(2017,2018,1):
    print (y)
    for m in range(1,7,1):
        no_of_days = calendar.monthrange(2014,m)[1]

##        for d in range (1,2,1):

        for d in range (1,no_of_days+1,1):
##        for d in range (1,2,1):

#            data = pd.read_csv("C:\\Users\\ahilan\\Dropbox\\Research\\Solar Forecast\\Solar Asia 2018\\Data\\Year %d\\D120318_%d%02d%02d_0000.csv"%(y,y,m, d));
            data = pd.read_csv("C:\\Users\\kahil\\Documents\\Dropbox\\Dropbox\\Research\\Solar Forecast\\Solar Asia 2018\\Data\\Year %d\\D120318_%d%02d%02d_0000.csv"%(y,y,m, d));
            
            if (pd.to_datetime(data['Date/time'][2]) -pd.to_datetime(data['Date/time'][1])).seconds ==600:
                new_data_temp = data[['Date/time','Anemometer;wind_speed;Avg','Wind Vane;wind_direction;Avg','Hygro/Thermo;humidity;Avg', 'Hygro/Thermo;temperature;Avg','Barometer;air_pressure;Avg','Pyranometer-Diffused;solar_irradiance;Avg', 'Pyranometer-Global;solar_irradiance;Avg', 'Silicon;voltage;Avg']][0:144].copy()
                new_data = new_data.append(new_data_temp)

                for i in range(len(new_data_temp)):
                    sample_times.append(datetime.datetime(y, m, d, 6, 00, 0)+ i*datetime.timedelta(minutes=10))
        
            elif (pd.to_datetime(data['Date/time'][2]) -pd.to_datetime(data['Date/time'][1])).seconds ==60:
                new_data_temp = data[['Date/time','Anemometer;wind_speed;Avg','Wind Vane;wind_direction;Avg','Hygro/Thermo;humidity;Avg', 'Hygro/Thermo;temperature;Avg','Barometer;air_pressure;Avg','Pyranometer-Diffused;solar_irradiance;Avg', 'Pyranometer-Global;solar_irradiance;Avg', 'Silicon;voltage;Avg']][0:1440].copy()
                new_data = new_data.append(new_data_temp)

                for i in range(len(new_data_temp)):
                    sample_times.append(datetime.datetime(y, m, d, 6, 00, 0)+ i*datetime.timedelta(minutes=1))
        
######################################################################################################################################################################


############Resampling the 10mins data into hourly data#################################################################################################################
new_data.columns=['time','wind_speed','wind_dir','humidity','temperature','pressure','dhi','ghi','voltage']
sample_times_series = pd.Series(sample_times)
new_data['time'] = sample_times_series.values

new_data = new_data.reset_index().set_index('time').resample('1min').mean()

#########################################################################################################################################################################

################Plotting the time series##################################################################################################################################
#Varibale names : wind_speed, 'wind_dir','humidity','temperature','pressure','dhi','ghi','voltage'
#Units names : 'm/s', 'angle', '%', '°C', 'mbar', '$W/m^2$', '$W/m^2$', 'V'
#Titles: Wind speed, Wind direction, Humidity, Atmospheric pressure, Diffuse Horizontal Irradiation, Global Horizontal Irradiation, Silicon voltage

unit = "°C"
title = "Temperature"
parm = 'temperature'

new_data['converted_data'] = new_data['%s'%(parm)].fillna(method='ffill')
dp.plot_graph(new_data['converted_data'], unit, title)


'''
##decomposition = seasonal_decompose(data, freq=12)
###fig = plt.figure()  
##fig = decomposition.plot()  
##fig.set_size_inches(15, 8)
##plt.show()

##dp.Dickey_Fuller_Test(new_data['temperature'])

##trend = decomposition.trend
##seasonal = decomposition.seasonal 
##residual = decomposition.resid

# Take a first difference of the data, and this should help to eliminate the overall trend from the data.
##new_data['first_difference'] = new_data['temperature'] - new_data['temperature'].shift()
##dp.plot_graph(new_data['first_difference'].dropna(inplace=False), unit, title)


new_data['temperature_log'] = np.log(new_data['converted_data'])
##dp.plot_graph(new_data['temperature_log'], unit, title)

# Removing trend variations
new_data['first_log_difference'] = new_data['temperature_log'] - new_data['temperature_log'].shift()

##dp.Dickey_Fuller_Test(new_data['first_log_difference'])

##dp.plot_graph(new_data['first_log_difference'].dropna(inplace=False), unit, title)


decomposition = seasonal_decompose(new_data['temperature_log'], model='additive', freq=30)  
fig = decomposition.plot()
plt.show()

decomposition = seasonal_decompose(new_data['first_log_difference'][1:], model='additive', freq=30)  
fig = decomposition.plot()
plt.show()


ar. plot_acf_pacf(new_data['first_log_difference'].dropna(inplace=False))


##model = sm.tsa.ARIMA(new_data['temperature_log'].iloc[1:], order=(2, 1, 6))
##model = sm.tsa.ARIMA(new_data['temperature_log'], order=(2, 1, 6))
model = sm.tsa.ARIMA(new_data['temperature_log'], order=(4, 1, 8))


##model = sm.tsa.ARIMA(new_data['temperature_log'], order=(1, 1, 1))
##model = smt.SARIMAX(new_data['temperature_log'], trend='c', order=(1, 1, 1))
##results = model.fit()

results = model.fit(disp=-1)

new_data['Forecast'] = results.fittedvalues
new_data[['first_log_difference', 'Forecast']].plot(figsize=(16, 12))
plt.title('RSS: %.4f'% sum((results.fittedvalues.values-new_data['first_log_difference'][1:].values)**2))
plt.legend(loc='best')
plt.show()
# Measuring the performance
print(results.summary())
residuals = pd.DataFrame(results.resid)
residuals.plot(kind='kde')
print(residuals.describe())
plt.show()


predictions_diff = pd.Series(results.fittedvalues, copy=True)

predictions_diff_cumsum = predictions_diff.cumsum()

predictions_log = pd.Series(new_data['temperature_log'].ix[0], index=new_data['temperature_log'].index)
predictions_log = predictions_log.add(predictions_diff_cumsum,fill_value=0)

predictions = np.exp(predictions_log)
new_data['%s'%(parm)].plot(label='Original')
predictions.plot(color='red',label='AR model fitted')
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions.values-new_data['converted_data'].values)**2)/len(new_data['converted_data'].values)))
plt.show()
'''

### Separting traing and testing data(60min*24)
new_data['temperature_log'] = np.log(new_data['converted_data'])

size = int(len(new_data['temperature_log']) - 1440)
train, test = new_data['temperature_log'][0:size], new_data['temperature_log'][size:len(new_data['temperature_log'])]
history = [x for x in train]
predictions = list()
print('Printing Predicted vs Expected Values...')
print('\n')



for t in range(len(test)):
    model = sm.tsa.ARIMA(history, order=(4, 1, 8))
    model_fit = model.fit(disp=-1)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(float(yhat))
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (np.exp(yhat), np.exp(obs)))

error = mean_squared_error(test, predictions)

print('\n')
print('Printing Mean Squared Error of Predictions...')
print('Test MSE: %.6f' % error)

predictions_series = pd.Series(predictions, index = test.index)
fig, ax = plt.subplots()
ax.set(title='Temperature forecasting', xlabel='Date', ylabel='Temperature')
ax.plot(new_data['temperature_log'][-5760:], 'o', label='observed')
ax.plot(np.exp(predictions_series), 'g', label='forecast')
legend = ax.legend(loc='upper left')
legend.get_frame().set_facecolor('w')


##Playing with prophet########################################################################################################







################################################################################################################################
#dp.plot_graph(data, unit, title)
#dp.Dickey_Fuller_Test(data)


#dp.Dickey_Fuller_Test(np.log(data))

#ts = data


#The data has a strong seasonal component.
#We can neutralize this and make the data stationary by taking the seasonal difference.
#That is, we can take the observation for a day and subtract the observation from the same day one year ago.


#ts_log = np.log(ts)
#ts_log_diff = ts_log - ts_log.shift()
#ts_log_diff.dropna(inplace=True)

#ar.plot_acf_pacf(ts_log_diff)
#ar.arima_model_fit(ts_log, ts_log_diff,order=(1,1,1))

##ts_log_diff.plot(color='blue',label='Original')

#Decomposing trend, seasonal, residual
