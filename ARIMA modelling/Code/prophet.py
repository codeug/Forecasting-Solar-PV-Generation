import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime
import seaborn as sns
from scipy.stats import pearsonr
from matplotlib import cm as cm
from statsmodels.tsa.stattools import adfuller
import calendar
import warnings
import itertools
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import data_plot as dp
import arima_model_fit as ar
#import prophet
#from fbprophet import Prophet

###########Loading the data into dataframes############################################################################################################################

# From Jan to July


y = 2015
new_data = pd.DataFrame()
sample_times = []
for y in range(2014,2015,1):
    print (y)
    for m in range(1,13,1):
        no_of_days = calendar.monthrange(2014,m)[1]

        for d in range (1,no_of_days+1,1):
#            data = pd.read_csv("C:\\Users\\ahilan\\Dropbox\\Research\\Solar Forecast\\Solar Asia 2018\\Data\\Year %d\\D120318_%d%02d%02d_0000.csv"%(y,y,m, d));
#            data = pd.read_csv("C:\\Users\\kahil\\Documents\\Dropbox\\Dropbox\\Research\\Solar Forecast\\Solar Asia 2018\\Data\\Year %d\\D120318_%d%02d%02d_0000.csv"%(y,y,m, d));

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


#new_data = new_data.reset_index()
#df = new_data
#df.set_index('time')['temperature'].plot()
#plt.show()

new_data = new_data.reset_index().set_index('time').resample('1H').mean()







#sample_times_series = pd.Series(sample_times)
#new_data['time'] = sample_times_series.values

#new_data.reset_index().set_index('time')
#new_data.set_index('time')



'''
new_data.reset_index().set_index('time')
new_data.set_index('time')

new_data = new_data.reset_index().set_index('time').resample('1H').mean()

#########################################################################################################################################################################

################Plotting the time series##################################################################################################################################
#Varibale names : wind_speed, 'wind_dir','humidity','temperature','pressure','dhi','ghi','voltage'
#Units names : 'm/s', 'angle', '%', '°C', 'mbar', '$W/m^2$', '$W/m^2$', 'V'
#Titles: Wind speed, Wind direction, Humidity, Atmospheric pressure, Diffuse Horizontal Irradiation, Global Horizontal Irradiation, Silicon voltage

pram = "dhi"
unit = "$W/m^2$"
title = "DHI"

data = new_data['%s'%(pram)].between_time('7:00','18:00')
#data.replace(0, np.nan).fillna(method='ffill').fillna(method='bfill')

data = data[data!=0]
dp.plot_graph(data, unit, title)
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
'''



