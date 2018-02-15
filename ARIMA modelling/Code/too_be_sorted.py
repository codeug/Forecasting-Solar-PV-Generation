###Cross-correlation measures the similarity between two dicrete simte squences x, y###################################################################################
### For a day
##fig, ax = plt.subplots(nrows=2, ncols=2)
##plt.subplot(2, 2, 1)
##plt.xcorr(new_data['wind_speed'], new_data['dhi'], usevlines=True, normed=True, lw=2)
##plt.title('Cross correlation between wind speed and DHI on 01/01/2014')
##plt.grid(True)
##plt.axhline(0, color='black', lw=2)
##
##plt.subplot(2, 2, 2)
##plt.xcorr(new_data['humidity'][0:80], new_data['dhi'][0:80], usevlines=True, normed=True, lw=2)
##plt.title('Cross correlation between humidity and DHI on 01/01/2014')
##plt.grid(True)
##plt.axhline(0, color='black', lw=2)
##
##plt.subplot(2, 2, 3)
##plt.xcorr(new_data['temperature'][0:80], new_data['dhi'][0:80], usevlines=True, normed=True, lw=2)
##plt.grid(True)
##plt.title('Cross correlation between temperature and DHI on 01/01/2014')
##plt.axhline(0, color='black', lw=2)
##
##plt.subplot(2, 2, 4)
##plt.xcorr(new_data['pressure'][0:80], new_data['dhi'][0:80], usevlines=True, normed=True, lw=2)
##plt.grid(True)
##plt.title('Cross correlation between pressure and DHI on 01/01/2014')
##plt.axhline(0, color='black', lw=2)
##plt.show()


### For a month#########################################################################################################################################
##data_series_windspeed = pd.Series(new_data['wind_speed'].values, sample_times)
##data_series_hum = pd.Series(new_data['humidity'].values, sample_times)
##data_series_temp = pd.Series(new_data['temperature'].values, sample_times)
##data_series_pres = pd.Series(new_data['pressure'].values, sample_times)
##data_series_ghi = pd.Series(new_data['ghi'].values, sample_times)
##
##fig, ax = plt.subplots(nrows=2, ncols=2)
##
##plt.subplot(2, 2, 1)
##plt.grid(True)
##plt.xcorr(data_series_windspeed.between_time('6:00','19:00'), data_series_ghi.between_time('6:00','19:00'), usevlines=True, normed=True, lw=2)
##plt.title('Cross correlation between wind speed and GHI on in 01/2014')
##plt.axhline(0, color='black', lw=2)
##
##plt.subplot(2, 2, 2)
##plt.grid(True)
##plt.xcorr(data_series_hum.between_time('6:00','19:00'), data_series_ghi.between_time('6:00','19:00'), usevlines=True, normed=True, lw=2)
##plt.title('Cross correlation between humidity and GHI on in 01/2014')
##plt.axhline(0, color='black', lw=2)
##
##plt.subplot(2, 2, 3)
##plt.grid(True)
##plt.xcorr(data_series_temp.between_time('6:00','19:00'), data_series_ghi.between_time('6:00','19:00'), usevlines=True, normed=True, lw=2)
##plt.title('Cross correlation between temperature and GHI on in 01/2014')
##plt.axhline(0, color='black', lw=2)
##
##plt.subplot(2, 2, 4)
##plt.grid(True)
##plt.xcorr(data_series_pres.between_time('6:00','19:00'), data_series_ghi.between_time('6:00','19:00'), usevlines=True, normed=True, lw=2)
##plt.title('Cross correlation between pressure and GHI on in 01/2014')
##plt.axhline(0, color='black', lw=2)
##
##
##plt.show()
##############################################################################################################################################################


####### Estimating Pearson r Correlation Coefficient#########################################################################################################
##data_series_windspeed = pd.Series(new_data['wind_speed'].values, sample_times)
##data_series_hum = pd.Series(new_data['humidity'].values, sample_times)
##data_series_temp = pd.Series(new_data['temperature'].values, sample_times)
##data_series_pres = pd.Series(new_data['pressure'].values, sample_times)
##data_series_ghi = pd.Series(new_data['ghi'].values, sample_times)
##data_series_dhi = pd.Series(new_data['dhi'].values, sample_times)
##
###### Pearson measures the linear correlation between weather parameters and GHI
##
##pearsonr(data_series_windspeed, data_series_ghi) # Output: (0.65685147374578012, 0.0)
##pearsonr(data_series_hum, data_series_ghi) # Output: (-0.81714316290770017, 0.0)
##pearsonr(data_series_temp, data_series_ghi) # Output: (0.81874399934548914, 0.0)
##pearsonr(data_series_pres, data_series_ghi) # Output: (0.22004590653393447, 2.0977881410589242e-45)
##
###### Pearson measures the linear correlation between weather parameters and GHI
##
##pearsonr(data_series_windspeed, data_series_dhi) # Output: (0.67255789555486223, 0.0)
##pearsonr(data_series_hum, data_series_dhi) # Output: (-0.68438055306327761, 0.0)
##pearsonr(data_series_temp, data_series_dhi) # Output: (0.77565603333571509, 0.0)
##pearsonr(data_series_pres, data_series_dhi) # Output: (0.20732300366962522, 2.1552825230723012e-40)
##
##
################################################################################################################################################################


#### Correlation between variable using heatmap###################################################################################################################
##def correlation_matrix(df):
##    fig = plt.figure()
##    ax1 = fig.add_subplot(111)
##    cmap = cm.get_cmap('jet', 10)
##    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
##    ax1.grid(True)
##    plt.title('Feature Correlation')
##    labels=['Wind speed', 'Humidity','Temperature','Pressure',]    
####    labels=['DHI', 'GHI','Silicon voltage',]
##    ax1.set_xticklabels(labels,fontsize=8)
##    ax1.set_yticklabels(labels,fontsize=8)
##    # Add colorbar, make sure to specify tick locations to match desired ticklabels
##    fig.colorbar(cax, ticks=[-1,-0.25,-0.5,-0.75,0.25,0.5,0.75,1.0])
##    plt.show()
##
##correlation_matrix(new_data[['wind_speed','humidity','temperature','pressure']])
####correlation_matrix(new_data[['dhi','ghi','voltage']])
##
###################################################################################################################################################################

######################### Plotting rolling mean and rolling standard deviation######################################################################################
###Varibale names : wind_speed, 'wind_dir','humidity','temperature','pressure','dhi','ghi','voltage'
###Units names : 'm/s', 'angle', '%', '°C', 'mbar', '$W/m^2$', '$W/m^2$', 'V'
###Titles: Wind speed, Wind direction, Humidity, Atmospheric pressure, Diffuse Horizontal Irradiation, Global Horizontal Irradiation, Silicon voltage
##
##pram = "wind_speed"
##unit = "m/s"
##tit = "Wind speed"
##
##rolmean = pd.rolling_mean(new_data['%s'%(pram)], window=12)
##rolstd = pd.rolling_std(new_data['%s'%(pram)], window=12)
##
##data_series = pd.Series(new_data['%s'%(pram)].values, sample_times)
##data_series.between_time('6:00','19:00').plot(color='blue',label='Original')
##
##data_series_rolmean = pd.Series(rolmean.values, sample_times)
##data_series_rolmean.between_time('6:00','19:00').plot(color='red', label='Rolling Mean')
##
###data_series_rolstd = pd.Series(rolstd.values, sample_times)
###data_series_rolstd.between_time('6:00','19:00').plot(color='black', label = 'Rolling Std')
##
##plt.legend(loc='best')
##plt.ylabel('%s/ %s'%(tit, unit))
##
##plt.title('%s rolling mean & standard deviation estimated between 06.00 and 19.00 on %02d/%02d/2014'%(tit,d,m))
##plt.show(block=False)
##plt.show()
##
##########################################################################################################################################################


####### Estimating trend and seasonal variations############################################################################################################
##
### Differencing
ts = new_data['temperature']
ts_log = np.log(new_data['temperature'])
ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)
##
##
###moving_avg = pd.rolling_mean(ts_log,12)
###ts_log_moving_avg_diff = ts_log - moving_avg
###ts_log_moving_avg_diff.dropna(inplace=True)
##
##
##ts_log_diff.plot(color='blue',label='Original')
##
###Decomposing trend, seasonal, residual
##from statsmodels.tsa.seasonal import seasonal_decompose
##
##decomposition = seasonal_decompose(ts_log,freq=1)
##trend = decomposition.trend
##seasonal = decomposition.seasonal
##residual = decomposition.resid
##
##plt.subplot(411)
##plt.plot(ts_log, label='Original')
##plt.legend(loc='best')
##plt.subplot(412)
##plt.plot(trend, label='Trend')
##plt.legend(loc='best')
##plt.subplot(413)
##plt.plot(seasonal,label='Seasonality')
##plt.legend(loc='best')
##plt.subplot(414)
##plt.plot(residual, label='Residuals')
##plt.legend(loc='best')
##plt.tight_layout()
##plt.show()
#####################################################################################################################################



######ARIMA model fitting parameters ##################################################################################################################################
###The ACF and PACF plots for the TS after differencing can be plotted as:
###ACF and PACF plots:
##from statsmodels.tsa.stattools import acf, pacf
##lag_acf = acf(ts_log_diff, nlags=20)
##lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
##
###Plot ACF:
##plt.subplot(121) 
##plt.plot(lag_acf)
##plt.axhline(y=0,linestyle='--',color='gray')
##plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
##plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
##plt.title('Autocorrelation Function')
##
##plt.subplot(122)
##plt.plot(lag_pacf)
##plt.axhline(y=0,linestyle='--',color='gray')
##plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
##plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
##plt.title('Partial Autocorrelation Function')
##plt.tight_layout()
##plt.show()

###################################################################################################################################


##from statsmodels.tsa.arima_model import ARIMA

### AR model
##model = ARIMA(ts_log, order=(2, 1, 0))  
##results_AR = model.fit(disp=-1)  
##ts_log_diff_series.plot(label='Original')
##results_AR.fittedvalues.plot(color='red',label='AR model fitted')
##
##plt.title('RSS: %.4f'% sum((results_AR.fittedvalues.values-ts_log_diff_series.values)**2))
##plt.legend(loc='best')
##plt.show()

#################################################################################################################################

### MA model
##model = ARIMA(ts_log_series, order=(0, 1, 2))
##results_MA = model.fit(disp=-1)  
##ts_log_diff_series.plot(label='Original')
##results_MA.fittedvalues.plot(color='red',label='MA model fitted')
##
##plt.title('RSS: %.4f'% sum((results_MA.fittedvalues.values-ts_log_diff_series.values)**2))
##plt.legend(loc='best')
##plt.show()

#################################################################################################################################
## ARIMA testing
#mod = sm.tsa.statespace.SARIMAX(new_data['dhi'], order=(2,1,2), enforce_stationarity=False, enforce_invertibility=False)
#results = mod.fit()
#print(results.summary().tables[1])



### ARMA model
model = ARIMA(ts_log, order=(2, 1, 4))  
results_ARMA = model.fit(disp=-1)  
ts_log_diff.plot(label='Original')
results_ARMA.fittedvalues.plot(color='red',label='AR model fitted')

plt.title('RSS: %.4f'% sum((results_ARMA.fittedvalues.values-ts_log_diff.values)**2))
plt.legend(loc='best')
plt.show()
##
##
##predictions_ARMA_diff = pd.Series(results_ARMA.fittedvalues, copy=True)
##print (predictions_ARMA_diff.head())
##
##predictions_ARMA_diff_cumsum = predictions_ARMA_diff.cumsum()
##print (predictions_ARMA_diff_cumsum.head())
##
##predictions_ARMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
##predictions_ARMA_log = predictions_ARMA_log.add(predictions_ARMA_diff_cumsum,fill_value=0)
##print(predictions_ARMA_log.head())
##
##predictions_ARMA = np.exp(predictions_ARMA_log)
##ts.plot(label='Original')
##predictions_ARMA.plot(color='red',label='AR model fitted')
##plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARMA.values-ts.values)**2)/len(ts.values)))
##plt.show()

#################################################################################################################################





##################### testing ##############################################################################################################################
##    data = pd.read_csv("C:\\Users\\ahilan\\Dropbox\\Research\\Solar Forecast\\Solar Asia 2018\\Data\\Year 2014\\D120318_2014%02d%02d_0000.csv"%(m, d))
##    new_data = data[['Date/time','Anemometer;wind_speed;Avg','Wind Vane;wind_direction;Avg','Hygro/Thermo;humidity;Avg', 'Hygro/Thermo;temperature;Avg','Barometer;air_pressure;Avg','Pyranometer-Diffused;solar_irradiance;Avg', 'Pyranometer-Global;solar_irradiance;Avg', 'Silicon;voltage;Avg']][0:144].copy()
##    new_data.columns=['time','wind_speed','wind_dir','humidity','temperature','pressure','dhi','ghi','voltage']

##data.head()
##print(data.dtypes)
##data.filter(items=['Date/time']).head()
##data.filter(items=['Hygro/Thermo;temperature;Avg']).head()
##print(list(data))

###########################################################################################################################################################
### Important wheather parameters
##Date/time, Anemometer;wind_speed;Avg, Wind Vane;wind_direction;Avg, Hygro/Thermo;humidity;Avg, Hygro/Thermo;temperature;Avg, Barometer;air_pressure;Avg,
##Pyranometer-Diffused;solar_irradiance;Avg, Pyranometer-Global;solar_irradiance;Avg, Silicon;voltage;Avg, 
##
##Statistics - Avg, Max, Min, stdDev,
############################################################################################################################################################
### Plotting the parameter at 10 mins interval
##starttime = datetime.datetime(2014, 1, 1, 6, 00, 0)
##delta = datetime.timedelta(minutes=10)
##sample_times = [starttime + i * delta for i in range(80)]
##label_locations = [d for d in sample_times if d.minute % 20 == 0]
###labels = [d.strftime('%Y-%m-%d %H:%M:%S') for d in label_locations]
##labels = [d.strftime('%H:%M') for d in label_locations]
##x = arange(144)
##plt.plot(sample_times, new_data['ghi'][0:80])
##plt.xticks(label_locations, labels, rotation=90)
###plt.ylabel('Temperature/ °C')
###plt.ylabel('Wind speed/ m/s')
###plt.ylabel('Wind direction/ angle')
###plt.ylabel('Humidity/ %')
###plt.ylabel('Atmospheric pressure/ mbar')
###plt.ylabel('Diffuse Horizontal Irradiation/ $W/m^2$')
##plt.ylabel('Global Horizontal Irradiation/ $W/m^2$')
###plt.ylabel('Silicon voltage/ V')
##
##plt.title('Global horizontal irradiation variations recorded in the interval of 10 mins on 03/01/2014')
##plt.show()

######Plotting DHI and GHI against weather parameters###########################################################################################
#Varibale names : wind_speed, 'wind_dir','humidity','temperature','pressure','dhi','ghi','voltage'
#Units names : 'm/s', 'angle', '%', '°C', 'mbar', '$W/m^2$', '$W/m^2$', 'V'
#Titles: Wind speed, Wind direction, Humidity, Atmospheric pressure, Diffuse Horizontal Irradiation, Global Horizontal Irradiation, Silicon voltage


##plt.plot(new_data['temperature'][0:60],new_data['dhi'][0:60])
##plt.ylabel('Diffuse horizontal irradiance')
##plt.show()



##ax2 = fig.add_subplot(212, sharex=ax1)
##ax2.acorr(new_data['dhi'][0:80], usevlines=True, normed=True, lw=2)
##ax2.grid(True)
##ax2.axhline(0, color='black', lw=2)

         
##pram = "wind_speed"
##unit = "m/s"
##tit = "Wind speed"
##data_series = pd.Series(new_data['%s'%(pram)].values, sample_times)
##data_series.between_time('6:00','19:00').plot()
##plt.ylabel('%s/ %s'%(tit, unit))
###plt.title('%s variations recorded between 06.00 and 19.00 in the interval of 10 mins on %02d/%02d/2014'%(tit,d,m))
##plt.title('%s variations recorded between 06.00 and 19.00 in the interval of 10 mins in %02d/2014'%(tit,m))
##plt.show()

