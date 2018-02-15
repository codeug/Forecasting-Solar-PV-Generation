import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime
from numpy import sin, arange
from math import pi

#for i in range(10):
#    print ('C:\\Users\\ahilan\\Dropbox\\Research\\Solar Forecast\\Solar Asia 2018\\Data\\Year 2014\\D120318_2014010{i}+'_0000.csv')

data = pd.read_csv("C:\\Users\\ahilan\\Dropbox\\Research\\Solar Forecast\\Solar Asia 2018\\Data\\Year 2014\\D120318_20140103_0000.csv")
new_data = data[['Date/time','Anemometer;wind_speed;Avg','Wind Vane;wind_direction;Avg','Hygro/Thermo;humidity;Avg', 'Hygro/Thermo;temperature;Avg','Barometer;air_pressure;Avg','Pyranometer-Diffused;solar_irradiance;Avg', 'Pyranometer-Global;solar_irradiance;Avg', 'Silicon;voltage;Avg']][0:144].copy()
new_data.columns=['time','wind_speed','wind_dir','humidity','temperature','pressure','dhi','ghi','voltage']

#data.loc[data['Anemometer;wind_speed;Avg'][0:144] == 0, 'B'] = df.A.shift(-1)

##data.head()
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


# Plotting the parameter at 10 mins interval
starttime = datetime.datetime(2014, 1, 1, 6, 00, 0)
delta = datetime.timedelta(minutes=10)
sample_times = [starttime + i * delta for i in range(80)]
label_locations = [d for d in sample_times if d.minute % 20 == 0]
#labels = [d.strftime('%Y-%m-%d %H:%M:%S') for d in label_locations]
labels = [d.strftime('%H:%M') for d in label_locations]
x = arange(144)
plt.plot(sample_times, new_data['ghi'][0:80])
plt.xticks(label_locations, labels, rotation=90)
#plt.ylabel('Temperature/ °C')
#plt.ylabel('Wind speed/ m/s')
#plt.ylabel('Wind direction/ angle')
#plt.ylabel('Humidity/ %')
#plt.ylabel('Atmospheric pressure/ mbar')
#plt.ylabel('Diffuse Horizontal Irradiation/ $W/m^2$')
plt.ylabel('Global Horizontal Irradiation/ $W/m^2$')
#plt.ylabel('Silicon voltage/ V')

plt.title('Global horizontal irradiation variations recorded in the interval of 10 mins on 03/01/2014')
plt.show()



