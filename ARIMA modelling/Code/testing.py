% x axis formatter
%%%%% x axis in 10 mins interval
%times = pd.date_range('2015-10-06', periods=144, freq='10min')
%a=[str(datetime(2014, 1, 1, hr, min, 0).time()) for hr in range(00,24) for min in range(0,60,10)]
#!/usr/bin/env python3
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#make some data
t = np.arange(0, 5664, 1) # Jan and Feb worth of 15 minute steps
s = np.sin(2*np.pi*t) # data measured

#plot the data
fig, ax = plt.subplots()
ax.plot(t, s)

#select formatting
days = mdates.DayLocator()
daysfmt = mdates.DateFormatter('%d')
months = mdates.MonthLocator()
monthsfmt = mdates.DateFormatter('\n%b')
#apply formatting
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsfmt)
ax.xaxis.set_minor_locator(days)
ax.xaxis.set_minor_formatter(daysfmt)

#select dates
datemin = dt.datetime.strptime('01/01/17', '%d/%m/%y')
datemax = dt.datetime.strptime('28/02/17', '%d/%m/%y')
ax.set_xlim(datemin, datemax)

