""" Calculate Time Series PM NEUS """

#  A brief note on the PM25_go_aqs_nearest_complete.csv file:
#
#     This file was created in a previous Matlab program. It contains PM2.5 data
#     from AQS stations in the NEUS with a matchin  PM2.5 value  from  MERRAero.
#     The  MERRAero  value was selected to be from the gridpoint nearest to each
#     AQS stations. Given that we only had MERRAero PM values for the summer, we
#     only show the PM values  from  AQS  for the  summer.  There is a .csv file 
#     AQS_PM25_all_1999_2013.csv which contains the complete record of AQS PM2.5
#     for the neus for every day of the year.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
import datetime
import scipy
import pandas as pd
from   netCDF4 import Dataset
from   timeit  import default_timer as timer

start = timer()

PATH_AOD = '/Users/Rola/Documents/Science/JHU/DATA/MERRAero/'
# PATH_AOD = '/Users/Tera/Desktop/Stagnation/'

df = pd.read_csv(PATH_AOD+'PM25_go_aqs_nearest_complete.csv'); # Loads the PM2.5 data.
date=df[['Year','Month','Day']].values                         # Exctracts date info.

datenum=np.zeros((99307, 1))                                   # Creates an gregorian ordinal of date vectors.
for j in range(0,99307):
      datenum[j]=datetime.date.toordinal(datetime.date(date[j,0],date[j,1],date[j,2]))

dat=np.unique(datenum)  # Finds unique values of date vector...
dat=dat.astype(int)     # ...and converst them to integers for the following loop.

aux1=np.zeros((982,))   # vs. aux1=np.zeros((982,1)) puts brackets in values 
aux2=np.zeros((982,))
DATE=list()
k=0                     # The loop gets the data from each station and calculates the daily neus mean.
for i in dat:
      print k/982.*100
      dt=datetime.date.fromordinal(i) # Gets the data info from the loop counter.
      yr=dt.year
      mn=dt.month
      dy=dt.day
      aux1[k]=df['PM25_AQS'][(df['Year'] == yr) & (df['Month'] == mn) & (df['Day'] == dy)].mean() # calculates the PM regional mean.
      aux2[k]=df['PM25_MERRAero'][(df['Year'] == yr) & (df['Month'] == mn) & (df['Day'] == dy)].mean()
      DATE.append(dt.strftime("%Y-%m-%d"))  # generates a date string array to be put in the output file.
      k += 1

ax1=aux1.tolist()       # Prepares the output ot be put in a pandas dataframe. 
ax2=aux2.tolist()

PM25_mean_neus=pd.DataFrame({'date': DATE, 'PM25_AQS': ax1, 'PM25_MERRAero': ax2})  # Creates the pandas dataframe.
PM25_mean_neus['date']=pd.to_datetime(PM25_mean_neus['date'], format="%Y-%m-%d")  # Converts the date column into the correct format.
cols = PM25_mean_neus.columns.tolist()  # Rearranges the order of the columns.
cols = cols[-1:] + cols[:-1]
PM25_mean_neus=PM25_mean_neus[cols]

PM25_mean_neus.to_csv(PATH_AOD+'PM25_neus_mean_series.csv', columns=cols, index=0)  # Saves the dataframe to a .csv file.

end = timer(); print "I've read and processed the files in", round(end - start), "seconds."

#-------------------------------------------------------------------------------
# Here on we test some plots
PM25_mean_neus = pd.read_csv(PATH_AOD+'PM25_neus_mean_series.csv', parse_dates=[0])
PM25_mean_neus=PM25_mean_neus.set_index('date') # Converts the date column into an index for pretty and easy plots

# PM25_mean_neus['2004-06'].plot()  # Can change indexing for individual months.
ax=PM25_mean_neus['2004'].plot()     # Just declare the year you want.
ax.set_ylabel("PM$_{2.5}$ $\mu$g m$^{-3}$")
ax.set_xlabel("")
plt.show()


