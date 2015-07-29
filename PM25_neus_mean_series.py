import pandas as pd
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
import datetime as datetime
import numpy as np

path='/Users/Rola/Desktop/'

PM=pd.read_csv(path+'PM25_go_aqs_nearest_complete.csv', index_col='date', parse_dates={'date': [0,1,2]}, date_parser=lambda x: pd.datetime.strptime(x, '%Y %m %d'))

date = pd.DatetimeIndex(PM.index.values)

PMM=PM.resample('D', how='mean')

PMM[['PM25_AQS','PM25_MERRAero']].to_csv(path+'PM25_neus_mean_series_new_method.csv')  # Saves the dataframe to a .csv file.

#  A brief note on the PM25_go_aqs_nearest_complete.csv file:
#
#  This file was created in a previous Matlab program. It contains PM2.5 data
#  from AQS stations in the NEUS with a matching PM2.5 value  from  MERRAero.
#  The  MERRAero  value was selected to be from the gridpoint nearest to each
#  AQS stations. Given that we only had MERRAero PM values for the summer, we
#  only show the PM values  from  AQS  for the  summer.  There is a .csv file
#  AQS_PM25_all_1999_2013.csv which contains the complete record of AQS PM2.5
#  for the neus for every day of the year.