""" Stagnation MERRA Timeseries """

# Part 1: analysis of data.

# --- Import tools and data --------------------------------------------

import pandas as pd
import numpy  as np
import statsmodels.api   as sm
import matplotlib.pyplot as plt

# Directory paths:

PATH_AOD = '/Users/Rola/Documents/Science/JHU/DATA/MERRAero/'
PATH_MET = '/Users/Rola/Documents/Science/JHU/DATA/MERRA_meteorology/'
PATH_RES = '/Users/Rola/Documents/Science/JHU/ANALYSIS/Stagnation/'

# Load data:

PM  = pd.read_csv(PATH_AOD+'PM25_neus_mean_series.csv', parse_dates=[0])
PM  = PM.set_index('date')
DAT = pd.read_csv(PATH_RES+'Stagnation_percentage.csv', index_col=0, parse_dates=True)
PMm = pd.read_csv(PATH_MET+'MERRA_PM_ass_unass.csv', index_col=0, parse_dates=True)


date = pd.DatetimeIndex(PM.index.values)

# --- Scatter PM vs % of stagnation with lags --------------------------

for i in range(-11,1):
	x=PM['PM25_MERRAero']
	y=DAT.shift(i)
	ax=plt.subplot(111)
	ax.scatter(x,y, c=u'k', marker=u'o')
	#ax.axis([-5,60,-5,120])
	ax.set_xlabel(r'$\mathregular{NEUS-mean \ MERRA \ PM_{2.5} \/(\mu g \/ m^{-3})}$', fontsize=14)
	ax.set_ylabel(r'$\mathregular{Stagnation \ \%}$', fontsize=14)
	file_name='StagPerc_vs_PM_lag%s.png' %i
	plt.savefig(PATH_RES+file_name, bbox_inches='tight', dpi = 200)
	plt.show()

# --- Time series PM and % of stagnation -------------------------------

for i in range(2002,2013):
	i=np.str(i)
	print i
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	PM['PM25_MERRAero'][i].plot(ax=ax1, style='-b')
	DAT[i].plot(ax=ax2, style='-g', secondary_y=True, legend=False)
	ax1.set_ylabel('NEUS PM2.5', color='b')
	ax2.right_ax.yaxis.set_label_text('% Stagnation', color='g')
	file_name='StagPerc_PM_series_%s.png' %i
	#plt.savefig(PATH_RES+file_name, bbox_inches='tight', dpi = 200)
	plt.show()

# --- Time series MERRA MODIS assimilated and % of stagnation ----------

for i in range(2011,2012):
	i=np.str(i)
	print i
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	PMm['MERRA assimilation'][i].plot(ax=ax1, style='-b')
	DAT[i].plot(ax=ax2, style='-g', secondary_y=True, legend=False)
	ax1.set_ylabel('MERRA PM2.5 assimilated', color='b')
	ax2.right_ax.yaxis.set_label_text('% Stagnation', color='g')
	file_name='StagPerc_PM_series_%s.png' %i
	#plt.savefig(PATH_RES+file_name, bbox_inches='tight', dpi = 200)
	plt.show()

# --- Time series MERRA MODIS unassimilated and % of stagnation --------

for i in range(2011,2012):
	i=np.str(i)
	print i
	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	PMm['MERRA unassimilated'][i].plot(ax=ax1, style='-b')
	DAT[i].plot(ax=ax2, style='-g', secondary_y=True, legend=False)
	ax1.set_ylabel('MERRA PM2.5 unassimilated', color='b')
	ax2.right_ax.yaxis.set_label_text('% Stagnation', color='g')
	file_name='StagPerc_PM_series_%s.png' %i
	#plt.savefig(PATH_RES+file_name, bbox_inches='tight', dpi = 200)
	plt.show()

# --- Time series MERRA MODIS ass + unass + EPA PM2.5 -----------------

for i in range(2011,2013):
	i=np.str(i)
	print i
	fig, ax1 = plt.subplots()
	PM['PM25_MERRAero'][i].plot(ax=ax1, style='-k')
	PMm['MERRA unassimilated'][i].plot(ax=ax1, style='-b')
	PMm['MERRA assimilation' ][i].plot(ax=ax1, style='-r')
	ax1.set_ylim([0,35])
	ax1.set_ylabel(r'$\mathregular{NEUS-mean \ PM_{2.5} \/(\mu g \/ m^{-3})}$', fontsize=14)
	ax1.legend(['EPA PM2.5','MERRA MODIS unassimilated','MERRA MODIS assimilated'], loc=2, fontsize = 'small')
	file_name='PM25_series_EPA_MERRA_unassimilated_%s.png' %i
	#plt.savefig(PATH_RES+file_name, bbox_inches='tight', dpi = 300)
	plt.show()


# --- Logistic regression PM vs % stagnation ---------------------------

PM['Bin']=0.0
PM.ix[PM['PM25_MERRAero'] >= 20, 'Bin']=1.0

DAT['intercept']=1.0

logit=sm.Logit(PM['Bin'],DAT[['Stag_Perc','intercept']]).fit()
print logit.summary()

#                             Logit Regression Results
# ==============================================================================
# Dep. Variable:                    Bin   No. Observations:                  982
# Model:                          Logit   Df Residuals:                      980
# Method:                           MLE   Df Model:                            1
# Date:                Mon, 06 Jul 2015   Pseudo R-squ.:                0.009353
# Time:                        11:27:31   Log-Likelihood:                -400.23
# converged:                       True   LL-Null:                       -404.01
#                                         LLR p-value:                  0.005975
# ==============================================================================
#                  coef    std err          z      P>|z|      [95.0% Conf. Int.]
# ------------------------------------------------------------------------------
# Stag_Perc     -0.0106      0.004     -2.590      0.010        -0.019    -0.003
# intercept     -1.5962      0.111    -14.319      0.000        -1.815    -1.378
# ==============================================================================

Perc = np.arange(0,100)
Prob = np.exp(-1.5962-0.0106*Perc)/(1+np.exp(-1.5962-0.0106*Perc))*100
plt.scatter(Perc,Prob)
plt.ylabel('% Probability of High PM')
plt.xlabel('Stagnation (%)'); plt.show()

# --- Hypotesis testing ------------------------------------------------

# H0: Stagnation does not cause high PM2.5
# H1: Stagnation causes PM2.5

# Translates into:

# H0: Days that are stagnant have the same concentrations as days that are not stagnat.
# H1: Days that are stagnant have higher concentrations than days that are not stagnat.

ANA=pd.concat([PM['PM25_MERRAero'],DAT['Stag_Perc']], axis=1)

treshold=50

STG=ANA.ix[ANA['Stag_Perc']>=treshold,'PM25_MERRAero']
NST=ANA.ix[ANA['Stag_Perc']< treshold,'PM25_MERRAero']
ANF=pd.concat([STG,NST], axis=1)
ANF.columns=['Stagnant','Non-Stagnant']
ANF.boxplot(); plt.show()

# Divide High pressure system location in quadrants in the US.

