""" Stagnation MERRA Meteorology """

# Load packages, tools, and stuff:

import numpy as np
import matplotlib
matplotlib.use('Agg') # uncomment for faster rendering.
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import datetime
import time
import pandas as pd
#import pylab
import scipy
import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from   cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from   cartopy.feature       import NaturalEarthFeature, COASTLINE, LAKES
from   scipy.interpolate     import RectSphereBivariateSpline as RSBS
from   netCDF4               import Dataset
from   timeit                import default_timer as timer

# Directory paths to data and analysis directories:

PATH_AOD = '/Users/Rola/Documents/Science/JHU/DATA/MERRAero/'
PATH_MET = '/Users/Rola/Documents/Science/JHU/DATA/MERRA_meteorology/'
PATH_RES = '/Users/Rola/Documents/Science/JHU/ANALYSIS/Stagnation/'

# Load PM2.5 mean file with PANDAS:

PM25_mean_neus = pd.read_csv(PATH_AOD+'PM25_neus_mean_series.csv', parse_dates=[0])

# Note on datetime function:
# datetime.date.toordinal(datetime.date(2000, 1, 1)): 730120

# Converts the date into an index for pretty and easy plots with PANDAS.
PM25_mean_neus=PM25_mean_neus.set_index('date')

# --- Select year for processing ---------------------------------------

PM=PM25_mean_neus['2007']

# ----------------------------------------------------------------------

# By converting the index values to datetime format we can use the
# conunter in the loop to get date information without formatting:
date = pd.DatetimeIndex(PM.index.values)

for i in date:

	start = timer()

	yr=i.year   # Get date info from index in loop.
	mn=i.month
	dy=i.day

	# Generate meteorology file names to be read.
	hgtfname='MERRA*3d*%4.0f%02.0f%02.0f*.nc'       %(yr,mn,dy)
	sfcfname='MERRA*2d_slv*%4.0f%02.0f%02.0f*.nc'   %(yr,mn,dy)
	prefname='MERRA*2d_lnd*%4.0f%02.0f%02.0f*.nc'   %(yr,mn,dy)
	aodfname='GOCART*%4.0f%02.0f%02.0f*average.nc4' %(yr,mn,dy)

	# UNIX-like ls of file names to be read. Finds file in directory.
	files_hgt=glob.glob(PATH_MET+hgtfname)
	files_sfc=glob.glob(PATH_MET+sfcfname)
	files_pre=glob.glob(PATH_MET+prefname)
	files_aod=glob.glob(PATH_AOD+aodfname)

	# 500 hPa variables.
	fn   = Dataset(files_hgt[0],mode='r')
	lon  = fn.variables['longitude'][:]
	lat  = fn.variables['latitude'][:]
	hgt  = fn.variables['h'][:]
	hgt  = np.squeeze(hgt)/10
	u    = fn.variables['u'][:]
	u5   = np.squeeze(u)
	v    = fn.variables['v'][:]
	v5   = np.squeeze(u)
	latp5=(90+lat)*np.pi/180
	# lon[lon<0]=lon[lon<0]+360  #converts to all-positive longitudes.
	lon=lon+180 # This takes the IDL as 0 and makes all lon's positive
			# for the interpolation scheme below.
	lonp5=lon*np.pi/180
	lon5,lat5 = np.meshgrid(lonp5, latp5)
	fn.close(); del u, v, lon, lat, fn

	# Surface (2m) variables.
	fn   = Dataset(files_sfc[0],mode='r')
	lon  = fn.variables['longitude'][:]
	lat  = fn.variables['latitude' ][:]
	u    = fn.variables['u2m'][:]
	u2   = np.squeeze(u)
	v    = fn.variables['v2m'][:]
	v2   = np.squeeze(v)
	loni,lati = np.meshgrid(lon,lat)
	latp2=(90+lat)*np.pi/180
	# lon[lon<0]=lon[lon<0]+360  #converts to all-positive longitudes.
	lon=lon+180
	lonp2=lon*np.pi/180
	lon2,lat2 = np.meshgrid(lonp2, latp2)
	fn.close(); del u, v, lon, lat, fn
	
	# Precipitation flux.
	fn   = Dataset(files_pre[0],mode='r')
	lon  = fn.variables['longitude'][:]
	lat  = fn.variables['latitude' ][:]
	prec = np.squeeze(fn.variables['prectot'][:])
	lonr,latr = np.meshgrid(lon,lat)
	fn.close(); del lon, lat, fn

	# MERRAero PM2.5 and AOD.
	fn   = Dataset(files_aod[0],mode='r')
	lona = fn.variables['Longitude'    ][:]
	lata = fn.variables['Latitude'     ][:]
	AOD  = fn.variables['TOTEXTTAU_avg'][:]
	# 1e9 converts to micrograms per cubic meter.
	# 1.375 factor converts SO4 to AmmSO4.
	PM1  = fn.variables['DUSMASS25_avg'][:]*1e9
	PM2  = fn.variables['SSSMASS25_avg'][:]*1e9
	PM3  = fn.variables['SO4SMASS_avg' ][:]*1e9*1.375
	PM4  = fn.variables['OCSMASS_avg'  ][:]*1e9
	PM5  = fn.variables['BCSMASS_avg'  ][:]*1e9
	PM25 = PM1+PM2+PM3+PM4+PM5   # By Definition from MERRAero
	fn.close(); del fn

	clon=-98.5795 # central lat/lon for imaging. Dead-center of US.
	clat= 39.8282

	# ----------------------------------------------------------------
	# Interpolates coarse (144x288) 500 hPa data into finer (361x540)
	# surface grid.

	lut=RSBS(latp5,lonp5,u5)
	u5i=lut.ev(lat2.ravel(),lon2.ravel()).reshape((361, 540))
	del lut
	lut=RSBS(latp5,lonp5,v5)
	v5i=lut.ev(lat2.ravel(),lon2.ravel()).reshape((361, 540))
	del lut
	lut=RSBS(latp5,lonp5,hgt)
	hgti=lut.ev(lat2.ravel(),lon2.ravel()).reshape((361, 540));

	del hgt, u5, v5, lonp5, latp5, lonp2, latp2, lon2, lat2, lon5, lat5

	#fig = plt.figure()
	#ax1 = fig.add_subplot(211)
	#ax1.imshow(hgt, interpolation='nearest')
	#ax2 = fig.add_subplot(212)
	#ax2.imshow(hgti, interpolation='nearest')
	#plt.show()

	# ----------------------------------------------------------------
	# Splits the data fields into Northern and Western Hemispheres!
	# (Note: The graphing time is reduced from ~220sec to ~30sec)

	#aux=lata[0]
	#aux=aux[aux>0]
	#aux.shape
	#
	#aux=lona[:,0]
	#aux=aux[aux<0]
	#aux.shape
	
	mski=(lati>0) & (loni<0) # Masks for logical or "fancy" indexing.
	mska=(lata>0) & (lona<0)

	u5t  =  u5i[mski].reshape(180,270)
	v5t  =  v5i[mski].reshape(180,270)
	u2t  =   u2[mski].reshape(180,270)
	v2t  =   v2[mski].reshape(180,270)
	hgt  = hgti[mski].reshape(180,270)
	pre  = prec[mski].reshape(180,270)
	lon  = loni[mski].reshape(180,270)
	lat  = lati[mski].reshape(180,270)
	latn = lata[mska].reshape(289,180)
	lonn = lona[mska].reshape(289,180)
	AODn =  AOD[mska].reshape(289,180)
	PM25n= PM25[mska].reshape(289,180)
	# ----------------------------------------------------------------
	# Stagnation Analysis

	w2=np.power(np.power(u2t,2)+np.power(v2t,2),0.5)
	w5=np.power(np.power(u5t,2)+np.power(v5t,2),0.5)
	wind=3.2*np.power(2./10,1./7) # Wind profile power law
	
	# Note on "wind": The stagnation index described in Horton et al.
	# requires a wind threshold of 3.2 m/s at 10m but the data from
	# MERRA is given at 2m.
	
	# Stagnation index from Horton et al., 2014:
	S =np.zeros((180,270))
	mask = (w2<wind) & (w5<13) & (pre<(1./24/3600))
	S[mask]=1
	
	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# ax.imshow(S)
	# plt.show()
	
	# ----------------------------------------------------------------
	# Figure
	end = timer()
	print "I've read and processed the files in", round(end - start), "sec and now I'm plotting ..."
	start = timer()

	fig1=plt.figure()
	#gs = gridspec.GridSpec(2,2,height_ratios=[3,1], width_ratios=[3,1])
	gs = gridspec.GridSpec(3,4)
	# --- Stagnation Field Plot ------
	geo_axes = plt.subplot(gs[:2,:2], projection=cartopy.crs.Miller())
	geo_axes.set_extent([-90,clon+35, 35, 48])
	cs = geo_axes.contour(lon, lat, hgt, 20, transform=ccrs.Miller(), linewidths=1.5, colors='darkgreen', linestyles='-')
	plt.clabel(cs, fontsize=11, fmt='%1.0f')
	cmap = plt.get_cmap('PuBu', 2)
	cp   = geo_axes.pcolor(lon, lat, S, transform=ccrs.Miller(), cmap=cmap)
	cbar = plt.colorbar(cp, orientation='horizontal', pad=0.01, spacing='uniform')
	geo_axes.set_title('Stagnation', position=(0.5, -0.3), fontsize=12)
	#cbar.ax.set_xticklabels(['Non-stagnant','Stagnant'])
	# STATES = NaturalEarthFeature(category='cultural', scale='10m', facecolor='none', name='admin_1_states_provinces_lakes')
	# geo_axes.add_feature(STATES, linewidth=0.5)
	COUNTRIES = NaturalEarthFeature(category='cultural', scale='10m', facecolor='none', name='admin_0_countries_lakes')
	geo_axes.add_feature(COUNTRIES, linewidth=0.5)
	shpfilename = shpreader.natural_earth(category='cultural', resolution='10m', name='admin_1_states_provinces_lakes')
	reader = shpreader.Reader(shpfilename)
	states = reader.records()
	code = ('MD','VA','DE','NJ','PA','WV','RI','VT','NH','NY','MA','ME','CT')
	for state in states:
		name = state.attributes['postal']
		if name in code:
			geo_axes.add_geometries(state.geometry, ccrs.PlateCarree(), facecolor='none', edgecolor='blue', linewidth=1)

	# --- PM Field Plot -------
	geo_axes = plt.subplot(gs[:2,2:], projection=cartopy.crs.Miller())
	geo_axes.set_extent([-90,clon+35, 35, 48])
	cs = geo_axes.contour(lon, lat, hgt, 20, transform=ccrs.Miller(), linewidths=1.5, colors='darkgreen', linestyles='-')
	plt.clabel(cs, fontsize=11, fmt='%1.0f')
	cp = geo_axes.pcolor(lonn, latn, PM25n, transform=ccrs.Miller(), cmap='Greys')
	amax=40
	amin=5
	cp.set_clim(vmin=amin, vmax=amax)
	cbar = plt.colorbar(cp, ticks=[amin, amax], orientation='horizontal', pad=0.01)
	geo_axes.set_title(r'$\mathregular{PM_{2.5} \/ (\mu g \/ m^{-3})}$', position=(0.5, -0.3), fontsize=12)
	# STATES = NaturalEarthFeature(category='cultural', scale='10m', facecolor='none', name='admin_1_states_provinces_lakes')
	# geo_axes.add_feature(STATES, linewidth=0.5)
	COUNTRIES = NaturalEarthFeature(category='cultural', scale='10m', facecolor='none', name='admin_0_countries_lakes')
	geo_axes.add_feature(COUNTRIES, linewidth=0.5)
	shpfilename = shpreader.natural_earth(category='cultural', resolution='10m', name='admin_1_states_provinces_lakes')
	reader = shpreader.Reader(shpfilename)
	states = reader.records()
	code = ('MD','VA','DE','NJ','PA','WV','RI','VT','NH','NY','MA','ME','CT')
	for state in states:
		name = state.attributes['postal']
		if name in code:
			geo_axes.add_geometries(state.geometry, ccrs.PlateCarree(), facecolor='none', edgecolor='blue', linewidth=1)
	
	# --- PM Time Series Plot -------
	axes=plt.subplot(gs[2,:])
	PM.plot(ax=axes, ylim=(0,50), linewidth=2)
	ymin, ymax = axes.get_ylim()
	# ------
	# Draw vertical line markers for individual events.
	#axes.vlines(x=date[13], ymin=ymin, ymax=ymax, color='r', linewidth=2)
	#axes.vlines(x=date[ 4], ymin=ymin, ymax=ymax, color='r', linewidth=2)
	# ------
	axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=12)
	axes.set_ylabel(r'$\mathregular{PM_{2.5} \/(\mu g \/ m^{-3})}$', fontsize=12)
	axes.set_xlabel('')
	axes.vlines(x=i, ymin=ymin, ymax=ymax, color='k', linewidth=2)
	plt.tight_layout()
	figname='aod_stag_%4.0f%02.0f%02.0f.png' %(yr,mn,dy)
	fig1.savefig(PATH_RES+figname, bbox_inches='tight', dpi = 200)

	plt.show()
	plt.close("all")
	del fig1, axes, geo_axes
	end = timer()
	print "I've created and saved the figure in", round(end - start), "sec."


## --- Test Figure for Logical Indexing --------------------------------
#fig2=plt.figure()
#ax = plt.axes(projection=cartopy.crs.Mercator())
##cp = ax.pcolor(loni, lati, hgti, transform=ccrs.PlateCarree(), cmap='PuBuGn')
#cp = ax.pcolor(lon , lat , hgt , transform=ccrs.PlateCarree(), cmap='PuBuGn')
#cp.set_clim(vmin=465, vmax=585)
#ax.set_extent([-179.5,180,-80,80]) # globe
##ax.set_extent([-90,clon+35, 35, 48]) # NEUS
##ax.set_extent([clon-35,clon+35, clat-15, clat+15]) # US
#gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, alpha=0.5, linestyle=':')
#gl.xlabels_top  = False
#gl.ylabels_left = False
#gl.xformatter = LONGITUDE_FORMATTER
#gl.yformatter = LATITUDE_FORMATTER
##STATES = NaturalEarthFeature(category='cultural', scale='10m', facecolor='none', name='admin_1_states_provinces_lakes')
##ax.add_feature(STATES, linewidth=0.5)
#ax.add_feature(COASTLINE, linewidth=0.5)
#plt.title('Stagnation + AOD Analysis')
#cbar=plt.colorbar(cp, orientation='horizontal')
#cbar.set_label('500 hPa height (m)')
#plt.tight_layout()
##fig2.savefig(PATH_RES+'cacaus5a.png', bbox_inches=0, dpi = 300)
#fig2.savefig(PATH_RES+'cacaus5b.png', bbox_inches=0, dpi = 300)
##plt.show()
## ---------------------------------------------------------------------