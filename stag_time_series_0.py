""" Stagnation MERRA Timeseries """

# Part 0: Preparation of data.

# Load packages, tools, and stuff:
import glob
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import shapely.geometry  as sgeom
import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from   cartopy.mpl.gridliner  import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from   cartopy.feature        import NaturalEarthFeature, COASTLINE, LAKES
from   scipy.interpolate      import RectSphereBivariateSpline as RSBS
from   netCDF4                import Dataset

# Directory paths to data and analysis directories:

PATH_AOD = '/Users/Rola/Documents/Science/JHU/DATA/MERRAero/'
PATH_MET = '/Users/Rola/Documents/Science/JHU/DATA/MERRA_meteorology/'
PATH_RES = '/Users/Rola/Documents/Science/JHU/ANALYSIS/Stagnation/'

# Load PM2.5 mean file with PANDAS:

PM = pd.read_csv(PATH_AOD+'PM25_neus_mean_series.csv', parse_dates=[0])
PM = PM.set_index('date')

date = pd.DatetimeIndex(PM.index.values)

DAT=pd.DataFrame([])

for i in date:

	yr=i.year   # Get date info from index in loop.
	mn=i.month
	dy=i.day
	
	print yr, mn, dy

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
	v5   = np.squeeze(v)
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
	
	# Stagnation index from Horton et al., 2014: A grid cell day is considered stagnant when daily-mean near-surface (10-m) wind speeds are <3.2 m/s, daily-mean mid-tropospheric (500 mb) wind speeds are <13 m/s, and daily-mean precipitation accumulation is <1 mm.
	
	S =np.zeros((180,270))
	mask = (w2<wind) & (w5<13) & (pre<(1./24/3600))
	S[mask]=1
	
	LT  = lat[S==1]
	LN  = lon[S==1]
	
	# Percentage of stagnant cells

	code = ('MD','VA','DE','NJ','PA','WV','RI','VT','NH','NY','MA','ME','CT')
	cord = sgeom.MultiPoint(list(zip(LN,LT)))
	shpfilename = shpreader.natural_earth(category='cultural', resolution='110m', name='admin_1_states_provinces_lakes')
	states = shpreader.Reader(shpfilename).records()

	CORD=np.empty([LN.size,2])*0.
	k=0
	for state in states:
		name = state.attributes['postal']
		if name in code:
			#print name
			for i in range(0,LN.size):
				if state.geometry.contains(cord[i]):
					CORD[k,:]=[cord[i].x, cord[i].y]
					k=k+1
	CORD=zip((CORD[CORD[:,0]!=0,0]),(CORD[CORD[:,1]!=0,1]))
	CORDS=pd.DataFrame(CORD)

	# Total of stagnant cells
#
#	code = ('MD','VA','DE','NJ','PA','WV','RI','VT','NH','NY','MA','ME','CT')
#	cord = sgeom.MultiPoint(list(zip(lon.reshape(48600,1),lat.reshape(48600,1))))
#	shpfilename = shpreader.natural_earth(category='cultural', resolution='110m', name='admin_1_states_provinces_lakes')
#	states = shpreader.Reader(shpfilename).records()
#
#	CORD=np.empty([lon.size,2])*0.
#	k=0
#	for state in states:
#		name = state.attributes['postal']
#		if name in code:
#			print name
#			for i in range(0,lon.size):
#				if state.geometry.contains(cord[i]):
#					CORD[k,:]=[cord[i].x, cord[i].y]
#					k=k+1
#	CORD=zip((CORD[CORD[:,0]!=0,0]),(CORD[CORD[:,1]!=0,1]))
#	TOTAL=pd.DataFrame(CORD)
#	TOTAL.shape[0] = 204
#
#	# --- Stagnation Field Plot --------------------------------------
#	geo_axes = plt.subplot(111, projection=cartopy.crs.Miller())
#	#clon=-98.5795 # central lat/lon for imaging. Dead-center of US.
#	#clat= 39.8282
#	#geo_axes.set_extent([clon-35,clon+35, clat-15, clat+15]) #US
#	geo_axes.set_extent([-84,-66, 36, 48]) # NEUS
#	geo_axes.scatter(CORDS[0], CORDS[1], 10, transform=ccrs.PlateCarree(), marker='o',facecolor='k', edgecolors='k')
#	geo_axes.scatter(TOTAL[0], TOTAL[1], 50, transform=ccrs.PlateCarree(), marker='s',facecolor='none', edgecolors='k')
#	COUNTRIES = NaturalEarthFeature(category='cultural', scale='10m', facecolor='none', name='admin_0_countries_lakes')
#	geo_axes.add_feature(COUNTRIES, linewidth=0.5)
#	shpfilename = shpreader.natural_earth(category='cultural', resolution='10m', name='admin_1_states_provinces_lakes')
#	states = shpreader.Reader(shpfilename).records()
#	code = ('MD','VA','DE','NJ','PA','WV','RI','VT','NH','NY','MA','ME','CT')
#	for state in states:
#		name = state.attributes['postal']
#		if name in code:
#			geo_axes.add_geometries(state.geometry, ccrs.PlateCarree(), facecolor='none', edgecolor='blue', linewidth=1)
#	plt.show()
#	#-----------------------------------------------------------------

	PER=pd.DataFrame([CORDS.shape[0]/204.*100])
	
	DAT=pd.concat([DAT,PER],axis=0)

	del PER

DAT=DAT.set_index(date)
DAT.columns=['Stag_Perc']
DAT.to_csv(PATH_RES+'Stagnation_percentage.csv')