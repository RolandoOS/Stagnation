""" Stagnation MERRA Timeseries """

# REGIONAL

# Part 2.0: Random forest. Preparation of data frame.

# In this analysis we'll attempt to gather the most relevant meteorological variables (eulerean) that affect the concentrations of PM2.5 in the NEUS during the summer. We speculate that random forest analysis we'll give us insights into this problem. We'll sample MERRA gridpoints in the NEUS and make a point-by-point data frame.

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


# ----------------------------------------------------------------------
# Locate gridpoint inside US

fn   = Dataset(PATH_AOD+'GOCART_output_20020701_daily_average.nc4',mode='r')
lona = fn.variables['Longitude'][:]
lata = fn.variables['Latitude' ][:]
lata = lata.T
lona = lona.T
fn.close()

mask = (lona>-125) & (lona<-65) & (lata>24) & (lata<50)
LN=lona[mask]
LT=lata[mask]

code = ('AL','AZ','AR','CA','CO','CT','DE','FL','GA','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY')
cord = sgeom.MultiPoint(list(zip(LN,LT)))
shpfilename = shpreader.natural_earth(category='cultural', resolution='110m', name='admin_1_states_provinces_lakes')
states = shpreader.Reader(shpfilename).records()

CORD=np.empty([LN.size,2])*0.
k=0
for state in states:
	name = state.attributes['postal']
	if name in code:
		print name
		for i in range(0,LN.size):
			if state.geometry.contains(cord[i]):
				CORD[k,:]=[cord[i].x, cord[i].y]
				k=k+1
CORD=zip((CORD[CORD[:,0]!=0,0]),(CORD[CORD[:,1]!=0,1]))
CORDS=pd.DataFrame(CORD)
CORDS=CORDS.drop_duplicates()

maskota = np.empty([361,576])*0
for i in range(0,CORDS.shape[0]):
		# print i
		aux=(lona==CORDS.ix[i,0]) & (lata==CORDS.ix[i,1])
		maskota=maskota+aux
maskcount=maskota
maskota=maskota>=1

#-----------------------------------------------------------------------
# Creation of Data Frame

DAT=pd.DataFrame([])

iii=1
for i in date:

	yr=i.year   # Get date info from index in loop.
	mn=i.month
	dy=i.day
	
	print yr, mn, dy, iii/982.*100

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
	lonp5=(lon+180)*np.pi/180
	lon5,lat5 = np.meshgrid(lonp5, latp5)
	fn.close(); del u, v, lon, lat, fn

	# Surface (2m) variables.
	fn  = Dataset(files_sfc[0],mode='r')
	lon = fn.variables['longitude'][:]
	lat = fn.variables['latitude' ][:]
	u   = fn.variables['u2m' ][:]
	v   = fn.variables['v2m' ][:]
	t   = fn.variables['t2m' ][:]
	q   = fn.variables['qv2m'][:]
	s   = fn.variables['slp' ][:]
	u2  = np.squeeze(u)
	v2  = np.squeeze(v)
	t2  = np.squeeze(t)
	q2  = np.squeeze(q)
	s2  = np.squeeze(s)
	u2  = u2[1:-1,1:-1]
	v2  = v2[1:-1,1:-1]
	t2  = t2[1:-1,1:-1]
	q2  = q2[1:-1,1:-1]
	s2  = s2[1:-1,1:-1]
	latp2=(90+lat[1:-1])*np.pi/180
	lonp2=(lon[1:-1]+180)*np.pi/180
	fn.close(); del u, v, t, q, s, lon, lat, fn
	
	# Precipitation flux.
	fn   = Dataset(files_pre[0],mode='r')
	lon  = fn.variables['longitude'][:]
	lat  = fn.variables['latitude' ][:]
	prec = np.squeeze(fn.variables['prectot'][:])
	prec = prec[1:-1,1:-1]
	lonr,latr = np.meshgrid(lon,lat) # same shape as 2m variables
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
	PM1  = PM1.T
	PM2  = PM2.T
	PM3  = PM3.T
	PM4  = PM4.T
	PM5  = PM5.T
	PM25 = PM25.T
	lata = lata.T
	lona = lona.T
	latai = (90+lata)*np.pi/180
	lonai = (lona+180)*np.pi/180
	fn.close(); del fn

	clon=-98.5795 # central lat/lon for imaging. Dead-center of US.
	clat= 39.8282

	# plt.imshow(PM25); plt.show()

	# ------------------------------------------------------------------
	# Interpolates 144x288 500 hPa and 361x540) meteorology into 361x576
	# MERRAero aerosol grid.
	
	def interpol(latitude,longitude,variable):
		''' This thing interpolates
		'''
		lut=RSBS(latitude,longitude,variable)
		interpolated_vairable=lut.ev(latai.ravel(),lonai.ravel()).reshape((361, 576))
		return interpolated_vairable
	
	u5i  = interpol(latp5,lonp5,u5)
	v5i  = interpol(latp5,lonp5,v5)
	hgti = interpol(latp5,lonp5,hgt)
	prei = interpol(latp2,lonp2,prec)
	u2i  = interpol(latp2,lonp2,u2)
	v2i  = interpol(latp2,lonp2,v2)
	t2i  = interpol(latp2,lonp2,t2)
	q2i  = interpol(latp2,lonp2,q2)
	s2i  = interpol(latp2,lonp2,s2)
	
	# ------------------------------------------------------------------
	
	PER=pd.DataFrame.from_items([('u_500hPa',u5i[maskota]),('v_500hPa',v5i[maskota]),('hgt_500hPa',hgti[maskota]),('precip',prei[maskota]),('u_2m',u2i[maskota]),('v_2m',v2i[maskota]),('temp_2m',t2i[maskota]),('spec_humidity',q2i[maskota]),('slp',s2i[maskota]),('dust',PM1[maskota]),('sea_salt',PM2[maskota]),('AmmSO4',PM3[maskota]),('OC',PM4[maskota]),('BC',PM5[maskota]),('PM25',PM25[maskota]),('lat',lata[maskota]),('lon',lona[maskota])])

	PER['date']=i
	
	DAT=pd.concat([DAT,PER],axis=0)

	del PER
	iii=iii+1

DAT=DAT.set_index(DAT['date'])
DAT.drop('date', inplace=True, axis=1)
DAT.to_csv(PATH_RES+'Stagnation_random_forest_matrix_usa.csv', index=True)

# --- Stagnation Field Plot --------------------------------------
geo_axes = plt.subplot(111, projection=cartopy.crs.Miller())
clon=-98.5795 # central lat/lon for imaging. Dead-center of US.
clat= 39.8282
geo_axes.set_extent([clon-35,clon+35, clat-15, clat+15]) #US
#geo_axes.set_extent([-84,-66, 36, 48]) # NEUS
geo_axes.scatter(lona[maskota], lata[maskota], 10, transform=ccrs.PlateCarree(), marker='o',facecolor='k', edgecolors='k')
COUNTRIES = NaturalEarthFeature(category='cultural', scale='10m', facecolor='none', name='admin_0_countries_lakes')
geo_axes.add_feature(COUNTRIES, linewidth=0.5)
shpfilename = shpreader.natural_earth(category='cultural', resolution='10m', name='admin_1_states_provinces_lakes')
states = shpreader.Reader(shpfilename).records()
code = ('MD','VA','DE','NJ','PA','WV','RI','VT','NH','NY','MA','ME','CT')
for state in states:
	name = state.attributes['postal']
	if name in code:
		geo_axes.add_geometries(state.geometry, ccrs.PlateCarree(), facecolor='none', edgecolor='blue', linewidth=1)
plt.show()
#-----------------------------------------------------------------