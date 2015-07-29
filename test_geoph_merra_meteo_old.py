# Stagnation Events

#import iris
#import iris.plot as iplt
#import iris.quickplot as qplt
#from iris.coords import DimCoord
#from iris.cube   import Cube
#from collections import OrderedDict
#from scipy.io import netcdf

import numpy as np
import scipy
import matplotlib.pyplot as plt
import glob
import pylab
from   netCDF4 import Dataset
from   mpl_toolkits.basemap import Basemap

PATH = '/Users/rolivas1/Documents/DATA/MERRA_meteorology/'
file = PATH+'MERRA300.prod.assim.inst3_3d_asm_Cp.20040707.SUB.nc'
fn   = Dataset(file,mode='r')
# print H
lons = fn.variables['longitude'][:]
lats = fn.variables['latitude'][:]
hgt  = fn.variables['h'][:]
fn.close()

lon,lat = np.meshgrid(lons, lats)

map = Basemap(width=6000000,height=4000000,resolution='l',projection='stere',lat_ts=20,lat_0=39.8282,lon_0=-98.5795)
xi,yi = map(lon, lat)
cs = map.pcolormesh(xi,yi,np.squeeze(hgt)/10,shading='flat',cmap=plt.cm.jet)
cs.set_clim(vmin=550, vmax=600)
map.drawcoastlines()
map.drawcountries()
map.drawstates()
map.drawmeridians(np.arange(-180., 181., 10.), labels=[0,0,0,1], fontsize=10)
map.drawparallels(np.arange(-80., 81., 10.)  , labels=[1,0,0,0], fontsize=10)
cbar = map.colorbar(cs, location='bottom', pad="10%")
cbar.set_label('dam')
plt.title('500 hPa Geopotential Height')
F = pylab.gcf()
F.savefig("test5_20020707.png", dpi = (500))
plt.show()

del map, cs, cbar

map = Basemap(projection='ortho',lon_0=-98.5795,lat_0=39.8282,resolution='l')
cs = map.pcolor(lon,lat,np.squeeze(hgt)/10,shading='flat',cmap=plt.cm.jet,latlon=True)
cs.set_clim(vmin=550, vmax=600)
map.drawcoastlines()
map.drawcountries()
map.drawmeridians(np.arange(-180., 181., 10.))
map.drawparallels(np.arange(-80., 81., 10.)  )
cbar = map.colorbar(cs, location='bottom', pad="10%")
cbar.set_label('dam')
plt.title('500 hPa Geopotential Height')
F = pylab.gcf()
F.savefig("test6_20020707.png", dpi = (500))
plt.show()

