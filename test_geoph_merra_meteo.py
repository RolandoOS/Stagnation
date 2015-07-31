""" Test Geopotential Height Mapping """

import numpy as np
import matplotlib.pyplot as plt
import pylab
import cartopy
import cartopy.crs as ccrs
from   cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from   cartopy.feature       import NaturalEarthFeature, COASTLINE, LAKES
from   netCDF4               import Dataset

PATH = '/Users/rolivas1/Documents/DATA/MERRA_meteorology/'
file = PATH+'MERRA300.prod.assim.inst3_3d_asm_Cp.20040707.SUB.nc'
fn   = Dataset(file,mode='r')
lons = fn.variables['longitude'][:]
lats = fn.variables['latitude'][:]
hgt  = fn.variables['h'][:]
hgt  = np.squeeze(hgt)/10
fn.close()

lon,lat = np.meshgrid(lons, lats)

print "I've read the files..."

clon=-98.5795
clat= 39.8282

# Figure
print "... and now I'm plotting ..."

fig1=plt.figure()
ax = plt.axes(projection=cartopy.crs.Mercator())
cp = ax.pcolor(lon, lat, np.squeeze(hgt), transform=ccrs.PlateCarree(),
               cmap='PuBuGn')
cp.set_clim(vmin=550, vmax=600)
ax.set_extent([clon-35,clon+35, clat-15, clat+15])
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, alpha=0.5, linestyle=':')
gl.xlabels_top  = False
gl.ylabels_left = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
STATES = NaturalEarthFeature(category='cultural', scale='10m', facecolor='none',
                             name='admin_1_states_provinces_lakes')
ax.add_feature(STATES, linewidth=0.5)
plt.title('500 hPa Geopotential Height')
cbar=plt.colorbar(cp, orientation='horizontal')
cbar.set_label('dam')
plt.tight_layout()
fig1.savefig('/Users/rolivas1/Documents/ANALYSIS/Stagnation/test_fig1.png', bbox_inches=0, dpi = 300)
#plt.show()

# Figure
print "... and plotting some more ..."

fig2=plt.figure()
ax = plt.axes(projection=cartopy.crs.Mercator())
cp = ax.pcolor(lon, lat, np.squeeze(hgt), transform=ccrs.PlateCarree(),
               cmap='PuBuGn')
cp.set_clim(vmin=570, vmax=590)
ax.set_extent([-90,clon+35, 35, 48])
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, alpha=0.5, linestyle=':')
gl.xlabels_top  = False
gl.ylabels_left = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
STATES = NaturalEarthFeature(category='cultural', scale='10m', facecolor='none',
                             name='admin_1_states_provinces_lakes')
ax.add_feature(STATES, linewidth=0.5)
plt.title('500 hPa Geopotential Height')
cbar=plt.colorbar(cp, orientation='horizontal')
cbar.set_label('dam')
plt.tight_layout()
fig2.savefig('/Users/rolivas1/Documents/ANALYSIS/Stagnation/test_fig1_zoom_neus.png', bbox_inches=0, dpi = 300)
#plt.show()
print "... and done."
