# Stagnation Events

import numpy as np
import matplotlib.pyplot as plt
import glob
from collections import OrderedDict
import pylab
import iris
import iris.plot as iplt
import iris.quickplot as qplt

PATH='/Users/Rola/Documents/Science/JHU/Wildfires/NCEP_meteorology'

fnameh  = [PATH+'/hgt.2002.nc']
hgt     = iris.load_cube(fnameh)
fnamev  = [PATH+'/vwnd.2002.nc']
vwnd    = iris.load_cube(fnamev)
fnameu  = [PATH+'/uwnd.2002.nc']
uwnd    = iris.load_cube(fnameu)
fnamevs = [PATH+'/vwnd.sig995.2002.nc']
vwnds   = iris.load_cube(fnamevs)
fnameus = [PATH+'/uwnd.sig995.2002.nc']
uwnds   = iris.load_cube(fnameus)



for i in range(40,45):
      print i
      qplt.pcolormesh(hgt[i,5,:,:])
      plt.gca().coastlines()
      plt.clim(4600,6000)
      F = pylab.gcf()
      name=str(i)
      F.savefig(name, dpi = (500))
      plt.clf()