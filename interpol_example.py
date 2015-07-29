import numpy as np
import matplotlib.pyplot as plt
from   scipy.interpolate import RectSphereBivariateSpline

lats = np.linspace(10, 170, 9) * np.pi / 180.
lons = np.linspace(0, 350, 18) * np.pi / 180.
data = np.dot(np.atleast_2d(90. - np.linspace(-80., 80., 18)).T,
np.atleast_2d(180. - np.abs(np.linspace(0., 350., 9)))).T

new_lats = np.linspace(1, 180, 180) * np.pi / 180
new_lons = np.linspace(1, 360, 360) * np.pi / 180
new_lats, new_lons = np.meshgrid(new_lats, new_lons)

lut = RectSphereBivariateSpline(lats, lons, data)

data_interp = lut.ev(new_lats.ravel(), new_lons.ravel()).reshape((360, 180)) #.T

#-----------------------------------------------------------------------

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.imshow(data, interpolation='nearest')
ax2 = fig.add_subplot(212)
ax2.imshow(data_interp, interpolation='nearest')
plt.show()

#fig2 = plt.figure()
#s = [3e9, 2e9, 1e9, 1e8]
#for ii in xrange(len(s)):
#lut = RectSphereBivariateSpline(lats, lons, data, s=s[ii])
#data_interp = lut.ev(new_lats.ravel(),
#new_lons.ravel()).reshape((360, 180)).T
#ax = fig2.add_subplot(2, 2, ii+1)
#ax.imshow(data_interp, interpolation='nearest')
#ax.set_title("s = %g" % s[ii])
#plt.show()