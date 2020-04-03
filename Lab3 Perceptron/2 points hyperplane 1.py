#!/usr/bin/env python
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.animation as animation


xmin = -6
xmax = 6
ymin = -6
ymax = 6

x1 = np.array([4, 4]) #from class 1
x2 = np.array([-4, -4]) # from class -1


w = (x1 - x2)/2.0
print(w)


delta = 0.01 #accuracy of the grid for vizualization only
x = np.arange(xmin, xmax, delta)
y = np.arange(ymin, ymax, delta)
X, Y = np.meshgrid(x, y)

fig = plt.figure()
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.plot(x1[0],x1[1],'o')
plt.plot(x2[0],x2[1],'o')
plt.plot(0,0,'o') #origin
    
#vizualization
Z = X*w[0] + Y*w[1]
Z[Z>0] = 1
Z[Z<=0] = -1
im = plt.imshow(Z, interpolation='bilinear', cmap=cm.hot,
            origin='lower', extent=[xmin, xmax, ymin, ymax],
            vmax=abs(Z).max(), vmin=-abs(Z).max())

plt.show()
print('end')

