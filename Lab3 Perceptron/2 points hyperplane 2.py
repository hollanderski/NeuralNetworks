#!/usr/bin/env python
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


xmin = -6
xmax = 6
ymin = -6
ymax = 6

x1 = np.array([-2, -1]) #from class 1
x2 = np.array([-4, -4]) # from class -1

v = x1 - x2
mid = (x1 + x2)/2.0

w = v
w0 = -v[0]*mid[0] - v[1]*mid[1]

print(w, w0)

print('for x1: ', np.dot(w,x1) + w0)
print('for x2: ', np.dot(w,x2) + w0)

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
Z = X*w[0] + Y*w[1] + w0
Z[Z>0] = 1
Z[Z<=0] = -1
im = plt.imshow(Z, interpolation='bilinear', cmap=cm.hot,
            origin='lower', extent=[xmin, xmax, ymin, ymax],
            vmax=abs(Z).max(), vmin=-abs(Z).max())
 

plt.show()

print ('end')