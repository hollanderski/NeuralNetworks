#!/usr/bin/env python
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#example data from two classes, normal distributions
num = 100
x2 = np.random.multivariate_normal([-1,-1], np.array([[1,0],[0,1]]),num)
x1 = np.random.multivariate_normal([2,2], np.array([[1,0],[0,1]]),num)
xall = np.vstack((x1,x2))
labels = np.ones(num*2)
labels[num:] = -1.0

xmin = -6
xmax = 6
ymin = -6
ymax = 6

#weights initialization - random, small values, both positive and negative
#bias signal is connected with w[0]
w = 2*np.random.rand(3) - 1
#w = np.zeros(3) #try it: not recommended in general
print ('w=',w)


delta = 0.01 #grid accuracy for visualization
x = np.arange(xmin, xmax, delta)
y = np.arange(ymin, ymax, delta)
X, Y = np.meshgrid(x, y)
Z = X*w[1] + Y*w[2] + w[0]

fig = plt.figure()

plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.plot(x1[:,0],x1[:,1],'o')
plt.plot(x2[:,0],x2[:,1],'o')

print (xall.shape)
print (w[1:].shape)
print (w.shape)
#one iteration of learning algorithm
def update(i):
    global w
    print()
    print ('iteration=',i)

    #neuron resposnses for examples from both classes
    #we assume they should be 1 or -1 depending on te class
    ans = np.dot(xall, w[1:]) + w[0]  
    
    #mean squared error
    mse = np.sqrt(((ans - labels) * (ans - labels)).sum())
    print ('mse=',mse)

    errors = (ans[0:num] <= 0).sum() + (ans[num:] > 0).sum()
    print ('errors=',errors)
    
    #learning rate
    eta = 0.001
    #online learning
    #weights modification
    for k in range(num*2):
        w[1] += eta*(labels[k] - ans[k])*xall[k,0]
        w[2] += eta*(labels[k] - ans[k])*xall[k,1]
        w[0] += eta*(labels[k] - ans[k])*1.0  #bias weight modification        
        #weights vector normalization
        #try it, it seems to be a bad idea here
#        w = w/np.linalg.norm(w) 
    print ('w=',w)
    #vizualization
    plt.clf()
    plt.plot(x1[:,0],x1[:,1],'o')
    plt.plot(x2[:,0],x2[:,1],'o')
    Z = X*w[1] + Y*w[2] + w[0]
    Z[Z>0] = 1
    Z[Z<=0] = -1
    im = plt.imshow(Z, interpolation='bilinear', cmap=cm.hot,
                origin='lower', extent=[xmin, xmax, ymin, ymax],
                vmax=abs(Z).max(), vmin=-abs(Z).max())

    
#start of the algoirthm
ani = animation.FuncAnimation(fig, update, interval=1000, blit=False)
plt.show()

print ('end')