#%matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from time import time


num = 100
x0 = np.random.multivariate_normal([2,2], np.array([[1,0],[0,1]]),num)
x0 = np.hstack((np.repeat(1,num).reshape(num,1), x0)) #adding bias signal explicitly
d0 = np.repeat(0, num)
x1 = np.random.multivariate_normal([-2,-2], np.array([[1,0],[0,1]]),num)
x1 = np.hstack((np.repeat(1,num).reshape(num,1), x1)) #adding bias signal explicitly
d1 = np.repeat(1, num)

X = np.vstack((x0,x1))
d = np.hstack((d0,d1))


def perceptron(x, w):
    u = np.dot(x, w)
    y = 1 if u>0 else 0
    return y

epoch = 100
eta = 0.01

iteration = epoch*X.shape[0]

weights = np.random.random(3) #perceptoron weights
a = -weights[1]/weights[2]
b = -weights[0]/weights[2]

xx = np.linspace(-6 ,6 ,100)
yy = a*xx+b



fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.scatter(x0[:,1], x0[:,2], color='r')
ax.scatter(x1[:,1], x1[:,2], color='g')
ax.plot(xx,yy, lw=2, c='k')

def animate(i):
    flag=False
    e, p = divmod(i,X.shape[0])
    global weights
    y = perceptron(X[p,:], weights)
    if y==1 and d[p]==0:
        flag = True
        weights -= eta*X[p,:]
    if y==0 and d[p]==1:
        flag=True
        weights += eta*X[p,:]
    if flag:
        a = -weights[1]/weights[2]
        b = -weights[0]/weights[2]
        yy = a*xx+b
        ax.clear()
        ax.scatter(x0[:,1], x0[:,2], color='r')
        ax.scatter(x1[:,1], x1[:,2], color='g')
        ax.scatter(X[p,1], X[p,2], color='k', s=80)
        ax.plot(xx,yy, lw=2, c='k')
        ax.text(0.0, -5.0, 'epoch: {0}, sample: {1}'.format(e,p))
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
    return ax,

dt = 1./30
t0 = time()
animate(0)
t1 = time()
interval = 100 * dt - (t1 - t0)

anim = animation.FuncAnimation(fig, animate, frames=iteration, interval=interval)

plt.show()