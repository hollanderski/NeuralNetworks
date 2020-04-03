#!/usr/bin/env python
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#example data from two classes, 2D normal distribution
num = 100
x2 = np.random.multivariate_normal([-2,-2], np.array([[1,0],[0,1]]),num)
x1 = np.random.multivariate_normal([2,2], np.array([[1,0],[0,1]]),num)
xmin = -6
xmax = 6
ymin = -6
ymax = 6

#initialization of weights - random, smal values, positive and negative values
#w[0] is the bias weight
w = 2*np.random.rand(3) - 1
#w = np.zeros(3) #try it: in general, initialization of weigths to zero is not a good idea
print(w)


delta = 0.01 #accuracy of the grid for vizualization only
x = np.arange(xmin, xmax, delta)
y = np.arange(ymin, ymax, delta)
X, Y = np.meshgrid(x, y)
Z = X*w[1] + Y*w[2] + w[0]

fig = plt.figure()

plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.plot(x1[:,0],x1[:,1],'o')
plt.plot(x2[:,0],x2[:,1],'o')

#this function is one iteration of perceptron learning
def update(i):
    global w
    print()
    print('iteration=',i)

    #perceptron responses to examplse from class 1, 
    #we assume they should be  > 0
    ans1 = np.dot(x1, w[1:]) + w[0]  
    errors1 = (ans1<=0).sum() # number of missclassifications from class 1
    print('errors1=',errors1)
    M1 = x1[ans1<=0] #selected examples that are missclassified from class 1
    criterion1 = ans1[ans1<=0].sum() #perceptron criterion - part 1 from the class 1
    print('criterion1=',criterion1)

    #perceptron responses to examplse from class 2, 
    #we assume they should be  <= 0    
    ans2 = np.dot(x2, w[1:]) + w[0]
    errors2 = (ans2>0).sum() # number of missclassifications from class 2
    print('errors2=',errors2)
    M2 = x2[ans2>0] #selected examples that are missclassified from class 2
    criterion2 = ans2[ans2>0].sum()#perceptron criterion - part 2 from the class 1
    print('criterion2=',criterion2)
    
    #full perceptron criterion -we should minimize it 
    criterion = np.abs(criterion1) + np.abs(criterion2) 
    print('criterion=',criterion)

    M1 = M1.sum(axis=0) #summed errors from class 1
    M2 = M2.sum(axis=0) #summed errors from class 2
    M = M1 - M2 #vector giving the direction of the change of the weights vector
    
    print('M=',M)
    eta = 0.005 #learning rate
    
    #weights modification
    if np.abs( M.sum() ) > 0: #or: if criterion > 0:
        w[1] += eta*M[0]
        w[2] += eta*M[1]
        w[0] += eta*(errors1-errors2) #bias weight modification 
        #a trick : normalization of the weight vector
        #connected with the learning rate - both influence the convergence of the training process        
        w = w/np.linalg.norm(w) 
    else:
        print('learning done')
    print('w=',w)
    
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

    
#start learning
ani = animation.FuncAnimation(fig, update, interval=1000, blit=False)
plt.show()

print('end')