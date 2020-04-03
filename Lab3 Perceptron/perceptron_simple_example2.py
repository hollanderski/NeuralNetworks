#!/usr/bin/env python
import numpy as np

#initial weights
w = np.array([0.3, -0.4])
w0 = 0.1 #bias weight

#input vector
x = np.array([-0.5, 0.9 ])
#class label
d = 1 

print('example x=',x,' is from class ',d)

#learning rate
eta = 0.1


y = np.dot(w,x) + w0
print('weighted sum, y=',y)

#activation function
if y > 0:
    u = 1
else:
    u = -1
    
print('percetron says: x is from class ',u)
    
while u!=d:
    w[0] = w[0] + eta*x[0]*d
    w[1] = w[1] + eta*x[0]*d
    w0 = w0 + eta*1*d
    print('\nweights after update, w=',w,' w0=',w0)
    y = np.dot(w,x) + w0
    print('weighted sum, y=',y)
    #activation function
    if y > 0:
        u = 1
    else:
        u = -1    
    print('percetron says: x is from class ',u)

print('learning done')