# -*- coding: utf-8 -*-
"""
Simulating a simple dynamical system with lateral inhibition.

x -1  0  0
p  1 -1 -b
l  1  0 -e

"""




import numpy as np
N = 100
x = np.array(N)
p = np.array(N)
l = np.array(N)
dt  = 0.1


s = 0
b = 1
e = 1

for i in range(N-1):
    x[i+1] = x[i] + dt*(x[i] + s)
    p[i+1] = - p[i] + dt*(x[i] - b*l[i])
    l[i+1] = l[i] +dt*(-l[i] - e*l[i] + x[i]) 
    

    
