'''
Created on Mar 16, 2015

@author: yuyin
'''
import numpy as np
import time
A = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
A = A.reshape(3,2,2)
print A
print np.mean(A[0,:,:])
print ("done")