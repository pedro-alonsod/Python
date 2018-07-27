import numpy as np
a = np.zeros((3, 2), dtype=np.float32) # create a 2x3 matrix filled with 0.0
print(a)
b = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32) # create an integer matrix from python lists
print(b)
m = np.random.rand(2, 3) # create a matrix filled with random numbers in [0, 1]
print(m) # display a matrix (abbreviated for large matrices) print(m.tolist()) # convert to list to print the whole thing print(m.shape) # display the number rows and columns the matrix
m[1,2] = 9 # access elements
m[0] = np.zeros((3,)) # set row at a time
print(m[:2, 1:3]) # use slices as indices to get a submatrix m = a + b # matrix addition
m = a.transpose() * b # elementwise multiplication
c = np.dot(a, b) # matrix multiplication

import numpy.distutils.system_info as sysinfo
    
sysinfo.get_info('openblas')