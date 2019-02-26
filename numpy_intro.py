# python knowledge
hw12 = '%s %s %d' % ('hello', 'world', 12) # print hello world 12
d = {'cat': 'cute', 'dog': 'furry'}
d['fish'] = 'wet'
del d['fish']
# Remove an element from a dictionary
print(d.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A"
'''
A tuple is an (immutable) ordered list of values. A tuple is in many ways similar to a list; 
one of the most important differences is that tuples can be used as keys in dictionaries and as elements of sets, 
while lists cannot.
'''

import numpy as np

'''
=======================
Defining arrays of data
=======================
-------------------------------------------------------------------------------
To define one-dimensional arrays (vectors), we can use numpy's "array" object 
and pass it a list of values: 
'''

a = np.array([1,2,3,4,5,6,7])

# or equivalently, using Python iterables:
a = np.array( range(1,8) )
print(a)

'''
-------------------------------------------------------------------------------
To make a 2D array (matrix), provide the constructor with a list of lists:
'''
A = np.array( [[1,2,3,4],[5,6,7,8],[9,10,11,12]] )
print(A)

'''
-------------------------------------------------------------------------------
For linear algebra-like operations will usually want "vectors" of size (m,1) 
or size (1,n) -- in other words, two-dimensional matrices:
'''
# A row vector can be created as:
b = np.array( [[1,2,3,4,5,6,7]])

# and a column vector as
bT = np.array( [[1],[2],[3],[4],[5],[6],[7]] )

# that's pretty inconvient, so we usually just use the "transpose" operator:
bT = np.array( [[1,2,3,4,5,6,7]]).T

'''
-------------------------------------------------------------------------------

One thing to notice is that  a and  b are not quite the same! a is a vector 
with shape (7,), while  b is actually a matrix with shape (1,7) -- it has two 
dimensions, one of which just happens to be one.
'''
print("a's shape: ", a.shape, "\t b's shape: ", b.shape)

'''
-------------------------------------------------------------------------------
You can use "atleast_2d" to force a vector to become a matrix, but be careful 
of what size is created:
'''
aNew = np.atleast_2d(a)
print ("shape after 2D: ", aNew.shape)

'''
-------------------------------------------------------------------------------
There are several useful constructors for matrices that automatically "fill" a 
matrix of some shape:

'''
A0 = np.zeros( (3,4) )    # 3x4 matrix of all zeros
A1 = np.ones( (4,4) )     # 3x4 matrix of all ones
Ru = np.random.rand(2,2)  # 2x2 matrix of uniform random numbers, in [0,1)
Rn = np.random.randn(3,2) # 3x2 matrix of standard Gaussian random numbers
B = np.tile(b, (3,2))     # "tiles" copies of b (3 copies by 2 copies)

'''
-------------------------------------------------------------------------------
A very useful constructor is linspace (and similarly logspace)
'''
# both inclusively, you an also set step 
# length-4 vector interpolating between 1.0 and 7.0
b = np.linspace(1.0,7.0,4) 

# length-7 vector interpolating between 10^0 and 10^2 logarithmically
c = np.logspace(0.0,2.0,7) 

# Create a constant array
d = np.full((2,2), 7)

# Create a 2x2 identity matrix
e = np.eye(2)


'''
=======================
Array indexing
=======================

Arrays can be indexed in many simple and useful ways.

-------------------------------------------------------------------------------
The (i,j)th entry in a matrix is simply:
'''
print( "A[2,0]=", A[2,0] )  # 3rd row, 1st column

'''
-------------------------------------------------------------------------------
To reference an entire row or column, use the slice operator ":"
'''
print( "A[0,:]=", A[0,:] )

print( "A[:,1]=", A[:,1] )

'''
-------------------------------------------------------------------------------
You can use more general slicing with ranges and steps:
'''
print( "A[1,0:2]=", A[1,0:2] )

print( "A[0,0:4:2]=", A[0,0:4:2] )

'''
-------------------------------------------------------------------------------
It is often useful to use lists to slice out particular rows or columns. 
You can do this with one row and several columns (or vice versa):
'''
print( "A[2, [1,4]]=", A[2, [0,3]] )

'''
-------------------------------------------------------------------------------
or all rows and selected columns (or vice versa)
'''
print( "A[:, [1,4]]=\n", A[:, [0,3]] )

'''
-------------------------------------------------------------------------------
integer array indexing
'''
a = np.array([[1,2], [3, 4], [5, 6]])
print(a[[0, 1, 2], [0, 1, 0]]) # Prints "[1 4 5]"

# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))# Prints "[1 4 5]"


'''
=======================
Data types
=======================
'''
x = np.array([1, 2], dtype=np.int64)
# Force a particular datatype
print(x.dtype) # Prints "int64"
x = np.array([1, 2], dtype=np.float64)

'''

=======================
Arithmetic operations
=======================

Arithmetic operations are defined for arrays, i.e., adds the scalar value 2 to 
every entry of a; similarly for *,-,/, etc.
-------------------------------------------------------------------------------

'''
a  = a + 2
# or
a += 2
# you can also use
# np.add(x, y)
# np.subtract(x, y)
# np.multiply(x, y)
# np.divide(x, y)
# np.sqrt(x)

'''
-------------------------------------------------------------------------------
You can add two vectors if they are the same size:
'''
print( "a + 2*c = ", a + 2 * c )

'''
-------------------------------------------------------------------------------
You cannot add two vectors that are not the same size (unless one is a scalar):
'''
try:
    print( "a + b = ", a + b ) # raises a ValueError exception
except:
    print( "Got exception!" )
    
'''
-------------------------------------------------------------------------------
Operators are interpreted as elementwise, so that a*c is a vector:
'''
print( a * c )

'''
-------------------------------------------------------------------------------
Linear-algebraic operations are also defined for vectors and matrices:
'''
# Note that taking the transpose of a rank 1 array does nothing:
a.dot(c.T)  # The dot product between vectors a and c

A.dot(b.T)  # The matrix-vector product of A and c

'''
-------------------------------------------------------------------------------
Elementwise powers are ** while matrix powers use the linalg module:
'''

# The elementwise square of A: R(i,j)=A(i,j)^2
R = A ** 2         

# The matrix product R=A1*A1: R(i,j)=\sum_k A1(i,k)*A1(k,j)           
R = np.linalg.matrix_power(A1,2)  

'''
-------------------------------------------------------------------------------
Many other mathamatical operations are included in numpy:
'''
ln_a = np.log(a) # Element-wise natural log of a
exp_a = np.exp(a) # e^x for each element in a
sin_a = np.sin(a) # Element-wise sine of a
# you can also use np.true_divide(a, b) to get an array of floating points
# etc.
 
'''
=======================
Reducing arrays
=======================

-------------------------------------------------------------------------------
It is often useful to reduce arrays to a single value, e.g. by taking a sum:
'''
sum_a = 0
for elem in a: # Loop through the values in a
    sum_a += elem

print(sum_a)

'''
-------------------------------------------------------------------------------
Numpy provides fast convinience functions for summations:
'''
print( np.sum(a) )
# or
print( a.sum() )


'''
-------------------------------------------------------------------------------
Similar functions exist for means, products etc.
'''
print ( "Mean of a:", np.mean(a) ) # Take the mean of the values in a
print ( "Product of a:", np.prod(a) ) # Take the product of the values in a

'''
-------------------------------------------------------------------------------
For matrices, we can also reduce along the rows or columns:
    
       a x i s  1
      _____________
   a | 1    2    3
   x | 
   i | 4    5    6
   s |
     | 7    8    9
   0 |
     
'''
# Take the sum of each column:
print("Column sum:", np.sum(A, axis=0) )

# Take the sum of each row
print("Row sum:", np.sum(A, axis=1) )

'''
=====================================
Logical operands and logical indexing
=====================================

-------------------------------------------------------------------------------
It's often useful to use elementwise logical operations, which produce new 
(logical) vectors and matrices:
'''
a = np.array([0,1,2])
b = np.array([0,0,2])

# comparison operators produce new logical vectors or matrices:
print( "a==b : ", a==b )  # prints logical vector [1,0,1]
print( "a!=b : ", a!=b )  # prints logical vector [0,1,0]
print( "a<2  : ", a<2  )  # prints logical vector [1,1,0]
print(a[a<2]) # prints vector[0, 1]

'''
-------------------------------------------------------------------------------
For us in flow control (if, etc.), you probably want to convert these to 
scalars with any or all:
'''
print( "Any? ", np.any( a!=b ) )   # true if any a(i)!=b(i) for some i 
print( "All? ", np.all( a==b ) )  # true if all a(i)=b(i) for every i 

'''
-------------------------------------------------------------------------------
For matrices, you may want to only collapse one or more dimensions:
'''
M=[[0,1],[0,0]];
# acts on individual columns of M; returns a logical row vector 
print( np.any( M , axis=0) )
   
'''
-------------------------------------------------------------------------------
We can use these logical vectors and matrices for indexing, such as to extract 
sub-matrices:
'''
print( "Positive entries of a: ", a[ a>0 ] )



'''
=======================
MatPlotLib for plotting
=======================

MatPlotLib gives a nice plotting interface similar to Matlab / Octave.
'''

import matplotlib.pyplot as plt

'''
===========================
Scatterplots and line plots
===========================

The most typical action in Matplotlib is to plot one sequence (x-values) 
against another (y-values). We can visualize this as either a scatter plot
or a line plot.

-------------------------------------------------------------------------------
Plotting a scatter of data points:
'''
x_values = np.random.rand(1,10)   # unformly in [0,1)
y_values = np.random.randn(1,10)  # Gaussian distribution
plt.plot(x_values, y_values, 'ko');

'''
-------------------------------------------------------------------------------
'ko' is the format string specifies the marker appearance k = black, 
o = circles. We can use the format string to change the appearance:
'''
plt.plot(x_values, y_values, 'rv'); # Plot with red triangle markers

'''
-------------------------------------------------------------------------------
If we connect those points using a line appearance specification 
('-','--',':',...), it will not look very good, because the points are not 
ordered in any meaningful way. Let's try a line plot using an ordered sequence 
of x values:
'''
x_values = np.linspace(0,8,100)
y_values = np.sin(x_values)
plt.plot(x_values,y_values,'b-'); # Plot a blue connected line

'''
-------------------------------------------------------------------------------
For plotting multiple point sets or curves, you can pass more vectors into the 
plot function, or call the function multiple times:
'''
x_values = np.linspace(0,8,100)
y1 = np.sin(x_values)         # sinusoidal function
y2 = (x_values - 3)**2 / 12   # a simple quadratic curve
y3 = 0.5*x_values - 1.0       # a simple linear function

plt.plot(x_values, y1, 'b-', x_values, y2, 'g--');  # plot two curves
plt.plot(x_values, y3, 'r:'); # add a curve to the plot

'''
-------------------------------------------------------------------------------

You may want to explicitly set the plot ranges -- perhaps the most common 
pattern is to plot something, get the plot's ranges, and then restore them 
later after plotting another function:
'''

plt.plot(x_values, y1, 'b-') 
ax = plt.axis()               # get the x and y axis ranges
print( ax )
# you can set or modify the axis values explicitly if you want...

# now plot something else (which will change the axis ranges):
plt.plot(x_values, y3, 'r:'); # add the linear curve
plt.axis(ax);                 # restore the original plot's axis ranges

'''
-------------------------------------------------------------------------------
Matplot lib includes many other options for customizing the appearance of plots
see the documentation for more details.
'''
# Plot two curves and give them labels
plt.plot(x_values, y1, 'b-', label='y=sin(x)')
plt.plot(x_values, y3, 'r:', label='y=0.5x-1')

plt.title('A simple test plot') # Add a title to the plot
plt.legend() # Add a lengend using the curve labels

'''
=======================
Histograms
=======================

-------------------------------------------------------------------------------
Histograms are also useful visualizations
'''
plt.hist(y2, bins=20);

'''

The outputs of hist include the bin locations, the number of data in each bin, 
and the "handles" to the plot elements to manipulate their appearance, if 
desired.


=======================
Subplots and plot sizes
=======================

-------------------------------------------------------------------------------
It is often useful to put more than one plot together in a group; you can do this using the subplot function:
'''
plt.rcParams['figure.figsize'] = (12.0, 3.0)
fig,ax = plt.subplots(1,3)      # make a 1 x 3 grid of plots:
ax[0].plot(x_values, y1, 'b-');   # plot y1 in the first subplot
ax[1].plot(x_values, y2, 'g--');  #   y2 in the 2nd
ax[2].plot(x_values, y3, 'r:');   #   and y3 in the last

plt.rcParams['figure.figsize'] = (6.0, 4.0);  # restore default plot size 

'''
=======================
Saving figures
=======================

-------------------------------------------------------------------------------
We can save a figure with the savefig function:
'''
plt.plot(x_values, y1, 'b-', label='y=sin(x)')
plt.plot(x_values, y3, 'r:', label='y=0.5x-1')

plt.title('A simple test plot')
plt.legend()

plt.savefig('testfig.pdf')




'''
=======================
Learn from homework
=======================

'''
# prod and elementwise computation
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
a = [1,2,3]
print(A*a) # elementwise
print(np.prod(A, axis=1)) # [1*2*3, 4*5*6, 7*8*9]

# np.invert()
A = np.array([[True, False],[False, True]])
B = np.invert(A)
print(B)

# something about np.argsort()
x=np.array([1,4,3,-1,6,9])
x.argsort()
'''
输出定义为y=array([3,0,2,1,4,5])。
我们发现argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y。例如：x[3]=-1最小，所以y[0]=3,x[5]=9最大，所以y[5]=5。
example: order from largest value 89 to lowest value 1
'''
M = np.array([10,9,5,7,1,4,3,12,89,71])
print(M[np.argsort(M)[::-1]])

# np.unique(M, return_index=True)
'''
对于一维数组或者列表，unique函数去除其中重复的元素，并按元素由大到小返回一个新的无元素重复的元组或者列表
return_index=True表示返回新列表元素在旧列表中的位置，并以列表形式储存在s中。
return_inverse=True 表示返回旧列表元素在新列表中的位置，并以列表形式储存在p中
'''

# np.array_split(): Split array into multiple sub-arrays of equal size.

'''
np.concatenate()
Join a sequence of arrays along an existing axis.
The arrays must have the same shape, except in the dimension corresponding to axis
axis:default is 0
'''
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])              
np.concatenate((a, b), axis=0)

# something about array index . It's symmetric. the indices are [0,1], [1,2], [2,3]
a = np.zeros((5,4))
print(a+10)
a[np.array([0,1,2]), np.array([1,2,3])]-=1
print(a)

'''
side note
1. you can use a vector - a matrix
2. you can use reshape or keepdims
3. np.max is just an alias for np.amax. This function only works on a single input array and finds the value of maximum element in that entire array 
(returning a scalar). Alternatively, it takes an axis argument and will find the maximum value along an axis of the input array (returning a new array).
The default behaviour of np.maximum is to take two arrays and compute their element-wise maximum. 
'''

# subplot
f, ax = plt.subplots(1, 2, figsize=(20, 8))  # set subplot to be height 1 and width 2
ax[0].scatter(X[:, 0], X[:, 1], c=Z, label='single linkage') # Plotting the data
ax[1].scatter(X[:, 0], X[:, 1], c=Z, label='complete linkage') # Plotting the data
ax[0].legend(fontsize=20, loc=0)
ax[1].legend(fontsize=20, loc=0)
plt.xlabel("n")
plt.ylabel("E[Sn]")
plt.show()
# plt.set_xlim




