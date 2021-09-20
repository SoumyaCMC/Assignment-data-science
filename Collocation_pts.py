import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate


shape_pts = np.array([[-8,1],[-6,10],[-5,-7],[0,-5],[1,9],[2,-8],[3,28],[5,-23],[7,12],[9,1],[10,10],[12,15],[14,18],[15,10]])

x = []
y = []
for i in range(len(shape_pts)):
    x.append(shape_pts[i][0])
    y.append(shape_pts[i][1])

x_min = shape_pts[0][0]
x_max = shape_pts[-1][0]
y_max=max(y)
y_min=min(y)

m = 100
assert m >= len(shape_pts)
y_tilde_pts = np.linspace(y_min,y_max,m)
x_tilde_pts = np.linspace(x_min, x_max, m)

print(x_tilde_pts,'\n')



print(x,'\n',y)
u = interpolate.interp1d(x, y, kind='nearest',fill_value='array-like')


y_new = u(x_tilde_pts)
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(5)
plt.step(x,y,where = 'mid')
plt.step(x,y,'r*',where = 'mid', label = 'Shape Points')
#plt.plot(x_tilde_pts,z ,color='b',marker='x')
plt.grid()
plt.title('Collocation Points on Target Function')
plt.scatter(x_tilde_pts,y_new,c='#3396FF',marker='X',label='Collocation Points')
plt.legend()
plt.show()