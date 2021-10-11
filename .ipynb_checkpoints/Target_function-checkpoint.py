import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

shape_pts = np.array([[-8,1],[-6,10],[-5,-7],[0,-5],[1,9],[2,-8],[3,28],[5,-23],[7,12],[9,1],[10,10],[12,15],[14,18],[15,10]])

def TargetFunction(lst):
    x = []
    y = []
    for i in range(len(lst)):
        x.append(lst[i][0])
        y.append(lst[i][1])
    #print(x)
    #print(y)
    #fig, f = plt.subplots(figsize=(10, 3))
#     f.step(x,y,where = 'mid')
#     line_up = f.step(x,y,'r*',where = 'mid',label = 'Shape points')
#     f.set_xlabel('x')
#     f.set_ylabel('f(x)')
#     f.grid(True)
#     f.set_title('Target Function')
    f = plt.figure()
    f.set_figwidth(15)
    f.set_figheight(5)
    plt.step(x,y,where = 'mid')
    plt.step(x,y,'r*',where = 'mid', label = 'Shape Points')
    plt.legend(['Shape Points'])
    plt.title('Target Function')
    plt.grid()
    plt.legend(loc ="upper right")
    return plt

#y=TargetFunction(shape_pts)[1]
#x=TargetFunction(shape_pts)[0]


# plt.step(x,y,where = 'mid')
# plt.step(x,y,'r*',where = 'mid', label = 'Shape Points',grid)
# plt.legend(['Shape Points'])
# plt.plot(n_plot_pts=200, show_shape_pts=True, title='Target Function')


f = TargetFunction(shape_pts)
#z = TargetFunction(shape_pts1)
display(f)
#display(z)
#f.plot(n_plot_pts=200, show_shape_pts=True, title='Target Function')
 
