

from numpy import exp, arange, linspace
from numpy.random import rand
import matplotlib.pyplot as plt

def my_sigmoid(x):
    return 1/(1+exp(-x))

x = 20*(rand(1000)-1/2)
y = my_sigmoid(x)
plt.scatter(x,y)
xTicks = arange(-10, 11)
plt.xticks(xTicks)
yTicks = linspace(0,1,num=11)
plt.yticks(yTicks)
plt.grid()
plt.show()

x = 20*(rand(1000)-1/2)
y = my_sigmoid(5*x)
plt.scatter(x,y)
xTicks = arange(-10, 11)
plt.xticks(xTicks)
yTicks = linspace(0,1,num=11)
plt.yticks(yTicks)
plt.grid()
plt.show()

x = 20*(rand(1000)-1/2)
y = my_sigmoid(x-5)
plt.scatter(x,y)
xTicks = arange(-10, 11)
plt.xticks(xTicks)
yTicks = linspace(0,1,num=11)
plt.yticks(yTicks)
plt.grid()
plt.show()

x = 20*(rand(1000)-1/2)
y = my_sigmoid(5*x) + my_sigmoid(x+5) 
plt.scatter(x,y)
xTicks = arange(-10, 11)
plt.xticks(xTicks)
yTicks = linspace(0,2,num=21)
plt.yticks(yTicks)
plt.grid()
plt.show()

x = 20*(rand(1000)-1/2)
y = my_sigmoid(x) + my_sigmoid(x-5) + my_sigmoid(3*(x-10)) 
plt.scatter(x,y)
xTicks = arange(-10, 11)
plt.xticks(xTicks)
yTicks = linspace(0,3,num=31)
plt.yticks(yTicks)
plt.grid()
plt.show()

x = 20*(rand(1000)-1/2)
y = my_sigmoid(x) - my_sigmoid(x-5) 
plt.scatter(x,y)
xTicks = arange(-10, 11)
plt.xticks(xTicks)
yTicks = linspace(0,1,num=11)
plt.yticks(yTicks)
plt.grid()
plt.show()


