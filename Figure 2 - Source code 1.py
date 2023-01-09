# Figure 2 - Source code 1.

#code to generate figure 2.

################# figure 2A ###############
import matplotlib.pyplot as plt
import numpy as np
import random

random.seed()


lambda_SS = 1


x = np.linspace(0, 10, 500)
n = len(x)


def plos_minus_random(p):
    """ This function returns +1 (or -1) with probability p (or 1-p) """
    random.seed()
    x = random.random()
    if x<=p:
        r = 1
    else:
        r = -1
    return r

def list_random_plos_minus(x_min, x_max, n):
    """ This function returns an array (size n) of 
    random numbers (with sign + or -, with probability 0.5) 
    uniformly distributed between x_max and x_min """
    randList = []
    for i in range(n):
        random.seed()
        randList.append(plos_minus_random(0.5)*((x_max-x_min)*random.random() + x_min))
    return np.array(randList)

y_SS = np.exp(-x/lambda_SS)

noise = 0.025

# noise steady-state profile
y_SSnoise = y_SS + plos_minus_random(0.5)*list_random_plos_minus(0,noise, len(x))
y_up = y_SS + 0.025


# Territory defined at a low threshold
T2 = 0.1  #threshold
xT2_down = -np.log(T2 - noise)
xT2_up = -np.log(T2 + noise)
T2s = np.linspace(T2-noise, T2 + noise, 150)

for i in T2s:
    plt.plot([-np.log(i), -np.log(i)], [0, i], c = "lightcoral")

for i in T2s:
    plt.plot([0, -np.log(i)], [i, i], c = "lightcoral")
    
plt.plot([0, xT2_up], [T2 + noise, T2 + noise], c = "r")
plt.plot([0, xT2_down], [T2 - noise, T2 - noise], c = "r")
plt.plot([-np.log(T2 + noise) for i in range(2)],[0, T2 + noise], c = "r")
plt.plot([-np.log(T2 - noise) for i in range(2)],[0, T2 - noise], c = "r")


# Territory defined at a high threshold
T1 = 0.75 #threshold
xT1_down = -np.log(T1 - noise)
xT1_up = -np.log(T1 + noise)
T1s = np.linspace(T1-noise, T1 + noise, 40)

for i in T1s:
    plt.plot([-np.log(i), -np.log(i)], [0, i], c = "mediumseagreen")

for i in T1s:
    plt.plot([0, -np.log(i)], [i, i], c = "mediumseagreen")
    
plt.plot([0, xT1_up], [T1 + noise, T1 + noise], c = "g")
plt.plot([0, xT1_down], [T1 - noise, T1 - noise], c = "g")
plt.plot([-np.log(T1 + noise) for i in range(2)],[0, T1 + noise], c = "g")
plt.plot([-np.log(T1 - noise) for i in range(2)],[0, T1 - noise], c = "g")


plt.plot(x, y_SSnoise , c = "k")

plt.xlabel(r"$\bar{x}$", fontsize = 14)
plt.ylabel("[Hh]", fontsize = 14)
plt.xlim(0,5)
plt.ylim(0,1)
plt.show()

################# figure 2B ###############
import matplotlib.pyplot as plt
import numpy as np

grid = plt.GridSpec(1, 3, wspace=0.4, hspace=0.3)
plt.subplot(grid[0, :2])



lambda_SS = 1
lambda_pseudo = 2.7
lambda_over = lambda_pseudo 

x = np.linspace(0, 10, 1000)
y_SS = np.exp(-x/lambda_SS)
y_over = np.exp(-x/lambda_over)


plt.plot([0,10], [0.2, 0.2],  ":", c = "tab:purple") #theshold
plt.plot(x,y_SS, c = "k", label = "real SS") # real SS
plt.plot(x, y_over, "--", c = "k", label = "overshoot") #overshoot


m_SS = (y_SS[160] - y_SS[162])/(x[160] - x[162]) #slope of the real-SS profile at 0.2 
yt_SS = m_SS*(x-x[161]) + y_SS[161] #tangent line to real-SS at 0.2

m_over = (y_over[433] - y_over[435])/(x[433] - x[435]) #slope of the overshoot at 0.2
yt_over = m_over*(x-x[434]) + y_over[434]  #tangent line to overshoot at 0.2

m_SS_xover = (y_SS[433] - y_SS[435])/(x[433] - x[435]) # slope of the real-SS at the position of territory defined by the overshoot
yt_SS_xover = m_SS_xover*(x-x[434]) + y_SS[434] # tangent line to the real-SS at the position of territory defined by the overshoot




plt.plot(x[133:193], yt_SS[133:193],c = "r")
plt.plot(x[340:500], yt_SS_xover[340:500],c = "tab:cyan")
plt.plot(x[350:490], yt_over[350:490],c = "r")
plt.plot([x[434], x[434]], [0, y_over[434]], ":",c = "tab:purple")

#blue box
plt.plot([3.25, 3.25],[0.001, 0.28], ":", c = "tab:blue")
plt.plot([5.15, 5.15],[0.001, 0.28], ":", c = "tab:blue")
plt.plot([3.25, 5.15],[0.001, 0.001], ":", c = "tab:blue")
plt.plot([3.25, 5.15],[0.28, 0.28], ":", c = "tab:blue")
plt.xlabel(r"$\bar{x}$", fontsize = 14)
plt.ylabel("[Hh]", fontsize = 14)
plt.legend(fontsize = 12)

plt.subplot(grid[0, 2])
plt.plot(x,y_SS, c = "k", label = "real SS")
plt.plot(x, y_over, "--", c = "k", label = "overshoot")
plt.plot(x[340:500], yt_SS_xover[340:500],c = "tab:cyan")
plt.plot(x[350:490], yt_over[350:490],c = "r")

plt.gca().get_xaxis().set_visible(False)
plt.gca().get_yaxis().set_visible(False)
plt.xlim(3.24, 5.16)
plt.ylim(0.0005,0.29)
plt.show()