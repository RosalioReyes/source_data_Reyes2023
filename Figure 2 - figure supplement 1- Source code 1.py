# Figure 2 - figure supplement 1. Source code 1
# code to generate  Figure 2 - figure supplement 1

# All uploaded or cited files are in the same repository 
# with the same name as they are cited


#########  Figure 2 - figure supplement 1 A-A' ##############
import matplotlib.pyplot as plt
import numpy as np
grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

x = np.linspace(0, 10, 1000)

f = 2.7
y_SS = np.exp(-x)
y_over = np.exp(-x/f)
y_nonLinear = 1/(x/(6**(1/2)) + (2/3)**(1/6))**2

plt.subplot(grid[:2, :2])
plt.plot(x,y_SS, c = "k", label = "simple")
plt.plot(x, y_nonLinear, c = "g", label = "non Linear")
plt.plot(x, y_over, "--", c = "k", label = "overshoot")
plt.legend()
plt.ylim(0,1)
plt.ylabel("concentration")
plt.xlabel(r"$\bar{x}$")

plt.subplot(grid[0, 2])

x = np.linspace(0, 10, 1000)
plt.plot(x, np.ones(len(x)), c = "k", linewidth = 0.5)
a = 2
y = a*np.exp(x*(1/a - 1))
for i in range(145, 999,1):
    plt.plot([x[i], x[i]], [y[i]+0.01,1], c = "r", linewidth = 1)
lb = "f = " + str(a)
plt.plot(x,y, label = lb, c = "k")
plt.text(4, 0.6, "Best   $M_{over}$")
plt.ylabel(r"$P_{over} \ / \ P_{simple}$")



plt.subplot(grid[1, 2])
x = np.linspace(0, 8, 1000)
a = 2
plt.plot(x, np.ones(len(x)), c = "k", linewidth = 0.5)
y = (2*a/pow(6,1/2))*np.exp(x/a)/pow(x/pow(6,1/2) + pow(2/pow(6,1/2),1/3),3)    

for i in range(161, 879,1):
    plt.plot([x[i], x[i]], [y[i]+0.01,1], c = "r", linewidth = 1)
lb = "f = " + str(a)
plt.plot(x,y, label = lb, c = "k")
plt.xlabel(r"$\bar{x}$")
plt.ylabel(r"$P_{over}\ / \ P_{nonLinear}$")
plt.text(1.6, 0.85, "Best   $M_{over}$")

plt.subplots_adjust(right=1.1)

plt.show()

#########  Figure 2 - figure supplement 1 B ##############
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
mult = [1.5, 2.5, 3.5]
plt.plot(x, np.ones(len(x)), c = "k", linewidth = 0.5)
for a in mult:
    y = a*np.exp(x*(1/a - 1))
    lb = "f = " + str(a)
    plt.plot(x,y, label = lb)
    
plt.legend()
plt.xlabel(r"$\bar{x}$")
plt.ylabel(r"$P_{over} \ / \ P_{simple}$")
plt.show()

#########  Figure 2 - figure supplement 1 C ##############
x = np.linspace(0, 20, 100)
mult = [1.5, 2.5, 3.5]
plt.plot(x, np.ones(len(x)), c = "k", linewidth = 0.5)
for a in mult:
    y = (2*a/pow(6,1/2))*np.exp(x/a)/pow(x/pow(6,1/2) + pow(2/pow(6,1/2),1/3),3)
    lb = "f = " + str(a)
    plt.plot(x,y, label = lb)
   
plt.legend()
plt.xlabel(r"$\bar{x}$")
plt.ylabel(r"$P_{over} \ / \ P_{nonLinear}$")
plt.ylim(0.3,1.2)
