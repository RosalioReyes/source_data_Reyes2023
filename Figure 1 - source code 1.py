# Figure 1 - source code 1. 

#code to generate figure 1

import numpy as np
import matplotlib.pyplot as plt
import math 

grid = plt.GridSpec(3, 2, wspace=0.4, hspace=0.5)


lambda_SS = 1
#We approximate the transient solution by an exponential function 
#whose lambda length lies between lambda_SS and lambda_SSpseudo

lambda_pseudo = 2.7
lambda_over = lambda_pseudo

x = np.linspace(0, 10, 1000)

# Full dose
y_SS = np.exp(-x/lambda_SS)
y_over = np.exp(-x/lambda_over)
y_pseudo = np.exp(-x/lambda_pseudo)

# Half dose
y_SS_05 = 0.5*np.exp(-x/lambda_SS)
y_over_05 = 0.5*np.exp(-x/lambda_over)
y_pseudo_05 = 0.5*np.exp(-x/lambda_pseudo)

# we consider that the collier territory is determined at 0.2
# of the source concentration. That implies that
# x_col = lambda_SS*ln(1/0.2) = lambda_SS*ln(5)
x_col = math.log(5)
x_colp = math.log(2.5)   # x_colp means the border of the territory upon half perturbation dosage
x_dpp = 2.7*math.log(5)  # x_dpp determined by overshoot
T_dpp = math.exp(-x_dpp) # threshold concentration in the Steady-state 
                         #model (two thresholds) where dpp is defined by the dynamical model
                         #(the second threshold)
x_dppP = math.log(0.5/T_dpp)  # x_dppP means the border of the dpp territory upon half perturbation dosage (in the steady-state model)
x_overP = 2.7*math.log(2.5) # x_overP means the border of the dpp territory upon half perturbation dosage (in the dynamical model)




plt.subplot(grid[0, 0])
plt.plot([0,10], [0.2, 0.2],  "--", linewidth = 0.9, c = "tab:purple")
plt.plot([0,10], [T_dpp, T_dpp], "--", linewidth = 0.9, c = "tab:purple")
plt.plot(x,y_SS, c = "k", label = "real SS")
plt.plot([x_col, x_col], [-0.05,0.2], c = "g")
plt.plot([x_dpp, x_dpp], [-0.05,T_dpp], c = "r")
plt.ylabel("[Hh]", fontsize=14)
plt.title("Steady-state \n (two thresholds)")
plt.xlim(0,5)
plt.ylim(-0.05,1)
plt.text(x_col-0.5, -0.25, "col")
plt.text(x_dpp, -0.25, "dpp")
plt.text(1.5, 0.7, "Unperturbed")

plt.subplot(grid[1, 0])
plt.plot([0,10], [0.2, 0.2],  "--", linewidth = 0.9, c = "tab:purple")
plt.plot([0,10], [T_dpp, T_dpp], "--", linewidth = 0.9, c = "tab:purple")
plt.plot(x,y_SS_05, c = "k", label = "real SS")
plt.plot([x_colp, x_colp], [-0.05,0.2], c = "g")
plt.plot([x_dppP, x_dppP], [-0.05,T_dpp], c = "r")
plt.ylabel("[Hh]", fontsize=14)
plt.xlim(0,5)
plt.text(x_colp-0.4, -0.25, "col")
plt.text(x_dppP-0.5, -0.25, "dpp")
plt.text(1.5, 0.7, "Perturbed")
plt.ylim(-0.05,1)



plt.subplot(grid[2, 0])
line = 0.2
h = 0.03
plt.plot([x_colp, x_col], [line, line], c = "g")
plt.plot([x_colp, x_colp], [line-h, line+h], c = "g")
plt.plot([x_col, x_col], [line-h, line+h], c = "g")

plt.plot([x_dppP, x_dpp], [line, line], c = "r")
plt.plot([x_dpp, x_dpp], [line-h, line+h], c = "r")
plt.plot([x_dppP, x_dppP], [line-h, line+h], c = "r")
plt.xlabel(r"$\bar{x}$", fontsize = 14)
plt.xlim(0,5)
plt.ylim(0.1,0.7)
plt.text(1.5, 0.55, "Displacement")
plt.text(x_colp, 0.3, "0.69")  # 0.69 = x_col - xcolp
plt.text(x_dppP, 0.3, "0.69")  # 0.69 = x_dpp - x_dppP
plt.gca().get_yaxis().set_visible(False)



plt.subplot(grid[0, 1])
plt.plot([0,10], [0.2, 0.2],  "--", linewidth = 0.9, c = "tab:purple")
plt.plot(x,y_SS, c = "k", label = "Steady-State")
plt.plot(x,y_over, "--" , c = "k", label = "Overshoot")
plt.plot([x_col, x_col], [-0.05,0.2], c = "g")
plt.plot([x_dpp, x_dpp], [-0.05,0.2], c = "r")
plt.ylabel("[Hh]", fontsize=14)
plt.title("Dynamical interpretation \n (single threshold)")
plt.xlim(0,5)
plt.ylim(-0.05,1)
plt.text(x_col-0.5, -0.25, "col")
plt.text(x_dpp, -0.25, "dpp")
plt.legend(fontsize = 6.5)



plt.subplot(grid[1, 1])
plt.plot([0,10], [0.2, 0.2],  "--", linewidth = 0.9, c = "tab:purple")
plt.plot(x,y_SS_05, c = "k", label = "real SS")
plt.plot(x,y_over_05, "--" , c = "k")
plt.plot([x_colp, x_colp], [-0.05,0.2], c = "g")
plt.plot([x_overP, x_overP], [-0.05,0.2], c = "r")
plt.ylabel("[Hh]", fontsize=14)
plt.xlim(0,5)
plt.ylim(-0.05,1)
plt.text(1.5, 0.7, "Perturbed")
plt.text(x_col-0.7, -0.25, "col")
plt.text(x_overP, -0.25, "dpp")


plt.subplot(grid[2, 1])
line = 0.2
h = 0.03
plt.plot([x_colp, x_col], [line, line], c = "g")
plt.plot([x_colp, x_colp], [line-h, line+h], c = "g")
plt.plot([x_col, x_col], [line-h, line+h], c = "g")

plt.plot([x_overP, x_dpp], [line, line], c = "r")
plt.plot([x_dpp, x_dpp], [line-h, line+h], c = "r")
plt.plot([x_overP, x_overP], [line-h, line+h], c = "r")
plt.xlabel(r"$\bar{x}$", fontsize = 14)
plt.xlim(0,5)
plt.ylim(0.1,0.7)
plt.text(1.5, 0.55, "Displacement")
plt.text(x_colp, 0.3, "0.69")
plt.text(x_overP + 0.6, 0.3, "1.87") # 1.87 = x_dpp - x_overP
plt.gca().get_yaxis().set_visible(False)


# alphabetical order
plt.text(-8.2, 2.5, "A", fontsize = 14, weight='bold')
plt.text(-1.2, 2.5, "A'", fontsize = 14, weight='bold')
plt.text(-8.2, 1.6, "B", fontsize = 14, weight='bold')
plt.text(-1.2, 1.6, "B'", fontsize = 14, weight='bold')
plt.text(-8.2, 0.6, "C", fontsize = 14, weight='bold')
plt.text(-1.2, 0.6, "C'", fontsize = 14, weight='bold')
plt.show()
