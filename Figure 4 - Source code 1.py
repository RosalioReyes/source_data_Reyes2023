# Figure 4 - Source code 1
# code to generate  figure 4

# All uploaded or cited files are in the same repository 
# with the same name as they are cited

############# figure 4A ############################
#After running the code_for_figure_4A.py code we get the data
# coef_var_Signal_over, that contains CD for the overshoot for different variations of alpha_ptc (by 0, 0.2, 0.4, 0.6, 0.8 and 1)
# coef_var_Signal_SS that contains CD for the steady state for different variations of alpha_ptc (by 0, 0.2, 0.4, 0.6, 0.8 and 1)

import matplotlib.pyplot as plt
alphas_ptc = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
coef_var_Signal_over = [11.6185964119825, 11.6053107991704, 11.6116591522483, 11.5986352948764, 11.5866848748941, 11.5581714353089, 11.5128277008768, 11.4858302233438, 11.4596223804869, 11.4804160593383, 11.4692407756499]
coef_var_Signal_SS = [11.6185964119825, 11.404459892379, 11.1962409818151, 10.9957608052504, 10.8059839126404, 10.6606756610001, 10.4837402845933, 10.3104257664269, 10.1873863923186, 10.0299010674943, 9.87721463299978]
plt.plot(alphas_ptc, coef_var_Signal_over, "o", label = "overshoot", color = "r")
plt.plot(alphas_ptc, coef_var_Signal_SS, "o", label = "steady-state", color = "green")
plt.xlabel(r"$\alpha_{ptc} \times$", fontsize = 14)
plt.ylabel("CD",fontsize = 14)
plt.legend(fontsize = 14)
plt.show()

############# figure 4B ############################
# After running the code_for_figure_4B.py code we get the data
# variation_all_parameter_alphaHh_alpha_ptc.csv
# it contains the reach for different variations on the set of parameters
# in each variation: First we vary all the parameters except alpha_Hh.
# We then varied alpha_Hh by 0.5 and 2. Finally we did the same, 
# considering alpha_ptc = 0.

# Then we load the file variation_all_parameter_alphaHh_alpha_ptc.csv 
# in order to compute CD(for the case alpha_ptc = 0) - CD(for the case alpha not equal to 0)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


data_ptc = pd.read_csv(r"C:\Users\rosalio\Documents\2022\Hh-robustness\variation_all_parameter_alphaHh_alpha_ptc.csv")

data_ptc_regulation = data_ptc[data_ptc["ptc regulation"] == "Yes"]
Signal_ptc_regulation = data_ptc_regulation[data_ptc_regulation["Function"] == "Signal"]
Signal_ptc_regulation_SS = Signal_ptc_regulation[Signal_ptc_regulation["Model"] == "SS"]
Signal_ptc_regulation_Over = Signal_ptc_regulation[Signal_ptc_regulation["Model"] == "Overshoot"]

Signal_ptc_regulation_SS_alphaHhx1 = Signal_ptc_regulation_SS[Signal_ptc_regulation_SS["alphaHh_by"] == 1.0]
Signal_ptc_regulation_SS_alphaHhx05 = Signal_ptc_regulation_SS[Signal_ptc_regulation_SS["alphaHh_by"] == 0.5]
Signal_ptc_regulation_SS_alphaHhx2 = Signal_ptc_regulation_SS[Signal_ptc_regulation_SS["alphaHh_by"] == 2.0]

Signal_ptc_regulation_Over_alphaHhx1 = Signal_ptc_regulation_Over[Signal_ptc_regulation_Over["alphaHh_by"] == 1.0]
Signal_ptc_regulation_Over_alphaHhx05 = Signal_ptc_regulation_Over[Signal_ptc_regulation_Over["alphaHh_by"] == 0.5]
Signal_ptc_regulation_Over_alphaHhx2 = Signal_ptc_regulation_Over[Signal_ptc_regulation_Over["alphaHh_by"] == 2.0]

# without ptc regulation

data_NOptc_regulation = data_ptc[data_ptc["ptc regulation"] == "No"]
Signal_NOptc_regulation = data_NOptc_regulation[data_NOptc_regulation["Function"] == "Signal"]
Signal_NOptc_regulation_SS = Signal_NOptc_regulation[Signal_NOptc_regulation["Model"] == "SS"]
Signal_NOptc_regulation_Over = Signal_NOptc_regulation[Signal_NOptc_regulation["Model"] == "Overshoot"]

Signal_NOptc_regulation_SS_alphaHhx1 = Signal_NOptc_regulation_SS[Signal_NOptc_regulation_SS["alphaHh_by"] == 1.0]
Signal_NOptc_regulation_SS_alphaHhx05 = Signal_NOptc_regulation_SS[Signal_NOptc_regulation_SS["alphaHh_by"] == 0.5]
Signal_NOptc_regulation_SS_alphaHhx2 = Signal_NOptc_regulation_SS[Signal_NOptc_regulation_SS["alphaHh_by"] == 2.0]

Signal_NOptc_regulation_Over_alphaHhx1 = Signal_NOptc_regulation_Over[Signal_NOptc_regulation_Over["alphaHh_by"] == 1.0]
Signal_NOptc_regulation_Over_alphaHhx05 = Signal_NOptc_regulation_Over[Signal_NOptc_regulation_Over["alphaHh_by"] == 0.5]
Signal_NOptc_regulation_Over_alphaHhx2 = Signal_NOptc_regulation_Over[Signal_NOptc_regulation_Over["alphaHh_by"] == 2.0]


# computacion of CD

#coef_reach_Signal_ptc_regulation_SS
coef_reach_Signal_ptc_regulation_SS = np.power(np.power(np.array(Signal_ptc_regulation_SS_alphaHhx1["x_reach"]) - np.array(Signal_ptc_regulation_SS_alphaHhx05["x_reach"]),2) + np.power(np.array(Signal_ptc_regulation_SS_alphaHhx1["x_reach"]) - np.array(Signal_ptc_regulation_SS_alphaHhx2["x_reach"]),2), 0.5)

#coef_reach_Signal_NOptc_regulation_SS 
coef_reach_Signal_NOptc_regulation_SS = np.power(np.power(np.array(Signal_NOptc_regulation_SS_alphaHhx1["x_reach"]) - np.array(Signal_NOptc_regulation_SS_alphaHhx05["x_reach"]),2) + np.power(np.array(Signal_NOptc_regulation_SS_alphaHhx1["x_reach"]) - np.array(Signal_NOptc_regulation_SS_alphaHhx2["x_reach"]),2), 0.5)


dif_coef_reach = coef_reach_Signal_NOptc_regulation_SS - coef_reach_Signal_ptc_regulation_SS

def random_minus_plos():
    """regresa de manera aleatoria un cero o un uno"""
    number = random.random()
    if number <= 0.5:
        result = 1
    else:
        result = -1
    return result

plt.plot([2 + random_minus_plos()*random.random()/1  for i in range(len(dif_coef_reach))], dif_coef_reach, ".", color = "k")
plt.plot([i for i in range(1,4)], [0 for i in range(1,4)], c = "r")
plt.xticks([0], [])
plt.xlabel("code runs", fontsize = 14)
plt.ylabel(r"CD($\alpha_{ptc} = 0 $) - CD($\alpha_{ptc} \times 1$)",  fontsize = 14)
plt.show()
