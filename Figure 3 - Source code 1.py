# Figure 3 - Source code 1
# code to generate  figure 3

# All uploaded or cited files are in the same repository 
# with the same name as they are cited
############# figure 3A ############################

from over_module import *
from SS_module import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import seaborn as sns

#load the parameters
params = pd.read_csv(r"C:\Users\rosalio\Documents\2021\Hh-robustez\Hh_robust2\params.csv")

#interval where the equations are solved
x_plot = np.linspace(-100,1,101)

######### alpha_Hh by 1 ####################
par = params["value"]
#transient
sol_tr = solve_over(par, x_plot)
x_over = sol_tr[2]
# steady state
sol_SS = solve_SS(par, x_plot)
x_SS = sol_SS[3]

######### alpha_Hh perturbed ####################

variations = np.array([1/2, 3/4, 1, 4/3, 2])
perturbations = []
for i in variations:
    perturbations.append(np.array([1,i,1,1,1,1,1,1,1,1,1,1,1,1,1,1]))

# Data frame for save computations
displacements_variations_alphaHh = pd.DataFrame(columns = ["case", "Displacement", "alpha_Hh by"], index = range(2*len(perturbations)))

n = 0
for i in perturbations:
    par_pert = par*i  # parameters perturbed
    
    #transient
    sol_tr_pert = solve_over(par_pert, x_plot)
    x_over_pert = sol_tr_pert[2]
    displacements_variations_alphaHh.iloc[n] = ["overshoot", np.abs(x_over_pert-x_over), str(i[1])]
    # steady state
    sol_SS_pert = solve_SS(par_pert, x_plot)
    x_SS_pert = sol_SS_pert[3]
    displacements_variations_alphaHh.iloc[n+1] = ["steady-state", np.abs(x_SS_pert-x_SS), str(i[1])]
    n = n + 2

displacements_variations_alphaHh.to_csv("data_figure_3A_displacements_alphaHhx2-05.csv")


my_pal = {"overshoot":"red", "steady-state":"green"}
sns.scatterplot(data = displacements_variations_alphaHh, x = "alpha_Hh by", y = "Displacement", hue = "case", palette = my_pal)
plt.legend(fontsize = 14)
plt.xticks([1,2,3,4,5], ["1/2", "3/4", "1", "4/3", "2"])
plt.xlabel(r"$\alpha_{Hh} \times$", fontsize = 14)
plt.ylabel(r"Displacement", fontsize = 14)
plt.show()

################# figure 3B #########################
from over_module import *
from SS_module import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time

def random_minus_plos():
    """regresa de manera aleatoria un cero o un uno"""
    number = random.random()
    if number <= 0.5:
        result = 1
    else:
        result = -1
    return result


# the file (data) that was obtained with the code code_for_figure_3B.py 
    
# if you want to run this code you need to download the file 
#variation_all_parameters_alphaHh_1.csv that is generated with 
#the code_for_figure_3B.py, and change the path to the one that 
#corresponds to the location of the downloaded file
    
data = pd.read_csv(r"C:\Users\rosalio\Documents\2022\Hh-robustness\variation_alpha_ptc_various_1.csv")
data_dropna = data.dropna()

# for Hh
Hh_x_reach_over = data_dropna["x_reach_Hh_over"]
Hh_x_reach_over_2 = data_dropna["x_reach_Hh_over_up"]
Hh_x_reach_over_05 = data_dropna["x_reach_Hh_over_down"]
Hh_x_reach_SS = data_dropna["x_reach_Hh_SS"]
Hh_x_reach_SS_2 = data_dropna["x_reach_Hh_SS_up"]
Hh_x_reach_SS_05 = data_dropna["x_reach_Hh_SS_down"]

Hh_over_05_delta_x = Hh_x_reach_over - Hh_x_reach_over_05
Hh_over_2_delta_x = Hh_x_reach_over - Hh_x_reach_over_2
Hh_over_coef_reach = np.sqrt(Hh_over_05_delta_x*Hh_over_05_delta_x  + Hh_over_2_delta_x*Hh_over_2_delta_x )

Hh_SS_05_delta_x = Hh_x_reach_SS - Hh_x_reach_SS_05
Hh_SS_2_delta_x = Hh_x_reach_SS - Hh_x_reach_SS_2
Hh_SS_coef_reach = np.sqrt(Hh_SS_05_delta_x*Hh_SS_05_delta_x  + Hh_SS_2_delta_x*Hh_SS_2_delta_x )

# for Signal
Signal_x_reach_over = data_dropna["x_reach_Signal_over"]
Signal_x_reach_over_2 = data_dropna["x_reach_Signal_over_up"]
Signal_x_reach_over_05 = data_dropna["x_reach_Signal_over_down"]
Signal_x_reach_SS = data_dropna["x_reach_Signal_SS"]
Signal_x_reach_SS_2 = data_dropna["x_reach_Signal_SS_up"]
Signal_x_reach_SS_05 = data_dropna["x_reach_Signal_SS_down"]

Signal_over_05_delta_x = Signal_x_reach_over - Signal_x_reach_over_05
Signal_over_2_delta_x = Signal_x_reach_over - Signal_x_reach_over_2
Signal_over_coef_reach = np.sqrt(Signal_over_05_delta_x*Signal_over_05_delta_x  + Signal_over_2_delta_x*Signal_over_2_delta_x )

Signal_SS_05_delta_x = Signal_x_reach_SS - Signal_x_reach_SS_05
Signal_SS_2_delta_x = Signal_x_reach_SS - Signal_x_reach_SS_2
Signal_SS_coef_reach = np.sqrt(Signal_SS_05_delta_x*Signal_SS_05_delta_x  + Signal_SS_2_delta_x*Signal_SS_2_delta_x )

dif_Signal_over_05_delta_x = np.abs(Signal_over_05_delta_x) - np.abs(Signal_SS_05_delta_x)
dif_Signal_over_2_delta_x = np.abs(Signal_over_2_delta_x) - np.abs(Signal_SS_2_delta_x)
dif_Signal_coef_reach = Signal_over_coef_reach - Signal_SS_coef_reach

diferencia_coef_reach = Signal_over_coef_reach - Signal_SS_coef_reach
plt.plot([2 + random_minus_plos()*random.random()/1  for i in range(len(diferencia_coef_reach))], diferencia_coef_reach, ".", color = "k")
plt.plot([i for i in range(1,4)], [0 for i in range(1,4)], c = "r")
plt.xticks([0], [])
plt.xlabel("code runs", fontsize = 14)
plt.ylabel("CD(overshoot) - CD(steady-state)", fontsize = 14)
plt.show()

################# figure 3C #########################

from over_module import *
from SS_module import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time

params = pd.read_csv(r"C:\Users\rosalio\Documents\2021\Hh-robustez\Hh_robust2\params.csv")
x_plot = np.linspace(-100,1,101)

def Hh_de_Signal(Signal, params):
    k = params[13]*params[9]/params[10]
    return pow((pow(k,params[15]))/((params[4]/params[8])*(1/Signal)-1), 1/params[15])

def x_over_umbral(sig,um):
    reachs = []
    for i in range(len(sig)):
        r = xd(sig[i][-1]/um, sig[i])
        reachs.append([r,i])
    reachs.sort()
    return reachs[-1]

up = 2
down = 0.5
by_up = np.array([1,up,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
by_down = np.array([1,down,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

######### alpha_Hh by 1 ####################
par = params["value"]
#transient
sol_tr = solve_over(par, x_plot)
Signal_tr = sol_tr[1]
# steady state
sol_SS = solve_SS(par, x_plot)
Signal_SS = sol_SS[2]


######### alpha_hh by 0.5 ###################
par_down = par*by_down
#transient
sol_tr_down = solve_over(par_down, x_plot)
Signal_tr_down = sol_tr_down[1]
# steady state
sol_SS_down = solve_SS(par_down, x_plot)
Signal_SS_down = sol_SS_down[2]

######## alpha_Hh by 2 ####################
par_up = par*by_up
#transient
sol_tr_up = solve_over(par_up, x_plot)
Signal_tr_up = sol_tr_up[1]
# steady state
sol_SS_up = solve_SS(par_up, x_plot)
Signal_SS_up = sol_SS_up[2]


set_for_bucle = range(0,len(Signal_tr),500)
data = pd.DataFrame(columns = ["alpha_Hh", "time", "x_reach"], index = range(3*len(set_for_bucle)))


count = 0
for i in set_for_bucle:
    x_reach1_tr = xd(np.amax(Signal_tr[i])*0.2, Signal_tr[i])
    data.iloc[count] = ["1", i, x_reach1_tr]
    
    x_reach05_tr = xd(np.amax(Signal_tr_down[i])*0.2, Signal_tr_down[i])
    count = count + 1
    data.iloc[count] = [str(down), i, x_reach05_tr]
    
    x_reach2_tr = xd(np.amax(Signal_tr_up[i])*0.2, Signal_tr_up[i])
    count = count + 1
    data.iloc[count] = [str(up), i, x_reach2_tr]
    count = count + 1

x_reach1 = data[data["alpha_Hh"] == "1"]
x_reach05 = data[data["alpha_Hh"] == "0.5"]
x_reach2 = data[data["alpha_Hh"] == "2"]

delta_down = np.array(x_reach1["x_reach"]) - np.array(x_reach05["x_reach"])
delta_up = np.array(x_reach2["x_reach"]) - np.array(x_reach1["x_reach"])

import math
import matplotlib as mpl 
import numpy as np
viridis = plt.get_cmap('viridis', 80000)
aaa = np.array(coef_reach[1:80])

bbb = []
for i in aaa:
    bbb.append(np.exp(i))
bbb = np.array(bbb)


count = 1
for i in bbb:
    n = int(round(i))
    if (count == 1):
        n1 = n
        
    if (count == 79):
        n2 = n
    plt.plot([count,count],[0,12], c = viridis(n), linewidth = 5)
    count = count + 1

plt.plot([0,0],[0,12], c = viridis(n1), linewidth = 5)
plt.plot([80,80],[0,12], c = viridis(n2), linewidth = 5)
#plt.plot([i for i in range(80)], [coef_reach[-1] for i in range(80)], c = "lightgray")
plt.plot([i for i in range(1,80)], aaa, c = "white", linewidth = 2)

# la barra de color
mult = np.linspace(2,11.5,80)
norm = mpl.colors.Normalize(vmin = mult[0], vmax = mult[-1]) 
sm = plt.cm.ScalarMappable(cmap=viridis, norm=norm) 
sm.set_array([]) 
plt.colorbar(sm, ticks=np.linspace(mult[0], mult[-1], 7), label = "CD") 


             
plt.ylabel("CD", fontsize = 14)
plt.xticks([i for i in range(0,80,10)], [500*i for i in range(0,80,10)], rotation = 30)
plt.xlabel("time step", fontsize = 14)
plt.xlim(0,80)
plt.ylim(0,12)

plt.text(12, 6, "overshoot", rotation = 90, fontsize = 14)
plt.text(75, 5, "steady-state", rotation = 90, color = "white", fontsize = 14)
plt.show()

################# figure 3D #########################
from over_module import *
from SS_module import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load the params
params = pd.read_csv(r"C:\Users\rosalio\Documents\2021\Hh-robustez\Hh_robust2\params.csv")
par = params["value"]

# interval where equations are soluted
x_plot = np.linspace(-100,1,101)

# compute the solution
sol_tr = solve_over(par, x_plot)
sol_SS = solve_SS(par, x_plot)

# the functions computed
Hh_SS = sol_SS[1]
Hh_tr = sol_tr[0][:,0:101]
x_over = sol_tr[2]
t_over = sol_tr[3]
x_SS = sol_SS[3]

# plots 
plt.plot(-x_plot, Hh_tr[t_over], "--", label = "overshoot",c = "k")
plt.plot(-x_plot, Hh_tr[57600],label = "steady-state", c = "k")
plt.plot([x_over, x_over], [-0.1, 0.04], ":", c = "k")

#slope of the steady-state profile
m_SS_xover = (Hh_tr[57600, -27] - Hh_tr[57600, -28])/(-x_plot[-27] + x_plot[-28])
print("m_SS_xover = " , m_SS_xover)

# slope of the overshoot profile
m_over_xover = (Hh_tr[t_over, -27] - Hh_tr[t_over, -28])/(-x_plot[-27] + x_plot[-28])
print("m_over_xover = " , m_over_xover)

# tangent line equations
yt_SS_xover = -m_SS_xover*(x_plot - x_plot[-28])+ Hh_tr[57600, -28]
yt_over_xover = -m_over_xover*(x_plot - x_plot[-28])+ Hh_tr[t_over, -28]

plt.plot(-x_plot[60:90], yt_SS_xover[60:90], c = "r")
plt.plot(-x_plot[60:90], yt_over_xover[60:90], c = "r")

plt.legend(fontsize = 14)
plt.xlim(-3,60)
plt.ylim(-0.05,0.5)
plt.ylabel("[Hh]", fontsize = 14)
plt.xlabel("position $(\mu m)$", fontsize = 12)
plt.text(23,-0.09, "dpp", fontsize = 14)
plt.show()

################# figure 3E #########################
from over_module import *
from SS_module import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load the params
params = pd.read_csv(r"C:\Users\rosalio\Documents\2021\Hh-robustez\Hh_robust2\params.csv")
par = params["value"]

# interval where equations are soluted
x_plot = np.linspace(-100,1,101)

# compute the solution
sol_tr = solve_over(par, x_plot)
sol_SS = solve_SS(par, x_plot)
# computed functions
signal_tr = sol_tr[1]
t_over = sol_tr[3]

plt.plot(-x_plot, signal_tr[t_over], "--" , label ="overshoot", c = "k")
plt.plot(-x_plot, signal_tr[57600],label = "steady-state", c = "k" )
plt.plot([x_over, x_over], [-0.1, 0.05], ":", c = "k")

m_signalover_xover  = (signal_tr[t_over, -28] - signal_tr[t_over, -27])/(-x_plot[-28] + x_plot[-27])
m_signalSS_xover  = (signal_tr[57600, -28] - signal_tr[57600, -27])/(-x_plot[-28] + x_plot[-27])

yt_signalSS_xover = -m_signalSS_xover*(x_plot - x_plot[-27]) + signal_tr[57600, -27]
yt_signalover_xover = -m_signalover_xover*(x_plot - x_plot[-27]) + signal_tr[t_over, -27]
plt.plot(-x_plot[60:90], yt_signalSS_xover[60:90], c = "r")
plt.plot(-x_plot[70:77], yt_signalover_xover[70:77], c = "r")

plt.ylabel("[Signal]",fontsize = 14)
plt.xlabel("position $(\mu m)$",fontsize = 14)

plt.xlim(-3,60)
plt.ylim(-0.03,0.3)
plt.legend(fontsize = 14)
plt.text(23,-0.05, "dpp",fontsize = 14)
plt.show()