# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 17:39:55 2021

@author: Rosalío Reyes
"""


from over_module import *
from SS_module import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

params = pd.read_csv(r"C:\Users\rosalio\Documents\2021\Hh-robustez\Hh_robust2\params.csv")


x_plot = np.linspace(-100,1,101)



#I make variations of alpha_ptc and then alpha_Hh

## I make variations of alpha_ptc by 0.25, 0.5, 0.75, 1.25, 1.5, 1.75 y 2.

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

by_param = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,0.25,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,0.5,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,0.75,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1.25,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1.5,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1.75,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1]]

by_up = [1,0.5, 1,1,1,1,1,1,1,1,1,1,1,1,1,1]

by_down = [1,2, 1,1,1,1,1,1,1,1,1,1,1,1,1,1]

data2 = pd.DataFrame(columns = ["alpha_ptc by","x_reach_Hh_over", "t_reach_Hh_over", 
                    "x_reach_Signal_over", "t_reach_Signal_over",
                    "x_reach_Hh_SS","x_reach_Signal_SS",
                    "x_reach_Hh_over_05", "t_reach_Hh_over_05", 
                    "x_reach_Signal_over_05", "t_reach_Signal_over_05",
                    "x_reach_Hh_SS_05","x_reach_Signal_SS_05",
                    "x_reach_Hh_over_2", "t_reach_Hh_over_2", 
                    "x_reach_Signal_over_2", "t_reach_Signal_over_2",
                    "x_reach_Hh_SS_2","x_reach_Signal_SS_2"], index = range(8))



for i in range(8):
    print(i)
    
    #varitions of alpha_ptc
    par = params["value"]*by_param[i]
    sol_tr = solve_over(par, x_plot)
    Hh_tr = sol_tr[0][:,0:101]
    Hh_over = x_over_umbral(Hh_tr, 10)
    x_reach_Hh_over = Hh_over[0]
    t_reach_Hh_over = Hh_over[1]
    Signal_tr = sol_tr[1]
    x_reach_Signal_over = sol_tr[2]
    t_reach_Signal_over = sol_tr[3]
    
    sol_SS = solve_SS(par, x_plot)
    Hh_SS = sol_SS[1]
    x_reach_Hh_SS = xd(Hh_SS[-1]/10, Hh_SS)
    Signal_SS = sol_SS[2]
    x_reach_Signal_SS = sol_SS[3]
    
    print("alpha_Hhx0.5")
    #variations of alpha_Hh x 0.5
    par_05 = par*by_down
    sol_tr = solve_over(par_05, x_plot)
    Hh_tr = sol_tr[0][:,0:101]
    Hh_over = x_over_umbral(Hh_tr, 10)
    x_reach_Hh_over_05 = Hh_over[0]
    t_reach_Hh_over_05 = Hh_over[1]
    Signal_tr = sol_tr[1]
    x_reach_Signal_over_05 = sol_tr[2]
    t_reach_Signal_over_05 = sol_tr[3]
    
    sol_SS = solve_SS(par_05, x_plot)
    Hh_SS = sol_SS[1]
    x_reach_Hh_SS_05 = xd(Hh_SS[-1]/10, Hh_SS)
    Signal_SS = sol_SS[2]
    x_reach_Signal_SS_05 = sol_SS[3]
    
    print("alpha_Hhx2")
    #variations of alpha_Hh x 2
    par_2 = par*by_up
    sol_tr = solve_over(par_2, x_plot)
    Hh_tr = sol_tr[0][:,0:101]
    Hh_over = x_over_umbral(Hh_tr, 10)
    x_reach_Hh_over_2 = Hh_over[0]
    t_reach_Hh_over_2 = Hh_over[1]
    Signal_tr = sol_tr[1]
    x_reach_Signal_over_2 = sol_tr[2]
    t_reach_Signal_over_2 = sol_tr[3]
    
    sol_SS = solve_SS(par_2, x_plot)
    Hh_SS = sol_SS[1]
    x_reach_Hh_SS_2 = xd(Hh_SS[-1]/10, Hh_SS)
    Signal_SS = sol_SS[2]
    x_reach_Signal_SS_2 = sol_SS[3]
    
    data2.iloc[i] = [str(by_param[i][2]),x_reach_Hh_over, t_reach_Hh_over, 
                    x_reach_Signal_over, t_reach_Signal_over,
                    x_reach_Hh_SS,x_reach_Signal_SS,
                    x_reach_Hh_over_05, t_reach_Hh_over_05, 
                    x_reach_Signal_over_05, t_reach_Signal_over_05,
                    x_reach_Hh_SS_05,x_reach_Signal_SS_05,
                    x_reach_Hh_over_2, t_reach_Hh_over_2, 
                    x_reach_Signal_over_2, t_reach_Signal_over_2,
                    x_reach_Hh_SS_2,x_reach_Signal_SS_2]
    print("----------")
    
data2.to_csv("variation_alpha_ptc_various_alpha_Hh_05_2.csv")