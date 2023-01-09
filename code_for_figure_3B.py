"""
Created on Tue Jan 25 12:34:10 2022

@author: Rosalío Reyes

Here I made random variations of all the parameters of the equation system in
box 3, leaving only alpha_Hh fixed. I then made variations of alpha_Hh by 2 
and by 0.5, to show that the differential robustness does not depend on the 
chosen parameters.

This code takes more than 10 hours running
"""

##### 25/01/2022
from over_module import *
from SS_module import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time

#load the parameters
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

par = params["value"]*[1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1]
    #solución transitoria


t11 = time.time()
for i in range(250):
    #t1 = time.time()
    print(i)
    byrandom = [byfunction(random.random()), 1, 1, 
            byfunction(random.random()), byfunction(random.random()), 
            byfunction(random.random()), byfunction(random.random()), 
            byfunction(random.random()), byfunction(random.random()), 
            byfunction(random.random()), byfunction(random.random()), 
            byfunction(random.random()), byfunction(random.random()), 
            byfunction(random.random()), byfunction(random.random()), 
            byfunction(random.random())]
################### WT ####################################
    par = params["value"]*byrandom
    #solución transitoria
    sol_tr = solve_over(par, x_plot)
    Hh_tr = sol_tr[0][:,0:101]
    Hh_over = x_over_umbral(Hh_tr, 10)
    x_reach_Hh_over = Hh_over[0]
    t_reach_Hh_over = Hh_over[1]
    Signal_tr = sol_tr[1]
    x_reach_Signal_over = sol_tr[2]
    t_reach_Signal_over = sol_tr[3]
    
    #solución estacionaria
    sol_SS = solve_SS(par, x_plot)
    Hh_SS = sol_SS[1]
    x_reach_Hh_SS = xd(Hh_SS[-1]/10, Hh_SS)
    Signal_SS = sol_SS[2]
    x_reach_Signal_SS = sol_SS[3]
    
#################### by up ##################################
    par_up = par*by_up
    #solución transitoria
    sol_tr_up = solve_over(par_up, x_plot)
    Hh_tr_up = sol_tr_up[0][:,0:101]
    Hh_over_up = x_over_umbral(Hh_tr_up, 10)
    x_reach_Hh_over_up = Hh_over_up[0]
    t_reach_Hh_over_up = Hh_over_up[1]
    Signal_tr_up = sol_tr_up[1]
    x_reach_Signal_over_up = sol_tr_up[2]
    t_reach_Signal_over_up = sol_tr_up[3]
    
    #solución estacionaria
    sol_SS_up = solve_SS(par_up, x_plot)
    Hh_SS_up = sol_SS_up[1]
    x_reach_Hh_SS_up = xd(Hh_SS_up[-1]/10, Hh_SS_up)
    Signal_SS_up = sol_SS_up[2]
    x_reach_Signal_SS_up = sol_SS_up[3]
################### by down #################################
    par_down = par*by_down
    #solución transitoria
    sol_tr_down = solve_over(par_down, x_plot)
    Hh_tr_down = sol_tr_down[0][:,0:101]
    Hh_over_down = x_over_umbral(Hh_tr_down, 10)
    x_reach_Hh_over_down = Hh_over_down[0]
    t_reach_Hh_over_down = Hh_over_down[1]
    Signal_tr_down = sol_tr_down[1]
    x_reach_Signal_over_down = sol_tr_down[2]
    t_reach_Signal_over_down = sol_tr_down[3]
    
    #solución estacionaria
    sol_SS_down = solve_SS(par_down, x_plot)
    Hh_SS_down = sol_SS_down[1]
    x_reach_Hh_SS_down = xd(Hh_SS_down[-1]/10, Hh_SS_down)
    Signal_SS_down = sol_SS_down[2]
    x_reach_Signal_SS_down = sol_SS_down[3]
################### by down #################################
    
    
    
    
    data.iloc[i] = [x_reach_Hh_over, t_reach_Hh_over, 
                    x_reach_Signal_over, t_reach_Signal_over,
                    x_reach_Hh_SS,x_reach_Signal_SS,
                    x_reach_Hh_over_up, t_reach_Hh_over_up, 
                    x_reach_Signal_over_up, t_reach_Signal_over_up,
                    x_reach_Hh_SS_up,x_reach_Signal_SS_up,
                    x_reach_Hh_over_down, t_reach_Hh_over_down, 
                    x_reach_Signal_over_down, t_reach_Signal_over_down,
                    x_reach_Hh_SS_down,x_reach_Signal_SS_down]
    print("----------")
    #t2 = time.time()
    #print(t2-t1)
    
    
    
    
data.to_csv("variation_all_parameters_alphaHh_1.csv")

t22 = time.time()
print("tiempo total", t22-t11)

