# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 14:28:04 2022

@author: Rosalío Reyes

Here I will consider two models. The original model and the model alpha_ptc = 0. 
So I am going to make variations in all the parameters by 0.5 and by 2, then 
variations in alpha_Hh by 0.5 and 2, to see how the robustness of the different 
models compares.
"""


from over_module import *
from SS_module import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time

params = pd.read_csv(r"C:\Users\rosalio\Documents\2021\Hh-robustez\Hh_robust2\params.csv")

x_plot = np.linspace(-100,1,101)

# All parameters can vary randomly by 0.5 or by 2,
# keeping alpha_Hh constant, then varying alphaHh x 2 or by 0.5

def byfunction(number):
    if 0<= number <= 0.33:
        result = 0.5
    if 0.33< number <= 0.66:
        result = 1
    if 0.66 < number <= 1:
        result = 2
    return result

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


up = 0.5
down = 2
by_up = np.array([1,up,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
by_down = np.array([1,down,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

#  ["Function", "Model", "ptc regulation", "alphaHh_by" ,"x_reach", "t_reach"]
data = pd.DataFrame(columns = ["step","Function", "Model", "ptc regulation",
                               "alphaHh_by" ,"x_reach", "t_reach"], 
                    index = range(250*24))
count = 0
t11 = time.time()
for i in range(250):
    print(i)
    byrandom = [byfunction(random.random()), 1, 1, 
            byfunction(random.random()), byfunction(random.random()), 
            byfunction(random.random()), byfunction(random.random()), 
            byfunction(random.random()), byfunction(random.random()), 
            byfunction(random.random()), byfunction(random.random()), 
            byfunction(random.random()), byfunction(random.random()), 
            byfunction(random.random()), byfunction(random.random()), 
            byfunction(random.random())]
    
######################################################################
####################### WITH PTC REGULATION ###########################
###################################################################### 
    
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
    ###################  ["Function", "Model", "ptc regulation", "alphaHh_by" ,"x_reach", "t_reach"]
    data.iloc[count] = [i,"Hh", "Overshoot", "Yes", 1, x_reach_Hh_over, t_reach_Signal_over]
    count = count + 1 
    data.iloc[count] = [i,"Signal", "Overshoot", "Yes", 1, x_reach_Signal_over, t_reach_Signal_over]
    count = count + 1
    
    
    # steady-state solution
    sol_SS = solve_SS(par, x_plot)
    Hh_SS = sol_SS[1]
    x_reach_Hh_SS = xd(Hh_SS[-1]/10, Hh_SS)
    Signal_SS = sol_SS[2]
    x_reach_Signal_SS = sol_SS[3]
    
    data.iloc[count] = [i,"Hh", "SS", "Yes", "1", x_reach_Hh_SS, False]
    count = count + 1 
    data.iloc[count] = [i,"Signal", "SS", "Yes", 1, x_reach_Signal_SS, False]
    count = count + 1
#################### by up ##################################
    par_up = par*by_up
    #transient solution
    sol_tr_up = solve_over(par_up, x_plot)
    Hh_tr_up = sol_tr_up[0][:,0:101]
    Hh_over_up = x_over_umbral(Hh_tr_up, 10)
    x_reach_Hh_over_up = Hh_over_up[0]
    t_reach_Hh_over_up = Hh_over_up[1]
    Signal_tr_up = sol_tr_up[1]
    x_reach_Signal_over_up = sol_tr_up[2]
    t_reach_Signal_over_up = sol_tr_up[3]
    
    data.iloc[count] = [i,"Hh", "Overshoot", "Yes", str(up), x_reach_Hh_over_up, t_reach_Hh_over_up]
    count = count + 1 
    data.iloc[count] = [i,"Signal", "Overshoot", "Yes", str(up), x_reach_Signal_over_up, t_reach_Signal_over_up]
    count = count + 1
    
    #steady-state solution
    sol_SS_up = solve_SS(par_up, x_plot)
    Hh_SS_up = sol_SS_up[1]
    x_reach_Hh_SS_up = xd(Hh_SS_up[-1]/10, Hh_SS_up)
    Signal_SS_up = sol_SS_up[2]
    x_reach_Signal_SS_up = sol_SS_up[3]
    
    data.iloc[count] = [i,"Hh", "SS", "Yes", str(up), x_reach_Hh_SS_up, False]
    count = count + 1 
    data.iloc[count] = [i,"Signal", "SS", "Yes", str(up), x_reach_Signal_SS_up, False]
    count = count + 1
################### by down #################################
    par_down = par*by_down
    #transient solution
    sol_tr_down = solve_over(par_down, x_plot)
    Hh_tr_down = sol_tr_down[0][:,0:101]
    Hh_over_down = x_over_umbral(Hh_tr_down, 10)
    x_reach_Hh_over_down = Hh_over_down[0]
    t_reach_Hh_over_down = Hh_over_down[1]
    Signal_tr_down = sol_tr_down[1]
    x_reach_Signal_over_down = sol_tr_down[2]
    t_reach_Signal_over_down = sol_tr_down[3]
    
    data.iloc[count] = [i,"Hh", "Overshoot", "Yes", str(down), x_reach_Hh_over_down, t_reach_Hh_over_down]
    count = count + 1 
    data.iloc[count] = [i,"Signal", "Overshoot", "Yes", str(down), x_reach_Signal_over_down, t_reach_Signal_over_down]
    count = count + 1
    
    #steady-state solution
    sol_SS_down = solve_SS(par_down, x_plot)
    Hh_SS_down = sol_SS_down[1]
    x_reach_Hh_SS_down = xd(Hh_SS_down[-1]/10, Hh_SS_down)
    Signal_SS_down = sol_SS_down[2]
    x_reach_Signal_SS_down = sol_SS_down[3]
    
    data.iloc[count] = [i,"Hh", "SS", "Yes", str(down), x_reach_Hh_SS_down, False]
    count = count + 1 
    data.iloc[count] = [i,"Signal", "SS", "Yes", str(down), x_reach_Signal_SS_down, False]
    count = count + 1
  

    
######################################################################
####################### WITHOUT PTC REGULATION, alpha_ptc = 0 ###########################
######################################################################
    
################### WT ####################################
    par = params["value"]*byrandom*np.array([1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1])
    #transient solution
    sol_tr = solve_over(par, x_plot)
    Hh_tr = sol_tr[0][:,0:101]
    Hh_over = x_over_umbral(Hh_tr, 10)
    x_reach_Hh_over = Hh_over[0]
    t_reach_Hh_over = Hh_over[1]
    Signal_tr = sol_tr[1]
    x_reach_Signal_over = sol_tr[2]
    t_reach_Signal_over = sol_tr[3]
    ###################  ["Function", "Model", "ptc regulation", "alphaHh_by" ,"x_reach", "t_reach"]
    data.iloc[count] = [i,"Hh", "Overshoot", "No", 1, x_reach_Hh_over, t_reach_Signal_over]
    count = count + 1 
    data.iloc[count] = [i,"Signal", "Overshoot", "No", 1, x_reach_Signal_over, t_reach_Signal_over]
    count = count + 1
    
    
    #steady-state solution
    sol_SS = solve_SS(par, x_plot)
    Hh_SS = sol_SS[1]
    x_reach_Hh_SS = xd(Hh_SS[-1]/10, Hh_SS)
    Signal_SS = sol_SS[2]
    x_reach_Signal_SS = sol_SS[3]
    
    data.iloc[count] = [i,"Hh", "SS", "No", "1", x_reach_Hh_SS, False]
    count = count + 1 
    data.iloc[count] = [i,"Signal", "SS", "No", 1, x_reach_Signal_SS, False]
    count = count + 1
#################### by up ##################################
    par_up = par*by_up
    #transient solution
    sol_tr_up = solve_over(par_up, x_plot)
    Hh_tr_up = sol_tr_up[0][:,0:101]
    Hh_over_up = x_over_umbral(Hh_tr_up, 10)
    x_reach_Hh_over_up = Hh_over_up[0]
    t_reach_Hh_over_up = Hh_over_up[1]
    Signal_tr_up = sol_tr_up[1]
    x_reach_Signal_over_up = sol_tr_up[2]
    t_reach_Signal_over_up = sol_tr_up[3]
    
    data.iloc[count] = [i,"Hh", "Overshoot", "No", str(up), x_reach_Hh_over_up, t_reach_Hh_over_up]
    count = count + 1 
    data.iloc[count] = [i,"Signal", "Overshoot", "No", str(up), x_reach_Signal_over_up, t_reach_Signal_over_up]
    count = count + 1
    
    #steady-state solution
    sol_SS_up = solve_SS(par_up, x_plot)
    Hh_SS_up = sol_SS_up[1]
    x_reach_Hh_SS_up = xd(Hh_SS_up[-1]/10, Hh_SS_up)
    Signal_SS_up = sol_SS_up[2]
    x_reach_Signal_SS_up = sol_SS_up[3]
    
    data.iloc[count] = [i,"Hh", "SS", "No", str(up), x_reach_Hh_SS_up, False]
    count = count + 1 
    data.iloc[count] = [i,"Signal", "SS", "No", str(up), x_reach_Signal_SS_up, False]
    count = count + 1
################### by down #################################
    par_down = par*by_down
    #transient solution
    sol_tr_down = solve_over(par_down, x_plot)
    Hh_tr_down = sol_tr_down[0][:,0:101]
    Hh_over_down = x_over_umbral(Hh_tr_down, 10)
    x_reach_Hh_over_down = Hh_over_down[0]
    t_reach_Hh_over_down = Hh_over_down[1]
    Signal_tr_down = sol_tr_down[1]
    x_reach_Signal_over_down = sol_tr_down[2]
    t_reach_Signal_over_down = sol_tr_down[3]
    
    data.iloc[count] = [i,"Hh", "Overshoot", "No", str(down), x_reach_Hh_over_down, t_reach_Hh_over_down]
    count = count + 1 
    data.iloc[count] = [i,"Signal", "Overshoot", "No", str(down), x_reach_Signal_over_down, t_reach_Signal_over_down]
    count = count + 1
    
    #steady-state solution
    sol_SS_down = solve_SS(par_down, x_plot)
    Hh_SS_down = sol_SS_down[1]
    x_reach_Hh_SS_down = xd(Hh_SS_down[-1]/10, Hh_SS_down)
    Signal_SS_down = sol_SS_down[2]
    x_reach_Signal_SS_down = sol_SS_down[3]
    
    data.iloc[count] = [i,"Hh", "SS", "No", str(down), x_reach_Hh_SS_down, False]
    count = count + 1 
    data.iloc[count] = [i,"Signal", "SS", "No", str(down), x_reach_Signal_SS_down, False]
    count = count + 1
    
    print("--------------------------------------------------")
    
    
data.to_csv("variation_all_parameter_alphaHh_alpha_ptc.csv")

t22 = time.time()
print("tiempo total", t22-t11)
