# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 09:22:55 2021

@author: rosalio
"""

import numpy as np
from scipy.integrate import solve_bvp


# Steady state

def solve_SS(params, x_plot):
    
    ################ functions ################################
    # the SS-equation for Hh
    def Hh_equation(x,sol):
        Hh_prime = sol[1]  #sol[1] es u
        Hh = sol[0]   # sol[0] es Hh
        u_prime = -1*S_p(x)*alpha_Hh/D + ((chi/D)*S_m(x)*Hh/(gamma_Hh_Ptc*Hh + beta_Ptc))*(alpha_ptc0 + alpha_ptc*pow(Hh, n*m)/(pow(eta,m)*pow(k**n + Hh**n,m) + S_m(x)*pow(Hh, n*m))) + beta_Hh*Hh/D
        return np.vstack((Hh_prime, u_prime))
    
    # the ptc function
    def ptc(x, Hh):
        patch = (alpha_ptc0/beta_ptc)*S_m(x) + (alpha_ptc/beta_ptc)*(S_m(x)*pow(Hh, n*m))/(pow(eta,m)*pow(k**n + Hh**n,m) + S_m(x)*pow(Hh, n*m))
        return patch
    
    #the Ptc function
    
    def Ptc(x,Hh):
        Patch = (T_Ptc/(gamma_Hh_Ptc*Hh + beta_Ptc))*(S_m(x)/beta_ptc)*(alpha_ptc0 + (alpha_ptc*pow(Hh, n*m))/(pow(eta,m)*pow(k**n + Hh**n,m) + S_m(x)*pow(Hh, n*m)))
        return Patch
    
    # the Hh_Ptc function 
    
    def Hh_Ptc(x, Hh):
        result = (gamma_Hh_Ptc/beta_Hh_Ptc)*Hh*Ptc(x, Hh)
        return result
    
    # the signal function
    def signal(x,Hh):
        sig = S_m(x)*(alpha_signal/beta_signal)*pow(Hh, n)/(pow(k,n) + pow(Hh,n))
        return sig
    
    # the S+(x) function (S_p -> S plos)
    def S_p(x):
        xx = np.zeros(len(x))
        for i in range(len(x)):
            if x[i]>0:
                sp = 1
            else:
                sp = 0
            xx[i] = sp
        return xx
    
    # the S-(x) function (S_m -> S minus)
    def S_m(x):
        xx = np.zeros(len(x))
        for i in range(len(x)):
            if x[i]<0:
                sp = 1
            else:
                sp = 0
            xx[i] = sp
        return xx
    
    #################### the boundary condition function #######################
    def bc(left_bc, right_bc):
        return np.array([left_bc[1], right_bc[1]])
    
    ###################### functions xd ##################################
    def xd(thresh, array):  #el array debe ser signal (recortado de wsol)
        dif = np.abs(array - thresh*np.ones(len(array)))
        closee = np.where(dif == np.amin(dif))
        if len(closee[0]) == 0:
            result = float("nan") 
        else:
            close = closee[0][-1]
            if close + 1 == 101:
                m = (array[close] - array[close - 1])/2
                result = (thresh - array[close])/m + close
            else:
                m = (array[close + 1] - array[close - 1])/2
                result = (thresh - array[close])/m + close
        return abs(result - 100)
    
    def x_over(sig):
        reachs = []
        for i in range(len(sig)):
            r = xd(np.amax(sig[i, :])/2, sig[i])
            reachs.append([r,i])
        reachs.sort()
        return reachs[-1]
    
    ###################### intervals ###################################
    # interval for solve equation
    x_solve = np.linspace(-100,100,50)
    
    #interval for plot (interpolate) solution
    #x_plot = np.linspace(-100,0,101)
    
    # the ansatz (or guess) solution
    y_ansatz = np.zeros((2,x_solve.size))
    
        
    
    ############### paramters ##################################
    
    D ,alpha_Hh, alpha_ptc, alpha_ptc0, alpha_signal, beta_Hh, beta_ptc, beta_Ptc, beta_signal, beta_Hh_Ptc, gamma_Hh_Ptc , T_Ptc, k_ptc, k_signal, m, n = params
    chi = T_Ptc*gamma_Hh_Ptc/beta_ptc
    k = k_signal*beta_Hh_Ptc/gamma_Hh_Ptc
    eta = k_ptc*beta_signal/alpha_signal
    
    SS_solution = solve_bvp(Hh_equation, bc, x_solve, y_ansatz, tol = 1e-6)
    Hh_SS = SS_solution.sol(x_plot)[0]
    signal_SS = signal(x_plot, Hh_SS)
    x_SS = xd(np.amax(signal_SS)/2, signal_SS)

    return SS_solution, Hh_SS, signal_SS, x_SS
