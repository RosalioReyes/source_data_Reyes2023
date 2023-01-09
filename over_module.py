# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 09:32:01 2021

@author: rosalio
"""


import numpy as np
from scipy.integrate import odeint


def solve_over(params, x_plot):

    ################## intervalo x donde se resuelve la ecuacion ###########################
    x = np.array([i for i in range(-100,101)])
    dx = x[1] - x[0] 
    
    ##################### condiciones iniciales #####################################
    Hh0 = np.array([0. for i in range(-100,101)])
    
    ptc0 = np.zeros(201)
    for i in range(0,101):
        ptc0[i] = float(params[3])/float(params[6])
    for i in range(101, 201):
        ptc0[i] = 0.0
        
    Ptc0 = np.zeros(201)
    for i in range(0,101):
        Ptc0[i] = float(params[3])*float(params[11])/float(params[6])*float(params[7])
    for i in range(101, 201):
        Ptc0[i] = 0.0 
        
    
    Hh_Ptc0 = np.array([0. for i in range(-100,101)])
    signal0 = np.array([0. for i in range(-100,101)])
    
    w0 = np.concatenate((Hh0, ptc0, Ptc0, Hh_Ptc0, signal0), axis = None)
    
    ########################## Tiempos #########################################
    T = 8*3600 
    dt = 0.5  
    N_t = int(round(float(T)/dt))
    t = np.linspace(0, N_t*dt, N_t + 1)
    
    ######################## ODE solver parameters ##############################
    abserr = 1.0e-8
    relerr = 1.0e-6
    
    ######################### the vector field ##################################
    def vectorfield(w, t, parametros):
        """ Defines the coupled differential equations with finite difference
          w = [H0, H1, ..., H200, ptc0, ptc1,...ptc200, ... signal0, ... signal 200]
          t = time interval
          p = vector of parameters """
        ## w[0] = H0, w[1] = H1, ..., w[200] = H200
        ## w[201] = ptc0, w[202] = ptc1, ...w[401] = ptc200
        ## w[402] = Ptc0, w[403] = Ptc1, ... w[602] = Ptc200
        ## w[603] = Hh_Ptc0, w[604] = Hh_Ptc1, ... w[803] = Hh_Ptc200
        ## w[804] = signal0, w[805] = signal2, ... w[1004] = signal200
        
        D, alpha_Hh, alpha_ptc, alpha_ptc0, alpha_signal, beta_Hh, beta_ptc, beta_Ptc, beta_signal, beta_Hh_Ptc, gamma_Hh_Ptc, T_Ptc, k_ptc, k_signal, m, n = parametros
        
        H = []
        p = []
        P = []
        H_P = []
        signal = []
        
        for i in range(0,201):
            H.append(w[i])
        for i in range(201, 402):
            p.append(w[i])
        for i in range(402, 603):
            P.append(w[i])
        for i in range(603,804):
            H_P.append(w[i])
        for i in range(804, 1005):
            signal.append(w[i])
            
        def dHhdt(Hh, ptc, Ptc, Hh_Ptc, signal):
            N = len(Hh) - 1  
            u = np.zeros(N + 1)
            u[0] = (D/dx**2)*(2*Hh[1] - 2*Hh[0]) + S_p(x[0])*alpha_Hh - gamma_Hh_Ptc*Hh[0]*Ptc[0] - beta_Hh*Hh[0]
            for i in range(1,N):
                u[i] = (D/dx**2)*(Hh[i+1] - 2*Hh[i] + Hh[i-1]) + S_p(x[i])*alpha_Hh - gamma_Hh_Ptc*Hh[i]*Ptc[i] - beta_Hh*Hh[i]
            u[N] = (D/dx**2)*(2*Hh[N - 1] - 2*Hh[N]) + S_p(x[N])*alpha_Hh - gamma_Hh_Ptc*Hh[N]*Ptc[N] - beta_Hh*Hh[N]
            return u
    
        #funcion para ptc
        def dptcdt(Hh, ptc, Ptc, Hh_Ptc, signal):
            N = len(ptc) - 1
            u = np.zeros(N + 1)
            for i in range(N+1):
                u[i] = S_m(x[i])*alpha_ptc0  + (alpha_ptc * signal[i]**m)/(k_ptc**m + signal[i]**m) - beta_ptc*ptc[i]
            return u
    
        # funcion para Ptc
        def dPtcdt(Hh, ptc, Ptc, Hh_Ptc, signal):
            N = len(Ptc) - 1
            u = np.zeros(N + 1)
            for i in range(N+1):
                u[i] = T_Ptc * ptc[i] - gamma_Hh_Ptc * Hh[i]*Ptc[i] - beta_Ptc*Ptc[i]
            return u
    
        # funcion para Hh_Ptc
        def dHh_Ptcdt(Hh, ptc, Ptc, Hh_Ptc, signal):
            N = len(Hh_Ptc) - 1
            u = np.zeros(N + 1)
            for i in range(N+1):
                u[i] = gamma_Hh_Ptc*Hh[i]*Ptc[i] - beta_Hh_Ptc*Hh_Ptc[i]
            return u
    
        # funcion para signal 
        def dsignaldt(Hh, ptc, Ptc, Hh_Ptc, signal):
            N = len(signal) - 1
            u = np.zeros(N + 1)
            for i in range(N+1):
                if (S_m(x[i]) == 0 and  Ptc[i] == 0):
                    u[i] = - beta_signal*signal[i]
                else:
                    u[i] = (S_m(x[i])*alpha_signal*(Hh_Ptc[i]/Ptc[i])**n)/(k_signal**n + (Hh_Ptc[i]/Ptc[i])**n) - beta_signal*signal[i]
            return u    
    
    
        # funcion S+
        def S_p(x):
            if x>0:
                sp = 1
            else:
                sp = 0
            return sp
    
        # funcion S-
        def S_m(x):
            if x<0 :
                sm = 1
            else:
                sm = 0
            return sm
        
          
        # f = [eq1, eq2, ..., eq1004]
        f = np.concatenate((dHhdt(H, p, P, H_P, signal), dptcdt(H, p, P, H_P, signal), dPtcdt(H, p, P, H_P, signal), dHh_Ptcdt(H, p, P, H_P, signal), dsignaldt(H, p, P, H_P, signal)), axis = None)
        return f
    
    
    ##############E reach function ###############################
        
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
            r = xd(np.amax(sig[i, :])/5, sig[i])
            reachs.append([r,i])
        reachs.sort()
        return reachs[-1]
    ##################### the solution ###############################
    par = params
    wsol = odeint(vectorfield, w0, t, args = (par,), atol = abserr, rtol = relerr)
    sig = wsol[:, 804:905]
    over = x_over(sig)
    x_over = over[0]
    t_over = over[1]
    return wsol, sig, x_over, t_over

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

