# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 10:38:25 2023

@author: Rosalío Reyes

función para analizar el ancho de los patrones de la imagen
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm
from scipy.optimize import curve_fit
import glob
import pandas as pd
import seaborn as sns
from scipy import stats


def analisis_img(img):
    #La parte donde se selecciona el punto de inicio
    cp_img = img.copy()
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append([x,y])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(cp_img, ".", (x,y), font, 1, (255,250,0), 2)
            cv2.imshow("ancho de banda", cp_img)
    pts = []
    cv2.imshow("ancho de banda", cp_img)
    cv2.setMouseCallback("ancho de banda", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # plot del recuadro seleccionado
    ancho_y = 30
    ancho_x = 180
    #esquina inferior izquierda
    x0 = pts[0][0]-0.3*ancho_x
    y0 = pts[0][1] - 30
    #esquina superior derecha
    x1 = x0 + ancho_x
    y1 = y0 - ancho_y
    
    inter = [i for i in range(900)]
    plt.imshow(img)
    plt.plot(inter, [y1 for i in range(900)], c = "g")
    plt.plot(inter, [y0 for i in range(900)], c = "r") ## y
    plt.plot([x0  for i in range(900)], inter, c = "skyblue")
    plt.plot([x1 for i in range(900)], inter, c = "k") ## x
    plt.show()

    #promedio sobre el recuadro
    ancho_xx = abs(int(x1) - int(x0))
    x00 = int(x0)
    x11 = int(x1)
    suma = np.zeros(ancho_xx)
    for i in range(ancho_y):
        suma = suma + img[y0-i, x00:x11]
    promedio = suma/ancho_y
    plt.plot([i for i in range(ancho_xx)], promedio/np.amax(promedio))
    plt.show()

    return promedio/np.amax(promedio)


def width_pattern(array, threshold):
    n = len(array)
    x = np.array([i for i in range(n)])
    x_inter = np.linspace(0, n-1, (n-1)*200)
    r_interp = np.interp(x_inter, x, array)
    x_max = np.where(array==np.amax(array))[0][0]
    resta = np.abs(r_interp-threshold*np.ones(r_interp.shape))
    pts_intersect = np.where(resta<0.001)
    
    intersections = x_inter[pts_intersect[0]]
    if len(intersections) == 0 :
        width_pattern = False
    else:
        first = np.where(intersections<x_max)
        position_before = first[0][-1]
        first_pto = intersections[position_before]
        if position_before + 1 == len(intersections):
            width_pattern = False
        else:
            second_pto = intersections[position_before+1]
            width_pattern = abs(second_pto - first_pto)
            plt.plot(x_inter, r_interp)
            plt.plot([first_pto, second_pto], [threshold, threshold])
            plt.show()
    return width_pattern