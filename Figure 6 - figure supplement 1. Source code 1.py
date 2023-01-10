# Figure 6 - figure supplement 1. Source code 1
# code to generate  Figure 2 - figure supplement 1

# All uploaded or cited files are in the same repository 
# with the same name as they are cited


# The width of the patterns was measured using the same functions that were 
# used in Figures 5 and 6. These functions are in the 
# code_for_analyze_pattern_widths.py code

#raw data was saved in csv format in the files:
# ptc182529C.csv for ptc widths at differents temperatures
# dpp182529C.csv for dpp widths at differents temperatures

############# ptc ##########################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data_ptc = pd.read_csv(r"C:\Users\rosalio\Documents\2023\Hh_robustness\ptc182529C.csv")

ptc18C_Width = data_ptc[data_ptc["temperature"] == 18]["width"]
ptc25C_Width = data_ptc[data_ptc["temperature"] == 25]["width"]
ptc29C_Width = data_ptc[data_ptc["temperature"] == 29]["width"]
        
bp = plt.boxplot([ptc18C_Width, ptc25C_Width, ptc29C_Width], patch_artist=True, showfliers = False)


for element in ['boxes', 'whiskers', 'fliers', 'means', 'caps']:
    plt.setp(bp[element], color="tab:cyan")
plt.setp(bp["medians"], color="black")

for patch in bp['boxes']:
    patch.set(facecolor="tab:cyan")     

plt.xticks([1,2,3], ["18", "25", "29"])
plt.xlabel("Temperature (°C)")
plt.ylabel("Width ($\mu m$)")

from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.stats import shapiro

def getWhisker(array):
    q3 = np.quantile(array, 0.75)
    q1 = np.quantile(array, 0.25)
    iqr = q3 - q1
    upper_whisker = array[array <= q3 + 1.5*iqr].max()
    lower_whisker = array[array >= q1 - 1.5*iqr].min()
    return upper_whisker, lower_whisker




def drawSignificance2(array1, array2, position_array1, position_array2, begin_line1, begin_line2, h, equalVar):
    w1 = begin_line1
    w2 = begin_line2
    wmax = np.max([w1,w2])
    h = h + wmax
    # draw lines
    x1 = position_array1 - 0.2
    x2 = position_array2 + 0.2
    plt.plot([x1, x2], [h,h], c = "k")
    plt.plot([x1, x1], [w1, h], c = "k")
    plt.plot([x2, x2], [w2, h], c = "k")
    
    # make proof
    # normality proof
    std_array1, pval_array1 = shapiro(array1)
    print("array1-statistic = ", std_array1,  "p_value_array1 = ", pval_array1)
    if(pval_array1 < 0.05):
        print("The values of array1 do not have a normal distribution")
        v1 = 1
    else:
        print("The values of array1 do have a normal distribution")
        v1 = 0
        
    std_array2, pval_array2 = shapiro(array2)
    print("array2-statistic = ", std_array2,  "p_value_array2 = ", pval_array2)
    if(pval_array2 < 0.05):
        print("The values of array2 do not have a normal distribution")
        v2 = 1
    else:
        print("The values of array2 do have a normal distribution")
        v2 = 0 
        
   # if both have normal distribution (v1 + v2 == 2) --> ttest
     # if any of them have no normal distribution --> mannwhitneyu parametric test
    if (v1 + v2 == 2):
        st, p = ttest_ind(a = array1, b = array2, equal_var=equalVar)
    else:
        st, p = mannwhitneyu(array1, array2)
    print("statistic = ",  st, "p=", p)
    
    # we write statistical significances
    if p >= 0.0001:
        plt.text(x1, h+0.5, "p=" + str(round(p,4)))
    else:
        pST = "{:.2e}".format(p)
        plt.text(x1, h+0.5, "p=" + str(pST))
    return


drawSignificance2(ptc18C_Width, ptc25C_Width, 1.2,1.75,18,15.5,2.5, False) # 1 y 2
drawSignificance2(ptc25C_Width, ptc29C_Width, 2.25,2.8, 15.5, 16.5, 3.9, False) # 2 y 3
drawSignificance2(ptc18C_Width, ptc29C_Width, 1.2,2.8,22.5, 22.5, 2, False)  # 1 y 3
plt.ylim(7.5, 27)
plt.show()


############# dpp ##########################################
data_dpp = pd.read_csv(r"C:\Users\rosalio\Documents\2023\Hh_robustness\dpp182529C.csv")

dpp18C_Width = data_dpp[data_dpp["temperature"] == 18]["width"]
dpp25C_Width = data_dpp[data_dpp["temperature"] == 25]["width"]
dpp29C_Width = data_dpp[data_dpp["temperature"] == 29]["width"]
        
bp = plt.boxplot([dpp18C_Width, dpp25C_Width, dpp29C_Width], patch_artist=True, showfliers = False)


for element in ['boxes', 'whiskers', 'fliers', 'means', 'caps']:
    plt.setp(bp[element], color="red")
plt.setp(bp["medians"], color="black")

for patch in bp['boxes']:
    patch.set(facecolor="red")     

plt.xticks([1,2,3], ["18", "25", "29"])
plt.xlabel("Temperature (°C)")
plt.ylabel("Width ($\mu m$)")


drawSignificance2(dpp18C_Width, dpp25C_Width, 1.2,1.75,26,27.5,5.5, False) # 1 y 2
drawSignificance2(dpp25C_Width, dpp29C_Width, 2.25,2.8, 27.5, 32.5, 0.5, False) # 2 y 3
drawSignificance2(dpp18C_Width, dpp29C_Width, 1.2,2.8,36, 36, 2, False)  # 1 y 3
plt.ylim(15, 40)
plt.show()
