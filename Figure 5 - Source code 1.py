# Figure 5 - Source code 1
# code to generate  figure 5

# All uploaded or cited files are in the same repository 
# with the same name as they are cited

############# figure 5A-D ############################

# First I analyze the images in ImageJ. To do this, I average a region outside
# of the studied pattern, and subtract the intensity average from the entire 
# image, to remove the background. Then I export the image in png without noise 
# and load it in python. Then, in python I select with the cursor a point that 
# will determine the region of the pattern to study. This point is used to determine
# a region of size 30 pixels by 180 pixels. Finally, average the pattern along 
# the y-axis (in the 30-pixel width direction).

#I do these steps using the analisis_img function found in
#  the code_for_analyze_pattern_widths.py code

# Once I get the pattern averaged over the selected box, I measure the width 
# of the pattern using the function width_pattern  in the code
#  code_for_analyze_pattern_widths.py file.

# The data obtained in this way was saved to the file
# data13_ago_2022_Hhpaper_col_dpp_hhx1_hhx2.csv.

############# figure 5E############################
import pandas as pd

# load the data
dataHh = pd.read_csv(r"C:\Users\rosalio\Documents\2022\Hh-robustness\data13_ago_2022_Hhpaper_col_dpp_hhx1_hhx2.csv")

import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import numpy as np

data_dpp = dataHh[dataHh["target-gene"] == "dpp"]
data_col = dataHh[dataHh["target-gene"] == "col"]

data_dppx1 = data_dpp[data_dpp["experiment"] == "hhx1"]
data_dppx2 = data_dpp[data_dpp["experiment"] == "hhx2"]
data_colx1 = data_col[data_col["experiment"] == "hhx1"]
data_colx2 = data_col[data_col["experiment"] == "hhx2"]

def box_plot(data, edge_color, fill_color):
    bp = ax.boxplot(data, patch_artist=True, showfliers = False)
    
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'caps']:
        plt.setp(bp[element], color=edge_color)
    plt.setp(bp["medians"], color="black")

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)       
        
    return bp
    
data1 = [data_dppx2["width"], data_dppx1["width"]]
data2 = [data_colx2["width"], data_colx1["width"]]

q05_colx1 = np.quantile(data_colx1["width"], 0.5)
q05_colx2 = np.quantile(data_colx2["width"], 0.5)
q05_dppx1 = np.quantile(data_dppx1["width"], 0.5)
q05_dppx2 = np.quantile(data_dppx2["width"], 0.5)




mean_dppx2 = data_dppx2["width"].mean()
mean_dppx1 = data_dppx1["width"].mean()
mean_colx2 = data_colx2["width"].mean()
mean_colx1 = data_colx1["width"].mean()



std_dppx2 = data_dppx2["width"].std()
std_dppx1 = data_dppx1["width"].std()
std_colx2 = data_colx2["width"].std()
std_colx1 = data_colx1["width"].std()

#normality test
from scipy.stats import shapiro
estadistico_dppx2, p_value_dppx2 = shapiro(data_dppx2["width"])
print("Estadístico dppx2 = ", estadistico_dppx2,  "p_value_dppx2 = ", p_value_dppx2)
if(p_value_dppx2> 0.05):
    print("Los valores de dppx2 no tienen una distribución normal")
else:
    print("Los valores de dppx2 sí tienen una distribución normal")
    
estadistico_dppx1, p_value_dppx1 = shapiro(data_dppx1["width"])
print("Estadístico dppx1 = ", estadistico_dppx1,  "p_value_dppx1 = ", p_value_dppx1)
if(p_value_dppx1> 0.05):
    print("Los valores de dppx1 no tienen una distribución normal")
else:
    print("Los valores de dppx1 sí tienen una distribución normal")

    
from scipy.stats import shapiro
estadistico_colx2, p_value_colx2 = shapiro(data_colx2["width"])
print("Estadístico colx2 = ", estadistico_colx2,  "p_value_colx2 = ", p_value_colx2)
if(p_value_colx2> 0.05):
    print("Los valores de colx2 no tienen una distribución normal")
else:
    print("Los valores de colx2 sí tienen una distribución normal")
    
estadistico_colx1, p_value_colx1 = shapiro(data_colx1["width"])
print("Estadístico colx1 = ", estadistico_colx1,  "p_value_colx1 = ", p_value_colx1)
if(p_value_colx1> 0.05):
    print("Los valores de colx1 no tienen una distribución normal")
else:
    print("Los valores de colx1 sí tienen una distribución normal")


# we make Mann-Whitney  U test
# because data has not normal distribution (after make shapiro proof)
st_col, pvalue_col = mannwhitneyu(data_colx1["width"], data_colx2["width"])
st_dpp, pvalue_dpp = mannwhitneyu(data_dppx1["width"], data_dppx2["width"])

print("p-value col = ", pvalue_col)
print("p-value dpp = ", pvalue_dpp)

fig, ax = plt.subplots()
bp1 = box_plot(data1, 'red', 'red')
bp2 = box_plot(data2, 'green', 'green')
ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Dpp', 'Col'], )
ax.set_ylim(10, 30)


plt.xticks([1,2], ["Wildtype", "hh${}^{+/-}$"], fontsize = 14)
plt.ylabel("Pattern width ($\mu m$)", fontsize = 14)

# significant differences
#dpp
plt.plot([1,2], [28, 28], c = "k", linewidth = 0.5)
plt.plot([1,1], [28, 27], c = "k", linewidth = 0.5)
plt.plot([2,2], [28, 23.5], c = "k", linewidth = 0.5)

#col
plt.plot([1,2], [mean_colx1 - 2, mean_colx1 - 2], c = "k", linewidth = 0.5)
plt.plot([1,1], [mean_colx1 - 2, mean_colx1 - 0.2], c = "k", linewidth = 0.5)
plt.plot([2,2], [mean_colx1 - 2, mean_colx1 - 1.3], c = "k", linewidth = 0.5)

# dpp
plt.text(1.3, 28.4, "0.0003")
# col
plt.text(1.3, mean_colx1 - 1.7, "0.007")

# difference in medians 
deltaDpp = q05_dppx2 - q05_dppx1
print("differences in medians dpp", deltaDpp)
plt.plot([2.2, 2.2], [q05_dppx1, q05_dppx1 + deltaDpp], c = "k")
plt.plot([2.13, 2.2], [q05_dppx1, q05_dppx1], c = "k")
plt.plot([2.13, 2.2], [q05_dppx1 + deltaDpp, q05_dppx1 + deltaDpp], c = "k")
plt.text(2.22, q05_dppx1 + 2, str(round(deltaDpp,2)))

deltaCol = q05_colx2 - q05_colx1
print("differences in medians col", deltaCol)
plt.plot([2.2,2.2], [q05_colx1, q05_colx1 + deltaCol], c = "k")
plt.plot([2.13, 2.2], [q05_colx1, q05_colx1], c = "k")
plt.plot([2.13, 2.2], [q05_colx1 + deltaCol, q05_colx1 + deltaCol], c = "k")
plt.text(2.22, mean_colx1 + 0.7, str(round(deltaCol,2)))

plt.show()