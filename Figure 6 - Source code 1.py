# Figure 6 - Source code 1
# code to generate  figure 6

# All uploaded or cited files are in the same repository 
# with the same name as they are cited

############# figure 6A-B ############################
# Width of patterns were measures following the process of figure 5
# data was saved in the file: data_ptcTP18-25-29C_col_widths.csv

############# figure 6C ############################

#load data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data_ptcTPT = pd.read_csv(r"C:\Users\rosalio\Documents\2023\Hh_robustness\data_ptcTP18-25-29C_col_widths.csv")

col2529 = data_ptcTPT[data_ptcTPT["temperature"] == "25-29"]
col2529C = col2529["width"]

col18 = data_ptcTPT[data_ptcTPT["temperature"] == "18"]
col18C = col18["width"]

q05_col2529C = np.quantile(col2529C, 0.5)
q05_col18C = np.quantile(col18C, 0.5)


bp = plt.boxplot([col18C, col2529C], patch_artist=True, showfliers = False)

for element in ['boxes', 'whiskers', 'fliers', 'means', 'caps']:
    plt.setp(bp[element], color="green")
plt.setp(bp["medians"], color="black")

for patch in bp['boxes']:
    patch.set(facecolor="green")     
    

#diferencias en quartiles
deltaColT = q05_col18C - q05_col2529C
plt.plot([2.5,2.5], [q05_col2529C, q05_col2529C + deltaColT], c = "k")
plt.plot([2.3, 2.5], [q05_col2529C, q05_col2529C], c = "k")
plt.plot([2.3, 2.5], [q05_col18C, q05_col18C], c = "k")
plt.text(2.55, q05_col2529C + 2, str(round(deltaColT,2)))


#normality test
from scipy.stats import shapiro
estadistico_col2529, p_value_col2529 = shapiro(col2529C)
print("Statistical col2529 = ", estadistico_col2529,  "p_value_col2529 = ", p_value_col2529)
if(p_value_col2529> 0.05):
    print("The values of col2529 do not have a normal distribution")
else:
    print("The values of col2529 do have a normal distribution")



estadistico_col18, p_value_col18 = shapiro(col18C)
print("Statistical col18 = ", estadistico_col18,  "p_value_col18 = ", p_value_col18)
if(p_value_col18> 0.05):
    print("The values of col18 do not have a normal distribution")
else:
    print("The values of col18 do have a normal distribution")
    

# Since the col2529 data does not have a normal distribution, a Mann-Whitney  U test
from scipy.stats import mannwhitneyu
st_col, pvalue_col = mannwhitneyu(col18C, col2529C)
print("p-value = ", pvalue_col)


plt.xticks([1,2], ["18", "$\geq$25"], fontsize = 14)
plt.xlabel("Temperature (°C)", fontsize = 14)
plt.ylabel("Pattern width ($\mu m$)", fontsize = 14)
plt.xlim(0.5, 2.8)

# statistical lines 
plt.plot([1,2], [26.5,26.5], c = "k", linewidth = 0.8)
plt.plot([1,1], [25.8, 26.5], c = "k", linewidth = 0.8)
plt.plot([2,2], [19.5, 26.5], c = "k", linewidth = 0.8)
plt.text(1.3, 26.7, "0.002" )
plt.ylim(15,28)
plt.show()