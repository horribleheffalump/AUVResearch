
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#filename = "D:\\projects.git\\NavigationResearch\\results\\results_20.txt"
#filename = "D:\\projects.git\\NavigationResearch\\results\\results_40.txt"
#filename = "D:\\projects.git\\NavigationResearch\\results\\results_80.txt"
filename = "D:\\projects.git\\NavigationResearch\\results\\results_159.txt"
data = pd.read_csv(filename, delimiter = " ", usecols= [5,6,7], header=None, dtype=float)
print(data.mean())
print(data.std())

#5   -0.972040
#6    0.857192
#7    0.013334
#dtype: float64
#print(data.std())
#5    0.185896
#6    0.464195
#7    0.028598

#5   -0.987751
#6    2.523302
#7    0.071505
#dtype: float64
#print(data.std())
#5    0.268394
#6    0.654990
#7    0.042049

#5   -1.677777
#6    3.057360
#7    0.119461
#dtype: float64
#print(data.std())
#5    0.430546
#6    0.911969
#7    0.056614

#5    2.836884
#6    3.612389
#7    0.147340
#dtype: float64
#print(data.std())
#5    0.536441
#6    1.084020
#7    0.159335