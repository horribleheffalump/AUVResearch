import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

t_history = np.array(range(0,361))

filename = "D:\\projects.git\\NavigationResearch\\results\\slope_known\\results_speed_0.txt"
data = pd.read_csv(filename, delimiter = ",", header=None, dtype=float)
Vx_known_mean = data.mean()
Vx_known_std = data.std()

filename = "D:\\projects.git\\NavigationResearch\\results\\slope_known\\results_speed_1.txt"
data = pd.read_csv(filename, delimiter = ",", header=None, dtype=float)
Vy_known_mean = data.mean()
Vy_known_std = data.std()

filename = "D:\\projects.git\\NavigationResearch\\results\\slope_known\\results_speed_2.txt"
data = pd.read_csv(filename, delimiter = ",", header=None, dtype=float)
Vz_known_mean = data.mean()
Vz_known_std = data.std()

filename = "D:\\projects.git\\NavigationResearch\\results\\slope_unknown\\results_speed_0.txt"
data = pd.read_csv(filename, delimiter = ",", header=None, dtype=float)
Vx_unknown_mean = data.mean()
Vx_unknown_std = data.std()

filename = "D:\\projects.git\\NavigationResearch\\results\\slope_unknown\\results_speed_1.txt"
data = pd.read_csv(filename, delimiter = ",", header=None, dtype=float)
Vy_unknown_mean = data.mean()
Vy_unknown_std = data.std()

filename = "D:\\projects.git\\NavigationResearch\\results\\slope_unknown\\results_speed_2.txt"
data = pd.read_csv(filename, delimiter = ",", header=None, dtype=float)
Vz_unknown_mean = data.mean()
Vz_unknown_std = data.std()

f = plt.figure(num=None, figsize=(15,5), dpi=200, facecolor='w', edgecolor='k')
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])     
gs.update(left=0.03, bottom=0.05, right=0.99, top=0.99, wspace=0.15, hspace=0.1)    

ax = plt.subplot(gs[0])
ax.plot(t_history[1:], Vx_known_mean[1:], color='black', linewidth=2.0)
ax.plot(t_history[1:], Vx_known_std[1:], color='black', linewidth=2.0, linestyle=':')

ax = plt.subplot(gs[1])
ax.plot(t_history[1:], Vy_known_mean[1:], color='black', linewidth=2.0)
ax.plot(t_history[1:], Vy_known_std[1:], color='black', linewidth=2.0, linestyle=':')

ax = plt.subplot(gs[2])
ax.plot(t_history[1:], Vz_known_mean[1:], color='black', linewidth=2.0)
ax.plot(t_history[1:], Vz_known_std[1:], color='black', linewidth=2.0, linestyle=':')

f.savefig('D:\\projects.git\\NavigationResearch\\results\\slope_known\\error_known_stats.pdf')


f = plt.figure(num=None, figsize=(15,5), dpi=200, facecolor='w', edgecolor='k')
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])     
gs.update(left=0.03, bottom=0.05, right=0.99, top=0.99, wspace=0.15, hspace=0.1)    

ax = plt.subplot(gs[0])
ax.plot(t_history[1:], Vx_unknown_mean[1:], color='red', linewidth=2.0)
ax.plot(t_history[1:], Vx_unknown_std[1:], color='red', linewidth=2.0, linestyle=':')

ax = plt.subplot(gs[1])
ax.plot(t_history[1:], Vy_unknown_mean[1:], color='red', linewidth=2.0)
ax.plot(t_history[1:], Vy_unknown_std[1:], color='red', linewidth=2.0, linestyle=':')

ax = plt.subplot(gs[2])
ax.plot(t_history[1:], Vz_unknown_mean[1:], color='red', linewidth=2.0)
ax.plot(t_history[1:], Vz_unknown_std[1:], color='red', linewidth=2.0, linestyle=':')

f.savefig('D:\\projects.git\\NavigationResearch\\results\\slope_unknown\\error_unknown_stats.pdf')



#plt.show()


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