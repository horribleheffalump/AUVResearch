import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

filters = ['cmnf', 'kalman', 'pseudo']
colors = ['red', 'green', 'blue']

#EstimateErrorFileNameTemplate = "D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\byvirt_4\\estimate\\estimate_error_[filter]_[pathnum].txt"
#ControlErrorFileNameTemplate = "D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\byvirt_4\\control\\control_error_[filter]_[pathnum].txt"
EstimateErrorFileNameTemplate = "D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\acoustic\\estimate\\estimate_error_[filter]_[pathnum].txt"
ControlErrorFileNameTemplate = "D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\acoustic\\control\\control_error_[filter]_[pathnum].txt"

mEstimateError = [None] * len(filters)
stdEstimateError = [None] * len(filters)

for k in range(0, len(filters)):
    mEstimateError[k] = np.loadtxt(EstimateErrorFileNameTemplate.replace('[filter]',filters[k]).replace('[pathnum]', 'mean'))
    stdEstimateError[k] = np.loadtxt(EstimateErrorFileNameTemplate.replace('[filter]',filters[k]).replace('[pathnum]', 'std'))

#for k in range(0, len(filters)):
#    mControlError = np.loadtxt(ControlErrorFileNameTemplate.replace('[filter]',filters[k]).replace('[pathnum]', 'mean'))
#    stdControlError = np.loadtxt(ControlErrorFileNameTemplate.replace('[filter]',filters[k]).replace('[pathnum]', 'std'))

t_history = range(0, mEstimateError[0].shape[0])

f = plt.figure(num=None, figsize=(15,5), dpi=200, facecolor='w', edgecolor='k')
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])     
gs.update(left=0.03, bottom=0.05, right=0.99, top=0.99, wspace=0.1, hspace=0.1)    

for k in [2]:
    for i in range(0,3):
        ax = plt.subplot(gs[i])
        ax.plot(t_history, stdEstimateError[k][:,i], color=colors[k], linewidth=2.0)
        ax.plot(t_history, mEstimateError[k][:,i], color=colors[k], linewidth=1.0, linestyle = 'dotted')
plt.show()

#f = plt.figure(num=None, figsize=(15,5), dpi=200, facecolor='w', edgecolor='k')
#gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])     
#gs.update(left=0.03, bottom=0.05, right=0.99, top=0.99, wspace=0.1, hspace=0.1)    

#for k in range(0,len(filters)):
#    for i in range(0,3):
#        ax = plt.subplot(gs[i])
#        ax.plot(t_history, stdControlError[k][:,i], color=colors[k], linewidth=2.0)
#        ax.plot(t_history, mControlError[k][:,i], color=colors[k], linewidth=1.0, linestyle = 'dotted')
#plt.show()

