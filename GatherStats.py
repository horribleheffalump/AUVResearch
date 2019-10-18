import numpy as np
import glob



filters = ['cmnf', 'kalman', 'pseudo']
colors = ['red', 'green', 'blue']

#EstimateErrorFileNameTemplate = "D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\byvirt_4\\estimate\\estimate_error_[filter]_[pathnum].txt"
#ControlErrorFileNameTemplate = "D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\byvirt_4\\control\\control_error_[filter]_[pathnum].txt"
EstimateErrorFileNameTemplate = "D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\acoustic\\estimate\\estimate_error_[filter]_[pathnum].txt"
ControlErrorFileNameTemplate = "D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\acoustic\\control\\control_error_[filter]_[pathnum].txt"

EstimateError = [None] * len(filters)
ControlError = [None] * len(filters)

for k in range(0, len(filters)):
    print(filters[k])
    m = 0
    files = glob.glob(EstimateErrorFileNameTemplate.replace("[filter]", filters[k]).replace("[pathnum]", "*"))
    for file in files:
        data = np.loadtxt(file)
        if EstimateError[k] is None:
            EstimateError[k] = np.zeros((len(files), data.shape[0], data.shape[1]))
        EstimateError[k][m,:,:] = data
        print(np.max(data, axis = 0))
        m += 1
        if m % 1000 == 0:
            print('done ', m, ' of ', len(files))
    #m = 0
    #files = glob.glob(ControlErrorFileNameTemplate.replace("[filter]", filters[k]).replace("[pathnum]", "*"))
    #for file in files:
    #    data = np.loadtxt(file)
    #    if ControlError[k] is None:
    #        ControlError[k] = np.zeros((len(files), data.shape[0], data.shape[1]))
    #    ControlError[k][m,:,:] = data
    #    m += 1
    
    

mEstimateError = [None] * len(filters)
stdEstimateError = [None] * len(filters)
#mControlError = [None] * len(filters)
#stdControlError = [None] * len(filters)




for k in range(0, len(filters)):
    mEstimateError[k] = np.mean(EstimateError[k], axis = 0)
    stdEstimateError[k] = np.std(EstimateError[k], axis = 0)
    np.savetxt(EstimateErrorFileNameTemplate.replace('[filter]',filters[k]).replace('[pathnum]', 'mean'),  mEstimateError[k], fmt='%f')
    np.savetxt(EstimateErrorFileNameTemplate.replace('[filter]',filters[k]).replace('[pathnum]', 'std'),  stdEstimateError[k], fmt='%f')




