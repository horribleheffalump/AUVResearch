import numpy as np
import glob

doCalculateControls = False
max_paths = 1e100

#filters = ['cmnf', 'pseudo', 'kalman']
filters = ['cmnf']
#filters = ['pseudo']

#filename_template_estimate_error = "D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\byvirt\\estimate\\estimate_error_[filter]_[pathnum].txt"
#ControlErrorFileNameTemplate = "D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\byvirt\\control\\control_error_[filter]_[pathnum].txt"
EstimateErrorFileNameTemplate = "D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\acoustic\\estimate\\estimate_error_[filter]_[pathnum].txt"
ControlErrorFileNameTemplate = "D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\acoustic\\control\\control_error_[filter]_[pathnum].txt"

EstimateError = [None] * len(filters)
ControlError = [None] * len(filters)
ControlErrorNorm = [None] * len(filters)

for k in range(0, len(filters)):
    print(filters[k])
    m = 0
    files = glob.glob(EstimateErrorFileNameTemplate.replace("[filter]", filters[k]).replace("[pathnum]", "*"))
    try:
        files.remove(EstimateErrorFileNameTemplate.replace("[filter]", filters[k]).replace("[pathnum]", "mean"))
        files.remove(EstimateErrorFileNameTemplate.replace("[filter]", filters[k]).replace("[pathnum]", "std"))
    except ValueError:
        pass
    for file in files:
        data = np.loadtxt(file)
        if EstimateError[k] is None:
            EstimateError[k] = np.zeros((len(files), data.shape[0], data.shape[1]))
        EstimateError[k][m,:,:] = data

        #if (np.max(np.linalg.norm(data, axis = 1)) > 10000.0):
        #    print(file)
        if np.max(data) > 5000.0:
            print(file)
        else:
            m += 1
        if m % 1000 == 0:
            print('estimate done ', m, ' of ', len(files))
        if m == max_paths:
            break
    EstimateError[k] = EstimateError[k][:m, :, :]
    if doCalculateControls:
        m = 0
        files = glob.glob(ControlErrorFileNameTemplate.replace("[filter]", filters[k]).replace("[pathnum]", "*"))
        try:
            files.remove(ControlErrorFileNameTemplate.replace("[filter]", filters[k]).replace("[pathnum]", "mean"))
            files.remove(ControlErrorFileNameTemplate.replace("[filter]", filters[k]).replace("[pathnum]", "std"))
        except ValueError:
            pass
        for file in files:
            data = np.loadtxt(file)
            if ControlError[k] is None:
                ControlError[k] = np.zeros((len(files), data.shape[0], data.shape[1]))
                ControlErrorNorm[k] = np.zeros((len(files), data.shape[0]))
            ControlError[k][m,:,:] = data
            ControlErrorNorm[k][m,:] = np.power(np.linalg.norm(data, axis = 1), 2)
            if np.max(np.power(np.linalg.norm(data, axis = 1),2)) > 1000:
                print(file)
            m += 1
            if m % 1000 == 0:
                print('control done ', m, ' of ', len(files))
    
    

mEstimateError = [None] * len(filters)
stdEstimateError = [None] * len(filters)
if doCalculateControls:
    mControlError = [None] * len(filters)
    stdControlError = [None] * len(filters)
    mControlErrorNorm = [None] * len(filters)
    for k in range(0, len(filters)):
        mControlErrorNorm[k] = np.max(np.mean(ControlErrorNorm[k], axis = 0))
        print(filters[k], mControlErrorNorm[k])



for k in range(0, len(filters)):
    mEstimateError[k] = np.mean(EstimateError[k], axis = 0)
    stdEstimateError[k] = np.std(EstimateError[k], axis = 0)
    np.savetxt(EstimateErrorFileNameTemplate.replace('[filter]',filters[k]).replace('[pathnum]', 'mean'),  mEstimateError[k], fmt='%f')
    np.savetxt(EstimateErrorFileNameTemplate.replace('[filter]',filters[k]).replace('[pathnum]', 'std'),  stdEstimateError[k], fmt='%f')

if doCalculateControls:
    for k in range(0, len(filters)):
        mControlError[k] = np.mean(ControlError[k], axis = 0)
        stdControlError[k] = np.std(ControlError[k], axis = 0)
        np.savetxt(ControlErrorFileNameTemplate.replace('[filter]',filters[k]).replace('[pathnum]', 'mean'),  mControlError[k], fmt='%f')
        np.savetxt(ControlErrorFileNameTemplate.replace('[filter]',filters[k]).replace('[pathnum]', 'std'),  stdControlError[k], fmt='%f')



