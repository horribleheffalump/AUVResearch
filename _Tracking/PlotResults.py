from DynamicModel.io import load_path
import pandas as pd
import numpy as np
import os.path
#'cmnf_basic',
filters = ['cmnf_ml_fp', 'cmnf_ml', 'cmnf_ls', 'cmnf_ml_fp_nodoppler', 'cmnf_ml_nodoppler', 'cmnf_ls_nodoppler']
filters_theor = ['cmnf_ml_fp', 'cmnf_ml', 'cmnf_ls', 'cmnf_ml_fp_nodoppler', 'cmnf_ml_nodoppler', 'cmnf_ls_nodoppler']
estimates = ['ml', 'ml_fp', 'ls']


cols = ['X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'x', 'y', 'z', 'v', 'phi', 'a', 'alpha', 'beta', 'RX', 'RY', 'RZ']
dir = '//172.17.10.161/z/Наука - Data/2019 - Sensors - Tracking/data/test_all/estimates/'
dir = 'Z:/Наука - Data/2019 - Sensors - Tracking/data/test/estimates/'

file_template = os.path.join(dir, '<filter>/estimate_error_<filter>_std.txt')
theor_file_template = os.path.join(dir, '<filter>/KHat.npy')

dfs = []
for f in filters:
    file_name = file_template.replace('<filter>', f)
    data = load_path(file_name, [f'{c}_{f}' for c in cols])
    dfs.append(data)

for f in filters_theor:
    file_name = theor_file_template.replace('<filter>', f)
    KHat = np.load(file_name)
    theor_data = pd.DataFrame(np.array([np.sqrt(KHat[:,i,i]) for i in range(0, len(cols))]).T, columns=[f'{c}_{f}_theor' for c in cols])
    dfs.append(theor_data)

for e in estimates:
    file_name = file_template.replace('<filter>', e)
    data = load_path(file_name, [f'{c}_{e}' for c in cols[:3]])
    dfs.append(data)



alldata = pd.concat(dfs, axis=1, sort=False)

def all_cols(col):
    f_cols_theor = [f'{col}_{f}_theor' for f in filters_theor]
    f_cols = [f'{col}_{f}' for f in filters]
    e_cols = [f'{col}_{e}' for e in estimates]
    return f_cols + f_cols_theor + e_cols

def f_cols(col):
    f_cols = [f'{col}_{f}' for f in filters]
    return f_cols


filters = []
filters_theor = ['cmnf_basic']
estimates = []

alldata[all_cols('X')][:].plot()
alldata[all_cols('Y')][:].plot()
alldata[all_cols('Z')][:].plot()

alldata[f_cols('VX')][:].plot()
alldata[f_cols('VY')][:].plot()
alldata[f_cols('VZ')][:].plot()

alldata[f_cols('v')][:].plot()
alldata[f_cols('phi')][:].plot()
alldata[f_cols('a')][:].plot()

alldata[f_cols('alpha')][:].plot()
alldata[f_cols('beta')][:].plot()


alldata[all_cols('X')][:].plot(ylim=(0,100))
alldata[all_cols('Y')][:].plot(ylim=(0,100))
alldata[all_cols('Z')][:].plot(ylim=(0,100))


alldata[all_cols('X')][:].plot(ylim=(0,25),xlim=(700,1000))
alldata[all_cols('Y')][:].plot(ylim=(0,50),xlim=(700,1000))
alldata[all_cols('Z')][:].plot(ylim=(0,50),xlim=(700,1000))

alldata[f_cols('VX')][:400].plot()
alldata[f_cols('VY')][:400].plot()
alldata[f_cols('VZ')][:400].plot()
