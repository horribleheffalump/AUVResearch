"""Save/load operations
"""

import numpy as np
import pandas as pd
from glob import glob
from sys import maxsize, float_info


def save_path(filename, x):
    np.savetxt(filename, x, fmt='%f')


def save_many(filename_template, x, start_from=0, total=None):
    if total is None:
        total = x.shape[0]
    for m in range(0, x.shape[0]):
        filename = filename_template.replace('*', str(start_from + m).zfill(int(np.log10(total))))
        np.savetxt(filename, x[m], fmt='%f')


def load_path(file_name_path, column_names, estimate_errors_files_dict=None):
    path_data = np.loadtxt(file_name_path)
    df = pd.DataFrame(path_data)
    df.columns = column_names

    if estimate_errors_files_dict is not None:
        dfs = [df]
        for contents, file_name in estimate_errors_files_dict.items():
            column_names_full = [f"{n}{contents}" for n in column_names]
            error_data = np.loadtxt(file_name)
            df = pd.DataFrame(path_data - error_data)
            df.columns = column_names_full
            dfs.append(df)
        df = pd.concat(dfs, axis=1, sort=False)

    return df


def calculate_stats(file_template, diverged_limit=float_info.max, max_paths=maxsize, silent=False, save=False):
    ensemble = None
    m = 0
    files = glob(file_template)
    mean_file = file_template.replace('*', 'mean')
    std_file = file_template.replace('*', 'std')
    ignore = [mean_file, std_file]
    for f in ignore:
        try:
            files.remove(f)
        except ValueError:
            pass
    for f in files:
        data = np.loadtxt(f)
        if ensemble is None:
            ensemble = np.zeros((len(files), data.shape[0], data.shape[1]))
        ensemble[m, :, :] = data

        if np.max(data) > diverged_limit:
            print(f)
        else:
            m += 1
        if m % 1000 == 0:
            if not silent:
                print('estimate done ', m, ' of ', len(files))
        if m == max_paths:
            break
    ensemble = ensemble[:m, :, :]
    mean = np.mean(ensemble, axis=0)
    std = np.std(ensemble, axis=0)
    if save:
        save_path(mean_file, mean)
        save_path(std_file, std)
    return mean, std
