"""Save/load operations
"""
import numpy as np
import pandas as pd


def save_path(filename, x):
    np.savetxt(filename, x, fmt='%f')


def save_many(filename_template, x):
    for m in range(0, x.shape[0]):
        filename = filename_template.replace('[num]', str(m).zfill(int(np.log10(x.shape[0]))))
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
