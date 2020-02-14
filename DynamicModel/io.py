"""Save/load operations
"""
import numpy as np

def save_path(filename, x):
    np.savetxt(filename, x, fmt='%f')

def save_many(filename_template, x):
    for m in range(0, x.shape[0]):
        filename = filename_template.replace('[num]', str(m).zfill(int(np.log10(x.shape[0]))))
        np.savetxt(filename, x[m], fmt='%f')
