"""Setting for target tracking model
"""

import os.path
from datetime import datetime
from functools import partial

from scipy.optimize import least_squares

from DynamicModel.io import *
from Filters.SimpleCMNFFilter import SimpleCMNFFilter
from multiprocessing import Pool, cpu_count

from joblib import dump, load

from StaticEstimates.PriorLearn import PriorLearn
from StaticEstimates.LeastSquares import LeastSquares

path = os.path.join('Z:\\Наука - Data\\2019 - Sensors - Tracking\\data\\test')
subdir_trajectories = "trajectories"
subdir_estimates = "estimates"
subdir_observations = "observations"
path_trajectories = os.path.join(path, subdir_trajectories)
path_estimates = os.path.join(path, subdir_estimates)
path_observations = os.path.join(path, subdir_observations)

paths = [path, path_trajectories, path_estimates, path_observations]

filename_template_estimate_error: str = "estimate_error_[filter]_*.txt"
filename_template_trajectory: str = "trajectory_*.txt"
filename_template_observations: str = "observations_*.txt"

for p in paths:
    if not os.path.exists(p):
        os.makedirs(p)

pipe_file_name = "Z:/Наука - Data/2019 - Sensors - Tracking/data/pipe.joblib"
pipe_lasso = load(pipe_file_name)
pipe_file_name = "Z:/Наука - Data/2019 - Sensors - Tracking/data/pipe_full_path.joblib"
pipe_lasso_full_path = load(pipe_file_name)

# ########## AUV model definition ###################
from _Tracking.TrackingModel import *

def zeta_basic(x, y):
    return np.concatenate(y - psi(x))

def zeta_ml(x, y):
    X_lasso = pipe_lasso.predict([y[:2 * Xb.shape[0]]])
    return np.concatenate((y - psi(x), X_lasso[0]))

def zeta_ml_fp(x, y):
    X_lasso = pipe_lasso_full_path.predict([y[:2 * Xb.shape[0]]])
    return np.concatenate((y - psi(x), X_lasso[0]))

def zeta_ls(x, y):
    x_ls = least_squares(lambda x_: cart2sphere(x_)[:2 * Xb.shape[0]] - y[:2 * Xb.shape[0]], x[0:3]).x
    return np.concatenate((y - psi(x), x_ls))

def zeta_ml_nodoppler(x, y):
    X_lasso = pipe_lasso.predict([y[:2 * Xb.shape[0]]])
    return np.concatenate(((y - psi(x))[:2 * Xb.shape[0]], X_lasso[0]))

def zeta_ml_fp_nodoppler(x, y):
    X_lasso = pipe_lasso_full_path.predict([y[:2 * Xb.shape[0]]])
    return np.concatenate(((y - psi(x))[:2 * Xb.shape[0]], X_lasso[0]))

def zeta_ls_nodoppler(x, y):
    x_ls = least_squares(lambda x_: cart2sphere(x_)[:2 * Xb.shape[0]] - y[:2 * Xb.shape[0]], x[0:3]).x
    return np.concatenate(((y - psi(x))[:2 * Xb.shape[0]], x_ls))



cmnf_basic = SimpleCMNFFilter(phi, psi, DW, DNu, xi, zeta)
cmnf_ls = SimpleCMNFFilter(phi, psi, DW, DNu, xi, zeta_ls)
cmnf_ml = SimpleCMNFFilter(phi, psi, DW, DNu, xi, zeta_ml)
cmnf_ml_fp = SimpleCMNFFilter(phi, psi, DW, DNu, xi, zeta_ml_fp)
cmnf_ls_nodoppler = SimpleCMNFFilter(phi, psi, DW, DNu, xi, zeta_ls_nodoppler)
cmnf_ml_nodoppler = SimpleCMNFFilter(phi, psi, DW, DNu, xi, zeta_ml_nodoppler)
cmnf_ml_fp_nodoppler = SimpleCMNFFilter(phi, psi, DW, DNu, xi, zeta_ml_fp_nodoppler)

filters = {
    'cmnf_basic': cmnf_basic,
    'cmnf_ls': cmnf_ls,
    'cmnf_ml': cmnf_ml,
    'cmnf_ml_fp': cmnf_ml_fp,
    'cmnf_ls_nodoppler': cmnf_ls_nodoppler,
    'cmnf_ml_nodoppler': cmnf_ml_nodoppler,
    'cmnf_ml_fp_nodoppler': cmnf_ml_fp_nodoppler,
}

estimator_ml = PriorLearn(pipe_lasso)
estimator_ml_fp = PriorLearn(pipe_lasso_full_path)
estimator_ls = LeastSquares(lambda x_, y_: cart2sphere(x_[0:3])[:2 * Xb.shape[0]] - y_[:2 * Xb.shape[0]], X0Hat[0:3])

estimators = {
    "ls": estimator_ls,
    "ml": estimator_ml,
    "ml_fp": estimator_ml_fp,
}

def filter_folder(name):
    return os.path.join(path_estimates, name)


for filter_name in filters:
    p = filter_folder(filter_name)
    if not os.path.exists(p):
        os.makedirs(p)

for filter_name in estimators:
    p = filter_folder(filter_name)
    if not os.path.exists(p):
        os.makedirs(p)

Mtrain = 100  # number of sample paths for CMNF parameters estimation (train set)
x, y = generate_sample_paths(Mtrain, int(T/delta))

if __name__ == '__main__':
    for (name, f) in filters.items():
        print(f'estimate started for {name}')
        f.EstimateParameters(x, y, X0Hat, silent=False)
        f.SaveParameters(os.path.join(filter_folder(name), "[param].npy"))


Mtest = 100000 # number of samples
pack_size = 1000


def process_pack(pack_start, size):
    x_, y_ = generate_sample_paths(pack_size, int(T/delta))

    save_many(os.path.join(path_trajectories, filename_template_trajectory), x_, pack_start, Mtest)
    save_many(os.path.join(path_observations, filename_template_observations), y_, pack_start, Mtest)

    for (name_, f_) in filters.items():
        f_.LoadParameters(os.path.join(filter_folder(name_), "[param].npy"))
        x_hat = f_.Filter(y_, X0Hat, silent=True)
        filename_template = os.path.join(filter_folder(name_), filename_template_estimate_error.replace('[filter]', name_))
        save_many(filename_template, x_ - x_hat, pack_start, Mtest)

    for name_, estimator_ in estimators.items():
        x_hat = np.array([estimator_.predict(var) for var in y_[:, :, :2*Xb.shape[0]]])
        filename_template = os.path.join(filter_folder(name_), filename_template_estimate_error.replace('[filter]', name_))
        save_many(filename_template, x_[:,:,:3] - x_hat, pack_start, Mtest)

    print(f'finished {pack_start}-{pack_start + size}')
    return pack_start + size


pack_start_nums = np.arange(0, Mtest, pack_size)
task = partial(process_pack, size=pack_size)

# for sequential calculation
#a = list(map(task, pack_start_nums))

# for parallel calculation
if __name__ == '__main__':
    with Pool(cpu_count()) as p:
        p.map(task, pack_start_nums)

    calculate_stats_filters_templates = [os.path.join(filter_folder(name_), filename_template_estimate_error.replace('[filter]', name_)) for name_ in filters]
    calculate_stats_estimators_templates = [os.path.join(filter_folder(name_), filename_template_estimate_error.replace('[filter]', name_)) for name_ in estimators]
    calculate_stats_templates = [os.path.join(path_trajectories, filename_template_trajectory)] + calculate_stats_filters_templates + calculate_stats_estimators_templates

    calculate_stats_limit = partial(calculate_stats, diverged_limit=1e40, save=True)

    with Pool(cpu_count()) as p:
        p.map(calculate_stats_limit, calculate_stats_templates)

    # for t in calculate_stats_templates:
    #     print(f'stats calculation started for {t}')
    #     calculate_stats(t, save=True)


#cpu_count()