"""Setting for target tracking model
"""

import os.path
from datetime import datetime
from functools import partial

from scipy.optimize import least_squares

from DynamicModel.io import *
from Filters.SimpleCMNFFilter import SimpleCMNFFilter
from multiprocessing import Pool, cpu_count


path = os.path.join('Z:\\Наука - Data\\2019 - Sensors - Tracking\\data\\test')
# path = os.path.join('Z:\\Наука - Data\\2019 - Sensors - Tracking\\data', 'observations')
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

# ########## AUV model definition ###################
from _Tracking.TrackingModel import *

def zeta_basic(x, y):
    return np.concatenate(y - psi(x))

# def zeta_ml(x, y):
#     X_lasso = pipe_lasso.predict([y[:2 * Xb.shape[0]]])
#     return np.concatenate((y - psi(x), X_lasso[0]))

def zeta_ls(x, y):
    x_ls = least_squares(lambda x_: cart2sphere(x_)[0] - y[:2 * Xb.shape[0]], x[0:3]).x
    return np.concatenate((y - psi(x), x_ls))


cmnf_basic = SimpleCMNFFilter(phi, psi, DW, DNu, xi, zeta)
cmnf_ls = SimpleCMNFFilter(phi, psi, DW, DNu, xi, zeta_ls)

filters = {
    'cmnf_basic': cmnf_basic,
    #'cmnf_ls': cmnf_ls,
}


def filter_folder(name):
    return os.path.join(path_estimates, name)


for filter_name in filters:
    p = filter_folder(filter_name)
    if not os.path.exists(p):
        os.makedirs(p)

Mtrain = 10000  # number of sample paths for CMNF parameters estimation (train set)
x, y = generate_sample_paths(Mtrain, int(T/delta))
#
# M_ls = 10
# X__ = y[:M_ls, :, :2 * Xb.shape[0]].reshape(M_ls * (N + 1), -1)
# Y__ = x[:M_ls, :, :3].reshape(M_ls * (N + 1), -1)


# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import MultiTaskLassoCV
# from sklearn.pipeline import make_pipeline
#
# # Alpha (regularization strength) of LASSO regression
# lasso_niter = 5000
# lasso_eps = 0.01
# lasso_nalpha = 20
# # Min and max degree of polynomials features to consider
# degree_min = 1
# degree_max = 3
# # Test/train split
# X_train, X_test, y_train, y_test = train_test_split(X__, Y__, test_size=0.2)
# # Make a pipeline model with polynomial transformation and LASSO regression with cross-validation, run it for increasing degree of polynomial (complexity of the model)
# degree = 2
# pipe_lasso = make_pipeline(PolynomialFeatures(degree, interaction_only=False), MultiTaskLassoCV(eps=lasso_eps, n_alphas=lasso_nalpha, normalize=True,cv=5, n_jobs=-1, max_iter=lasso_niter))
# pipe_lasso.fit(X_train, y_train)
# test_pred = np.array(pipe_lasso.predict(X_test))
# #RMSE=np.sqrt(np.sum(np.square(predict_ml-y_test)))
# std = np.std(test_pred - y_test, axis=0)
# score = pipe_lasso.score(X_test,y_test)

if __name__ == '__main__':
    for (name, f) in filters.items():
        print(f'estimate started for {name}')
        f.EstimateParameters(x, y, X0Hat, silent=True)
        f.SaveParameters(os.path.join(filter_folder(name), "[param].npy"))


Mtest = 10000  # number of samples
pack_size = 100


def process_pack(pack_start, size):
    x_, y_ = generate_sample_paths(pack_size, int(T/delta))

    save_many(os.path.join(path_trajectories, filename_template_trajectory), x_, pack_start, Mtest)
    save_many(os.path.join(path_observations, filename_template_observations), y_, pack_start, Mtest)

    for (name_, f_) in filters.items():
        f_.LoadParameters(os.path.join(filter_folder(name_), "[param].npy"))
        x_hat = f_.Filter(y_, X0Hat, silent=True)
        filename_template = os.path.join(filter_folder(name_), filename_template_estimate_error.replace('[filter]', name_))
        save_many(filename_template, x_ - x_hat, pack_start, Mtest)
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

    calculate_stats_templates = [os.path.join(filter_folder(name_), filename_template_estimate_error.replace('[filter]', name_)) for name_ in filters]
    calculate_stats_templates.append(os.path.join(path_trajectories, filename_template_trajectory))

    for t in calculate_stats_templates:
        print(f'stats calculation started for {t}')
        calculate_stats(t, save=True)

#
#
# for i in range(0, int(Mtest/pack_size)):
#     pack_start = i * pack_size
#     pack_size = min(pack_size, Mtest - pack_start)
#     x, y = generate_sample_paths(pack_size, int(T/delta))
#
#     save_many(os.path.join(path_trajectories, filename_template_trajectory), x, pack_start, Mtest)
#     save_many(os.path.join(path_observations, filename_template_observations), y, pack_start, Mtest)
#
#     for (name, f) in filters.items():
#         xHat = cmnf.Filter(y, X0Hat)
#         p = os.path.join(os.path.join(path_estimates, name))
#         filename_template = os.path.join(p, filename_template_estimate_error.replace('[filter]', name))
#         save_many(filename_template, x - xHat, pack_start, Mtest)

        # err = xHat - x
        # m_err = np.mean(err, axis=0)
        # std_err = np.std(err, axis=0)
        #
        # save_path(filename_template.replace('*', 'mean'), m_err)
        # save_path(filename_template.replace('*', 'std'), std_err)


### plot one
# import matplotlib.pyplot as plt
# dir = '2020-02-04-18-24-39'
# n = '0000'
# path_filename = f"Z:/Наука - Data/2019 - Sensors - Tracking/data/{dir}/trajectories/path_cmnf_{n}.txt"
# estimate_error_filename = f"Z:/Наука - Data/2019 - Sensors - Tracking/data/{dir}/estimates/estimate_error_cmnf_{n}.txt"
# predict_filename = f"Z:/Наука - Data/2019 - Sensors - Tracking/data/{dir}/estimates/predict_cmnf_{n}.txt"
# correct_filename = f"Z:/Наука - Data/2019 - Sensors - Tracking/data/{dir}/estimates/correct_cmnf_{n}.txt"
#
# column_names = ['X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'x', 'y', 'z', 'v', 'phi', 'a', 'alpha', 'beta', 'RX', 'RY', 'RZ']
#
# df = load_path(path_filename, column_names, {'est': estimate_error_filename, 'predict': predict_filename, 'correct': correct_filename, })
#
# def cols(str):
#     postfixes = ['', '_est', '_predict','_correct']
#     return [str + p for p in postfixes]
#
# for var in column_names:
#     df[cols(var)].iloc[:].plot()
#     plt.show()
