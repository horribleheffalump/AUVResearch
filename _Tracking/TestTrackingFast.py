"""Setting for target tracking model
"""

import os.path
from datetime import datetime

from scipy.optimize import least_squares

from DynamicModel.io import *
from Filters.SimpleCMNFFilter import SimpleCMNFFilter


path = os.path.join('Z:\\Наука - Data\\2019 - Sensors - Tracking\\data', datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
# path = os.path.join('Z:\\Наука - Data\\2019 - Sensors - Tracking\\data', 'observations')
subdir_trajectories = "trajectories"
subdir_estimates = "estimates"
subdir_observations = "observations"
path_trajectories = os.path.join(path, subdir_trajectories)
path_estimates = os.path.join(path, subdir_estimates)
path_observations = os.path.join(path, subdir_observations)

paths = [path, path_trajectories, path_estimates, path_observations]

filename_template_estimate_error: str = "estimate_error_[filter]_[num].txt"
filename_template_trajectory: str = "trajectory_[num].txt"
filename_template_observations: str = "observations_[num].txt"

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


cmnf = SimpleCMNFFilter(phi, psi, DW, DNu, xi, zeta)

filters = {
    'cmnf': cmnf,
}


Mtrain = 100  # number of sample paths for CMNF parameters estimation (train set)
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


for (name, f) in filters.items():
    f.EstimateParameters(x, y, X0Hat)
    p = os.path.join(os.path.join(path_estimates, name))
    if not os.path.exists(p):
        os.makedirs(p)
    cmnf.SaveParameters(os.path.join(p, "[param].npy"))


Mtest = 100  # number of samples

x, y = generate_sample_paths(Mtest, int(T/delta))

save_many(os.path.join(path_trajectories, filename_template_trajectory), x)
save_many(os.path.join(path_observations, filename_template_observations), y)

for (name, f) in filters.items():
    xHat = cmnf.Filter(y, X0Hat)
    p = os.path.join(os.path.join(path_estimates, name))
    filename_template = os.path.join(p, filename_template_estimate_error.replace('[filter]', name))
    save_many(filename_template, xHat)

    err = xHat - x
    m_err = np.mean(err, axis=0)
    std_err = np.std(err, axis=0)

    save_path(filename_template.replace('[num]', 'mean'), m_err)
    save_path(filename_template.replace('[num]', 'std'), std_err)


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
