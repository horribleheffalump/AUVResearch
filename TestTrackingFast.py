"""Setting for target tracking model
"""

import os.path
from datetime import datetime

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

for p in paths:
    if not os.path.exists(p):
        os.makedirs(p)

# ########## AUV model definition ###################
from TrackingModel import *

def zeta(x, y):
    X_lasso = pipe_lasso.predict([y[:2 * Xb.shape[0]]])
    return np.concatenate((y - psi(x), X_lasso[0]))
    #x_ls = least_squares(lambda x_: cart2sphere(x_)[0] - y[:2 * Xb.shape[0]], x[0:3]).x
    #return np.concatenate((y - psi(x), x_ls))


cmnf = SimpleCMNFFilter(phi, psi, DW, DNu, xi, zeta)

Mtrain = 1000  # number of sample paths for CMNF parameters estimation (train set)
x, y = generate_sample_paths(Mtrain, int(T/delta))


M_ls = 10
X__ = y[:M_ls, :, :2 * Xb.shape[0]].reshape(M_ls * (N + 1), -1)
Y__ = x[:M_ls, :, :3].reshape(M_ls * (N + 1), -1)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.pipeline import make_pipeline

# Alpha (regularization strength) of LASSO regression
lasso_niter = 5000
lasso_eps = 0.01
lasso_nalpha = 20
# Min and max degree of polynomials features to consider
degree_min = 1
degree_max = 3
# Test/train split
X_train, X_test, y_train, y_test = train_test_split(X__, Y__, test_size=0.2)
# Make a pipeline model with polynomial transformation and LASSO regression with cross-validation, run it for increasing degree of polynomial (complexity of the model)
degree = 2
pipe_lasso = make_pipeline(PolynomialFeatures(degree, interaction_only=False), MultiTaskLassoCV(eps=lasso_eps, n_alphas=lasso_nalpha, normalize=True,cv=5, n_jobs=-1, max_iter=lasso_niter))
pipe_lasso.fit(X_train, y_train)
test_pred = np.array(pipe_lasso.predict(X_test))
#RMSE=np.sqrt(np.sum(np.square(predict_ml-y_test)))
std = np.std(test_pred - y_test, axis=0)
score = pipe_lasso.score(X_test,y_test)





cmnf.EstimateParameters(x, y, X0Hat)
Mtest = 1000  # number of samples

x, y = generate_sample_paths(Mtest, int(T/delta))
xHat = cmnf.Filter(y, X0Hat)

err = xHat - x
m_err = np.mean(err, axis=0)
std_err = np.std(err, axis=0)

# for i in range(0,17):
#     fig = plt.figure(figsize=(10, 6), dpi=200)
#     ax = fig.gca()
#     ax.plot(np.arange(0, T+delta/2, delta), x[0, :, i], linewidth=2.0, color='black')
#     ax.plot(np.arange(0, T+delta/2, delta), xHat[0, :, i], linewidth=2.0, color='red')
#     plt.show()

estimate_error_filename_template: str = "estimate_error_[filter]_[pathnum].txt"
filename_estimate_mean = os.path.join(path_estimates, estimate_error_filename_template.replace('[filter]', 'cmnf_new').replace('[pathnum]', 'mean'))
filename_estimate_std = os.path.join(path_estimates, estimate_error_filename_template.replace('[filter]', 'cmnf_new').replace('[pathnum]', 'std'))

np.savetxt(filename_estimate_mean, m_err, fmt='%f')
np.savetxt(filename_estimate_std, std_err, fmt='%f')
