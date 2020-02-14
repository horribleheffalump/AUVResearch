"""Setting for target tracking model
"""

import os.path
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Filters.CMNFFilter import *
from Filters.SimpleCMNFFilter import SimpleCMNFFilter

from NonlinearModel.SimpleNonlinearModel import *

from numba import jit
from scipy.optimize import least_squares

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

T = 100.0  # simulation time limit
delta = 1.0  # simulation discretization step
N = int(T / delta)  # number of time instants after discretization

lamb = 0.01
nu = 0.0
mu = 1e-2

# normal coords  | in turned plane | turn angle | plane shift
# X Y Z VX VY VZ   x y z v phi a     alpha beta   RX RY RZ
# 0 1 2 3  4  5    6 7 8 9 10  11    12    13     14 15 16

m_W = np.array([0.0] * (6 + 6 + 5))
std_W = np.array([0.0] * (6 + 6 + 5))
std_W[11] = mu
std_W = std_W * np.sqrt(delta)
DW = np.power(std_W, 2.0)


class PseudoAUV:
    def step(self, XHat):
        # do nothing
        _ = XHat


def Phi(model, k, X, XHat):
    # state transformation in the plane of the turn
    [x_, y_, z_, v_, phi_, an_] = X[6:12]
    alpha = X[12]
    beta = X[13]
    R0 = X[14:17]
    dx = v_ * np.cos(phi_) * delta
    dy = v_ * np.sin(phi_) * delta
    dz = 0.0
    dv = 0.0
    dphi = an_ / v_ * delta
    dan = (-lamb * an_ + nu) * delta
    x_turned_new = np.array([x_ + dx, y_ + dy, z_ + dz, v_ + dv, phi_ + dphi, an_ + dan])
    x_orig = toOriginalCoordinates(x_turned_new, alpha, beta, R0)
    # original coordinates, coordinates in turned plane, alpha, beta, R0
    return np.hstack((x_orig, x_turned_new, alpha, beta, R0))

# @jit(cache=True)
# def phi(X):
#     # state transformation in the plane of the turn
#     [x_, y_, z_, v_, phi_, an_] = X[6:12]
#     alpha = X[12]
#     beta = X[13]
#     R0 = X[14:17]
#     dx = v_ * np.cos(phi_) * delta
#     dy = v_ * np.sin(phi_) * delta
#     dz = 0.0
#     dv = 0.0
#     dphi = an_ / v_ * delta
#     dan = (-lamb * an_ + nu) * delta
#     x_turned_new = np.array([x_ + dx, y_ + dy, z_ + dz, v_ + dv, phi_ + dphi, an_ + dan])
#     x_orig = toOriginalCoordinates(x_turned_new, alpha, beta, R0)
#     # original coordinates, coordinates in turned plane, alpha, beta, R0
#     return np.hstack((x_orig, x_turned_new, np.array([alpha, beta]), R0))
# # ########## Observation model definition ###################

@jit(cache=True)
def toOriginalCoordinates(X, alpha, beta, R0):
    X_ = X[0:3]
    v_ = X[3]
    phi_ = X[4]
    V_ = np.array([v_ * np.cos(phi_), v_ * np.sin(phi_), 0.0])
    A = np.array([[np.cos(beta), np.sin(alpha) * np.sin(beta), np.sin(beta) * np.cos(alpha)],
                  [1.0, np.cos(alpha), -np.sin(alpha)],
                  [-np.sin(beta), np.sin(alpha) * np.cos(beta), np.cos(beta) * np.cos(alpha)]])
    return np.concatenate((A @ X_ + R0, A @ V_))


# array of sensors' positions
Xb = np.array([[-10000.0, 0.0, -25.0], [-10000.0, 0.0, -50.0], [-5000.0, 1000.0, -25.0], [-5000.0, 1000.0, -50.0],
               [5000.0, 1000.0, -25.0], [5000.0, 1000.0, -50.0], [10000.0, 0.0, -25.0], [10000.0, 0.0, -50.0]])

# standard variation and the covariance matrix of the noise in observations
std_Nu0 = np.sin(1 * np.pi / 180.0)  # ~1 degree
std_Nu = np.concatenate(([std_Nu0] * Xb.shape[0], [std_Nu0] * Xb.shape[0], [0.005] * Xb.shape[0]))
m_Nu = np.zeros_like(std_Nu)
DNu = np.power(std_Nu, 2.0)
omega0 = 20.0  # [Hz] frequency of sound signal
C = 1500.0  # [m/s] sound speed

@jit(cache=True)
def cart2sphere(_x):
    #_R = np.linalg.norm(_x - Xb, axis=1)
    _R = np.sqrt((_x[0] - Xb[:, 0]) ** 2 + (_x[1] - Xb[:, 1]) ** 2 + (_x[2] - Xb[:, 2]) ** 2)
    _nu = (_x[2] - Xb[:, 2]) / _R
    #_r = np.linalg.norm((_x - Xb)[:,:2], axis=1)
    _r = np.sqrt((_x[0] - Xb[:, 0]) ** 2 + (_x[1] - Xb[:, 1]) ** 2)
    _xi = (_x[0] - Xb[:, 0]) / _r
    return np.hstack((_xi, _nu, _R))

def Psi(model, k, X, y):
    # observation transformation
    _x = X[0:3]
    # _angles, _R = cart2sphere(_x)
    _sphere = cart2sphere(_x)
    _angles = _sphere[:2 * Xb.shape[0]]
    _R = _sphere[-Xb.shape[0]:]
    _v = X[3:6]
    _V = ((_x[0] - Xb[:, 0]) * _v[0] + (_x[1] - Xb[:, 1]) * _v[1] + (_x[2] - Xb[:, 2]) * _v[2]) / _R
    _omega = omega0 / (1.0 - _V / C)
    return np.hstack((_angles, _omega))



    # return np.array[0.0]

# @jit(cache=True)
# def psi(X):
#     # observation transformation
#     _x = X[0:3]
#     _sphere = cart2sphere(_x)
#     _angles = _sphere[:2 * Xb.shape[0]]
#     _R = _sphere[-Xb.shape[0]:]
#     _v = X[3:6]
#     _V = ((_x[0] - Xb[:, 0]) * _v[0] + (_x[1] - Xb[:, 1]) * _v[1] + (_x[2] - Xb[:, 2]) * _v[2]) / _R
#     _omega = omega0 / (1.0 - _V / C)
#     return np.hstack((_angles, _omega))

# ########## generate a sample path ###################


#model = PseudoAUV()

m_x0 = np.array([0.0, 0.0, 0.0])
# std_x0 = np.array([10.0, 10.0, 10.0])

min_v0 = np.array([5.0])
max_v0 = np.array([12.0])
m_v0 = 0.5 * (min_v0 + max_v0)

m_phi0 = np.array([-np.pi / 2])
std_phi0 = np.array([0.1])

# m_an0 = np.array([0.3])
# std_an0 = np.array([0.0])

min_an0 = np.array([-0.2])
max_an0 = np.array([0.2])
m_an0 = 0.5 * (min_an0 + max_an0)

m_rotate = np.array([0.0, 0.0])
std_rotate = np.array([np.pi / 36.0, np.pi / 36.0])

m_shift = np.array([0.0, 20000.0, -1000.0])
std_shift = np.array([1000.0, 1000.0, 100.0])

turned_coords = np.concatenate((m_x0, m_v0, m_phi0, m_an0))
orig_coords = toOriginalCoordinates(turned_coords, m_rotate[0], m_rotate[1], m_shift)

X0Hat = np.concatenate((orig_coords, turned_coords, m_rotate, m_shift))

@jit(cache=True)
def sample_X0():
    # x0 = np.random.normal(m_x0, std_x0)
    x0 = m_x0
    v0 = np.random.uniform(min_v0[0], max_v0[0])
    phi0 = np.random.normal(m_phi0[0], std_phi0[0])
    an0 = np.random.uniform(min_an0[0], max_an0[0])
    turned_coords = np.concatenate((x0, np.array([v0, phi0, an0])))
    rotate = sample_normal(m_rotate, std_rotate)
    shift = sample_normal(m_shift, std_shift)
    orig_coords = toOriginalCoordinates(turned_coords, rotate[0], rotate[1], shift)
    return np.concatenate((orig_coords, turned_coords, rotate, shift))

@jit(cache=True)
def sample_normal(m, std):
    x = np.zeros_like(m)
    for i in range(0, m.shape[0]):
       x[i] = np.random.normal(m[i], std[i])
    return x


# fig = plt.figure(figsize=(5, 5), dpi=200)
# ax = Axes3D(fig)  # fig.gca(projection='3d')


# n_plots = 100
# for i in range(0, n_plots):
#    Xs = [sample_X0()]
#    for t in range(1, N + 1):
#        x = Phi(model, t - 1, Xs[t - 1], []) + np.random.normal(m_W, std_W)
#        Xs.append(x)  # store the current position
#    Xs = np.array(Xs)
#    ax.plot(Xs[:, 0] - Xs[:, -3], Xs[:, 1] - Xs[:, -2], Xs[:, 2] - Xs[:, -1], linewidth=2.0)
#    #ax.plot(Xs[:, 0], Xs[:, 1], Xs[:, 2], linewidth=2.0)
# plt.show()

# plot in the turn plane
# fig = plt.figure(figsize=(10, 6), dpi=200)
# ax = fig.gca()
# ax.plot(Xs[:, 6], Xs[:, 7], color='red', linewidth=2.0)
# plt.show()

# plot in the original coordinates
# fig = plt.figure(figsize=(10, 6), dpi=200)
# ax = Axes3D(fig) #fig.gca(projection='3d')
# ax.plot(Xs[:, 0], Xs[:, 1], Xs[:, 2], color='red', linewidth=2.0)
# plt.show()


# ################ Generate paths ###################################

# # samples calculation
# MSamples = 10
# Path = np.zeros((MSamples, N + 1, X0Hat.shape[0]))
# Observations = np.zeros((MSamples, N + 1, Psi(PseudoAUV(), [], X0Hat, []).shape[0]))
# for m in range(0, MSamples):
#     if m % 1000 == 0:
#         print('Sample path m=', m)
#     X0 = sample_X0()
#
#     # init a sample path
#     model = PseudoAUV()
#
#     Path[m, 0, :] = X0
#     Observations[m, 0, :] = Psi(model, [], X0, [])
#
#     # calculate a sample path and estimate step-by-step
#     for i in range(0, N):
#         Path[m, i + 1, :] = Phi(model, [], Path[m, i, :], []) + np.random.normal(m_W, std_W)
#         Observations[m, i + 1, :] = Psi(model, [], Path[m, i + 1, :], []) + np.random.normal(m_Nu, std_Nu)
#
# X__ = Observations[:, :, :2 * Xb.shape[0]].reshape(MSamples * (N + 1), -1)
# Y__ = Path[:, :, :3].reshape(MSamples * (N + 1), -1)
#
#
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import MultiTaskLassoCV
# from sklearn.pipeline import make_pipeline
#
# # Alpha (regularization strength) of LASSO regression
# lasso_niter = 5000
# lasso_eps = 0.01
# lasso_nalpha=20
# # Min and max degree of polynomials features to consider
# degree_min = 1
# degree_max = 3
# # Test/train split
# x_train, x_test, y_train, y_test = train_test_split(X__, Y__, test_size=0.2)
# stds = []
# scores = []
# # Make a pipeline model with polynomial transformation and LASSO regression with cross-validation, run it for increasing degree of polynomial (complexity of the model)
# degree = 2
# pipe_lasso = make_pipeline(PolynomialFeatures(degree, interaction_only=False), MultiTaskLassoCV(eps=lasso_eps, n_alphas=lasso_nalpha, normalize=True,cv=5, n_jobs=-1, max_iter=lasso_niter))
# pipe_lasso.fit(x_train, y_train)
# predict_ml = np.array(pipe_lasso.predict(x_test))
# #RMSE=np.sqrt(np.sum(np.square(predict_ml-y_test)))
# stds.append(np.std(predict_ml - y_test, axis=0))
# scores.append(pipe_lasso.score(x_test,y_test))

# predict_ml = []
# for obs in x_test:
#     y = least_squares(lambda x: cart2sphere(x)[0] - obs, X0Hat[:3]).x
#     predict_ml.append(y)
# predict_ml = np.array(predict_ml)
# stds.append(np.std(predict_ml - y_test, axis=0))

# ########### FILTERS #########################

# CMNF filter definition

def Xi(model, k, XHat):
    # CMNF basic prediction
    return Phi(model, k, XHat, XHat)


def Zeta(model, k, X, y):
    # CMNF basic correction
    #X_ls = least_squares(lambda x: cart2sphere(x)[0] - y[:2 * Xb.shape[0]], X[0:3]).x
    # X_lasso = pipe_lasso.predict([y[:2 * Xb.shape[0]]])
    # return np.concatenate((y - Psi(model, k, X, y), X_lasso[0]))
    return y - Psi(model, k, X, y)


cmnf = CMNFFilter(Phi, Psi, DW, DNu, Xi, Zeta)

# uncomment if parameters are estimated anew
Mtrain = 10000  # number of sample paths for CMNF parameters estimation (train set)
X0all = np.array(list(map(lambda i_: sample_X0(), range(0, Mtrain))))  # initial point for the training sample paths
models = np.array(list(map(lambda i_: PseudoAUV(), range(0, Mtrain))))  # models for compatibility with CMNF
cmnf.EstimateParameters(models, X0all, X0Hat, N, Mtrain, os.path.join(path, "[param].npy"))
cmnf.SaveParameters(os.path.join(path, "[param].npy"))



# ########### estimation and control samples calculation ##############

M = 10000  # number of samples

# set of filters for position estimation, their names and do they need the pseudomeasurements

filters = [cmnf]
names = ['cmnf']
needsPseudoMeasurements = [False]

# initialization

Path = []  # array to store path samples
Observations = []  # array to store observations
EstimateError = []  # array to store position estimation error samples
Predictions = []  # array to store predictions
Corrections = []  # array to store corrections

path_filename_template: str = "path_[filter]_[pathnum].txt"
observations_filename_template: str = "obs_[filter]_[pathnum].txt"
estimate_error_filename_template: str = "estimate_error_[filter]_[pathnum].txt"

for k in range(0, len(filters)):
    Path.append(np.zeros((M, N + 1, X0Hat.shape[0])))
    Observations.append(np.zeros((M, N + 1, Psi(PseudoAUV(), [], X0Hat, []).shape[0])))
    EstimateError.append(np.zeros((M, N + 1, X0Hat.shape[0])))
    Predictions.append(np.zeros((M, N + 1, X0Hat.shape[0])))
    Corrections.append(np.zeros((M, N + 1, X0Hat.shape[0])))

# samples calculation
for m in range(0, M):
    print('Sample path m=', m)
    X0 = sample_X0()

    models = []  # auv model for each filter
    Xs = []  # real position for each filter
    Ys = []  # observations for each filter
    XHats = []  # position estimate for each filter
    KHats = []  # estimate error covariance (or its estimate) for each filter
    XTildes = []
    XCorrs = []

    # do the same for every filter
    for k in range(0, len(filters)):
        # init a sample path
        models.append(PseudoAUV())
        Xs.append([X0])
        Ys.append([Psi(models[k], [], X0, [])])
        XHats.append([X0Hat])
        KHats.append([np.diag(DW)])
        XTildes.append([X0Hat])
        XCorrs.append([np.zeros_like(X0Hat)])

        # calculate a sample path and estimate step-by-step
        for i in range(0, N):
            x = Phi(models[k], [], Xs[k][-1], []) + np.random.normal(m_W, std_W)
            y = Psi(models[k], [], x, []) + np.random.normal(m_Nu, std_Nu)
            Xs[k].append(x)  # store the current position
            Ys[k].append(y)  # store the current position
            XHat_, KHat_, XTilde_, XCorr_ = filters[k].Step(models[k], i + 1, y, XHats[k][i], KHats[k][i])
            XHats[k].append(XHat_)  # store the current estimate
            KHats[k].append(KHat_)  # store the current estimate error covariance estimate
            XTildes[k].append(XTilde_)
            XCorrs[k].append(XCorr_)
        # calculate the estimate error and
        XHats[k] = np.array(XHats[k])
        Xs[k] = np.array(Xs[k])
        Ys[k] = np.array(Ys[k])
        Path[k][m, :, :] = Xs[k]
        Observations[k][m, :, :] = Ys[k]
        EstimateError[k][m, :, :] = Xs[k] - XHats[k]
        XTildes[k] = np.array(XTildes[k])
        XCorrs[k] = np.array(XCorrs[k])
        Predictions[k][m, :, :] = XTildes[k]
        Corrections[k][m, :, :] = XCorrs[k]
        # uncomment to save each path, estimate error and position deviation from the nominal path in separate files
        filename_path = os.path.join(
            path_trajectories,
            path_filename_template.replace('[filter]', names[k]).replace('[pathnum]', str(m).zfill(int(np.log10(M))))
        )
        np.savetxt(filename_path, Path[k][m, :, :], fmt='%f')
        filename_observations = os.path.join(
            path_observations,
            observations_filename_template.replace('[filter]', names[k]).replace('[pathnum]',
                                                                                 str(m).zfill(int(np.log10(M))))
        )
        np.savetxt(filename_observations, Observations[k][m, :, :], fmt='%f')
        filename_estimate = os.path.join(
            path_estimates,
            estimate_error_filename_template.replace('[filter]', names[k]).replace('[pathnum]',
                                                                                   str(m).zfill(int(np.log10(M))))
        )
        np.savetxt(filename_estimate, EstimateError[k][m, :, :], fmt='%f')
        np.savetxt(filename_estimate.replace('estimate_error', 'predict'), Predictions[k][m, :, :], fmt='%f')
        np.savetxt(filename_estimate.replace('estimate_error', 'correct'), Corrections[k][m, :, :], fmt='%f')

# calculate the mean and std for the estimate error and position deviation
# this may be done later by GatherStats.py script
mEstimateError = []
stdEstimateError = []
mPath = []
stdPath = []

for k in range(0, len(filters)):
    mEstimateError.append(np.mean(EstimateError[k], axis=0))
    stdEstimateError.append(np.std(EstimateError[k], axis=0))
    filename_estimate_mean = os.path.join(path_estimates,
                                          estimate_error_filename_template.replace('[filter]', names[k]).replace(
                                              '[pathnum]', 'mean'))
    filename_estimate_std = os.path.join(path_estimates,
                                         estimate_error_filename_template.replace('[filter]', names[k]).replace(
                                             '[pathnum]', 'std'))
    np.savetxt(filename_estimate_mean, mEstimateError[k], fmt='%f')
    np.savetxt(filename_estimate_std, stdEstimateError[k], fmt='%f')
    mPath.append(np.mean(Path[k], axis=0))
    stdPath.append(np.std(Path[k], axis=0))
    filename_path_mean = os.path.join(path_trajectories,
                                      path_filename_template.replace('[filter]', names[k]).replace('[pathnum]', 'mean'))
    filename_path_std = os.path.join(path_trajectories,
                                     path_filename_template.replace('[filter]', names[k]).replace('[pathnum]', 'std'))
    np.savetxt(filename_path_mean, mPath[k], fmt='%f')
    np.savetxt(filename_path_std, stdPath[k], fmt='%f')
