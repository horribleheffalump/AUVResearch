"""
Setting for target tracking model 
"""

import numpy as np
from Filters.CMNFFilter import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path = "Z:\\Наука - Data\\2019 - Sensors - Tracking\\data\\"

# ########## AUV model definition ###################

T = 10.0  # simulation time limit
delta = 0.05  # simulation discretization step
N = int(T / delta)  # number of time instants after discretization

lamb = 0.95
nu = 0.095
mu = 1e-2

sigmaW = np.array([0.0] * (6 + 8 + 5))
sigmaW[6 + 7] = mu
sigmaW = sigmaW * np.sqrt(delta)
DW = np.power(sigmaW, 2.0)


class PseudoAUV:
    def step(self, XHat):
        # do nothing
        _ = XHat


def Phi(model, k, X, XHat):
    # state transformation in the plane of the turn
    X_turned = X[6:14]
    alpha = X[14]
    beta = X[15]
    R0 = X[16:19]
    [x_, y_, z_, vx_, vy_, vz_, phi_, an_] = X_turned
    v_ = np.linalg.norm(np.array([vx_, vy_, vz_]))
    dx = v_ * np.cos(phi_) * delta
    dy = v_ * np.sin(phi_) * delta
    dz = 0.0
    dvx = -an_ * np.sin(phi_) * delta
    dvy = an_ * np.cos(phi_) * delta
    dvz = 0.0
    dphi = an_ / v_ * delta
    dan = (-lamb * an_ + nu) * delta
    deltaX = np.array([dx, dy, dz, dvx, dvy, dvz, dphi, dan]) * delta
    x_orig, v_orig = toOriginalCoordinates(X_turned + deltaX, alpha, beta, R0)
    # original coordinates, coordinates in turned plane, alpha, beta, R0
    return np.hstack((x_orig, v_orig, X_turned + deltaX, X[14:19]))


# ########## Observation model definition ###################

def toOriginalCoordinates(X, alpha, beta, R0):
    X_ = X[0:3]
    V_ = X[3:6]
    A = np.array([[np.cos(beta), np.sin(alpha) * np.sin(beta), np.sin(beta) * np.cos(alpha)],
                  [1.0, np.cos(alpha), -np.sin(alpha)],
                  [-np.sin(beta), np.sin(alpha) * np.cos(beta), np.cos(beta) * np.cos(alpha)]])
    X = A.T @ X_ + R0
    V = A.T @ V_
    return X, V


# array of sensors' positions
Xb = np.array([[-10000.0, 0.0, 0.0], [-5000.0, 0.0, 0.0], [5000.0, 0.0, 0.0], [10000.0, 0.0, 0.0]])

# standard variation and the covariance matrix of the noise in observations
sigmaNu0 = np.sin(1 * np.pi / 180.0)  # ~1 degree
sigmaNu = np.array([sigmaNu0, sigmaNu0, sigmaNu0] * Xb.shape[0])
DNu = np.power(sigmaNu, 2.0)
omega0 = 20.0  # [Hz] frequency of sound signal
C = 1500.0  # [m/s] sound speed


def Psi(model, k, X, y):
    # observation transformation 
    # x, v = toOriginalCoordinates(X, model)
    x = X[0:3]
    v = X[3:6]
    R = np.sqrt((x[0] - Xb[:, 0]) * (x[0] - Xb[:, 0]) + (x[1] - Xb[:, 1]) * (x[1] - Xb[:, 1]) + (x[2] - Xb[:, 2]) * (
                x[2] - Xb[:, 2]))
    V = ((x[0] - Xb[:, 0]) * v[0] + (x[1] - Xb[:, 1]) * v[1] + (x[2] - Xb[:, 2]) * v[2]) / R
    theta = np.arcsin((x[2] - Xb[:, 2]) / R)
    psi = np.arctan2(x[0] - Xb[:, 0], x[1] - Xb[:, 1])
    xi = np.sin(theta) * np.cos(psi)
    nu = np.cos(theta)
    omega = omega0 / (1.0 - V / C)
    return np.hstack((xi, nu, omega))
    # return np.array[0.0]


# ########## generate a sample path ###################


model = PseudoAUV()

mX0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 7.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 20000.0, -1000.0])
sigmaX0 = np.array(
    [1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e2, 1e2, 1e2, 2.0, 2.0, 2.0, 1e-1, 1e-1, np.pi / 50.0, np.pi / 50.0, 1000,
     1000, 100])
X0 = mX0 + sigmaX0 * np.array(np.random.normal(0, 1, sigmaX0.shape[0]))
Xs = [X0]

for t in range(1, N + 1):
    x = Phi(model, t - 1, Xs[t - 1], []) + sigmaW * np.array(np.random.normal(0.0, 1.0, Xs[t - 1].shape[0]))
    Xs.append(x)  # store the current position

Xs = np.array(Xs)

# plot in the turn plane
fig = plt.figure(figsize=(10, 6), dpi=200)
ax = fig.gca()
ax.plot(Xs[:, 6], Xs[:, 7], color='red', linewidth=2.0)
plt.show()

# plot in the original coordinates
fig = plt.figure(figsize=(10, 6), dpi=200)
ax = fig.gca(projection='3d')
ax.plot(Xs[:, 0], Xs[:, 1], Xs[:, 2], color='red', linewidth=2.0)
plt.show()


# ########### FILTERS #########################

# CMNF filter definition

def Xi(model, k, XHat):
    # CMNF basic prediction
    return Phi(model, k, XHat, XHat)


def Zeta(model, k, X, y):
    # CMNF basic correction
    return y - Psi(model, k, X, y)


cmnf = CMNFFilter(Phi, Psi, DW, DNu, Xi, Zeta)

# uncomment if parameters are estimated anew
Mtrain = 10000  # number of sample paths for CMNF parameters estimation (train set)
X0all = np.array(list(map(lambda i: mX0 + sigmaX0 * np.array(np.random.normal(0, 1, sigmaX0.shape[0])),
                          range(0, Mtrain))))  # initial point for the training sample paths
models = np.array(list(map(lambda i: PseudoAUV(), range(0, Mtrain))))  # models for compatibility with CMNF
cmnf.EstimateParameters(models, X0all, mX0, N, Mtrain)
cmnf.SaveParameters(path + "_[param].npy")

# uncomment to load precalculated parameters
# cmnf.LoadParameters(path + "_[param].npy")


# ########### estimation and control samples calculation ##############

M = 10000  # number of samples

# set of filters for position estimation, their names and do they need the pseudomeasurements

filters = [cmnf]
names = ['cmnf']
needsPseudoMeasurements = [False]

# initialization

Path = [None] * len(filters)  # array to store path samples
EstimateError = [None] * len(filters)  # array to store position estimation error samples

PathFileNameTemplate = path + "path\\path_[filter]_[pathnum].txt"
EstimateErrorFileNameTemplate = path + "estimate\\estimate_error_[filter]_[pathnum].txt"

for k in range(0, len(filters)):
    Path[k] = np.zeros((M, N + 1, mX0.shape[0]))
    EstimateError[k] = np.zeros((M, N + 1, mX0.shape[0]))

# samples calculation
for m in range(0, M):
    print('Sample path m=', m)
    X0 = mX0 + sigmaX0 * np.array(np.random.normal(0, 1, sigmaX0.shape[0]))
    XHat0 = mX0 + np.array(np.random.normal(0, 1, sigmaX0.shape[0]))

    models = [None] * len(filters)  # auv model for each filter
    Xs = [None] * len(filters)  # real position for each filter
    XHats = [None] * len(filters)  # position estimate for each filter
    KHats = [None] * len(filters)  # estimate error covariance (or its estimate) for each filter

    # do the same for every filter
    for k in range(0, len(filters)):
        # init a sample path
        models[k] = PseudoAUV()
        Xs[k] = [X0]
        XHats[k] = [XHat0]
        KHats[k] = [np.diag(DW)]

        # calculate a sample path and estimate step-by-step
        for i in range(0, N):
            x = Phi(models[k], [], Xs[k][-1], []) + sigmaW * np.array(np.random.normal(0.0, 1.0, DW.shape[0]))
            y = Psi(models[k], [], x, []) + sigmaNu * np.array(np.random.normal(0.0, 1.0, DNu.shape[0]))
            Xs[k].append(x)  # store the current position
            XHat_, KHat_ = filters[k].Step(models[k], i + 1, y, XHats[k][i], KHats[k][i])
            XHats[k].append(XHat_)  # store the current estimate
            KHats[k].append(KHat_)  # store the current estimate error covariance estimate
        # calculate the estimate error and 
        XHats[k] = np.array(XHats[k])
        Xs[k] = np.array(Xs[k])
        Path[k][m, :, :] = Xs[k]
        EstimateError[k][m, :, :] = Xs[k] - XHats[k]
        # uncomment to save each path, estimate error and position deviation from the nominal path in separate files 
        np.savetxt(
            PathFileNameTemplate.replace('[filter]', names[k]).replace('[pathnum]', str(m).zfill(int(np.log10(M)))),
            Path[k][m, :, :], fmt='%f')
        np.savetxt(EstimateErrorFileNameTemplate.replace('[filter]', names[k]).replace('[pathnum]',
                                                                                       str(m).zfill(int(np.log10(M)))),
                   EstimateError[k][m, :, :], fmt='%f')

# calculate the mean and std for the estimate error and position deviation
# this may be done later by GatherStats.py script
mEstimateError = [None] * len(filters)
stdEstimateError = [None] * len(filters)
mPath = [None] * len(filters)
stdPath = [None] * len(filters)

for k in range(0, len(filters)):
    mEstimateError[k] = np.mean(EstimateError[k], axis=0)
    stdEstimateError[k] = np.std(EstimateError[k], axis=0)
    np.savetxt(EstimateErrorFileNameTemplate.replace('[filter]', names[k]).replace('[pathnum]', 'mean'),
               mEstimateError[k], fmt='%f')
    np.savetxt(EstimateErrorFileNameTemplate.replace('[filter]', names[k]).replace('[pathnum]', 'std'),
               stdEstimateError[k], fmt='%f')
    mPath[k] = np.mean(Path[k], axis=0)
    stdPath[k] = np.std(Path[k], axis=0)
    np.savetxt(PathFileNameTemplate.replace('[filter]', names[k]).replace('[pathnum]', 'mean'), mPath[k], fmt='%f')
    np.savetxt(PathFileNameTemplate.replace('[filter]', names[k]).replace('[pathnum]', 'std'), stdPath[k], fmt='%f')
