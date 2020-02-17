"""Setting for target tracking model
"""

import numpy as np
from numba import jit

T = 100.0  # simulation time limit
delta = 0.1  # simulation discretization step
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


@jit(cache=True)
def phi(X):
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
    return np.hstack((x_orig, x_turned_new, np.array([alpha, beta]), R0))
# ########## Observation model definition ###################


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


@jit(cache=True)
def psi(X):
    # observation transformation
    _x = X[0:3]
    _sphere = cart2sphere(_x)
    _angles = _sphere[:2 * Xb.shape[0]]
    _R = _sphere[-Xb.shape[0]:]
    _v = X[3:6]
    _V = ((_x[0] - Xb[:, 0]) * _v[0] + (_x[1] - Xb[:, 1]) * _v[1] + (_x[2] - Xb[:, 2]) * _v[2]) / _R
    _omega = omega0 / (1.0 - _V / C)
    return np.hstack((_angles, _omega))


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


@jit(cache=True)
def generate_sample_path(n):
    """
    Generates a sample path
    :param n: path length
    :returns x, y: state and observations sample path
    """
    x = np.zeros((n + 1, m_W.shape[0]))
    y = np.zeros((n + 1, m_Nu.shape[0]))
    # w = np.random.normal(self.m_w, self.std_w, (n + 1, self.m_w.shape[0]))
    # nu = np.random.normal(self.m_nu, self.std_nu, (n + 1, self.m_nu.shape[0]))
    x[0, :] = sample_X0()
    y[0, :] = psi(x[0, :]) + sample_normal(m_Nu, std_Nu)
    for i in range(0, n):
        x[i + 1, :] = phi(x[i, :]) + sample_normal(m_W, std_W)
        y[i + 1, :] = psi(x[i + 1, :]) + sample_normal(m_Nu, std_Nu)
    return x, y


@jit(cache=True)
def generate_sample_paths(m, n):
    """
    Generates a sample path
    :param n: path length
    :returns x, y: state and observations sample path
    """
    x = np.zeros((m, n + 1, m_W.shape[0]))
    y = np.zeros((m, n + 1, m_Nu.shape[0]))
    # w = np.random.normal(self.m_w, self.std_w, (n + 1, self.m_w.shape[0]))
    # nu = np.random.normal(self.m_nu, self.std_nu, (n + 1, self.m_nu.shape[0]))
    for j in range(0, m):
        x[j, 0, :] = sample_X0()
        y[j, 0, :] = psi(x[j, 0, :]) + sample_normal(m_Nu, std_Nu)
        for i in range(0, n):
            x[j, i + 1, :] = phi(x[j, i, :]) + sample_normal(m_W, std_W)
            y[j, i + 1, :] = psi(x[j, i + 1, :]) + sample_normal(m_Nu, std_Nu)
    return x, y


@jit(cache=True)
def xi(x):
    # CMNF basic prediction
    return phi(x)


@jit(cache=True)
def zeta(x, y):
    return y - psi(x)

