import numpy as np
from sklearn import linear_model as lm
from scipy.optimize import fsolve
from Seabed.Profile import *
from Utils.SlopeApproximator import *


class Sensor():
    """Acoustic sensor model"""

    def __init__(self, X0, Xhat0, accuracy, Phi, Theta, seabed, estimateslope=True):
        self.accuracy = accuracy
        self.X_estimate = np.array(Xhat0)
        self.X_estimate_history = Xhat0
        self.delta_X_estimate = np.array(self.X_estimate.shape)
        self.delta_X_estimate_history = np.zeros(Xhat0.shape)
        self.Phi = np.array(Phi)
        self.Theta = np.array(Theta)
        self.seabed = seabed
        self.estimateslope = estimateslope
        # self.L_net_previous = np.zeros((self.Phi.size, self.Theta.size))
        # self.L_net_current = np.zeros((self.Phi.size, self.Theta.size))
        # self.dzdx_current = np.zeros((self.Phi.size, self.Theta.size))
        # self.dzdy_current = np.zeros((self.Phi.size, self.Theta.size))
        # self.M_current = np.zeros((self.Phi.size, self.Theta.size))

        self.e = np.transpose(
            np.reshape(np.fromfunction(lambda i, j: self.__e(Phi[i], Theta[j]), (Phi.size, Theta.size), dtype=int),
                       (3, Phi.size * Theta.size)))
        # self.e = np.zeros((Phi.size * Theta.size, 3))

        # for i in range(Phi.size):
        #    for j in range(Theta.size):
        #        self.e[i*Phi.size  + j] = self.__e(Phi[i], Theta[j])

        if (estimateslope):
            self.sa = SlopeApproximator()

        self.L_current = np.zeros(Phi.size * Theta.size)
        _, self.L_current, _, _, _ = self.beamnet(X0)

    def __e(self, phi, theta):
        return np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            -np.cos(phi)])

    # def __l(self, X, e):
    #    func = lambda l : Seabed.z(X[0] + e[0] * l, X[1] + e[1] * l) - X[2] - e[2] * l
    #    self.L_current = fsolve(func, self.L_current)
    #    return self.L_current + np.random.normal(0,self.accuracy)
    def __l(self, X):
        func = lambda l: self.seabed.Z(X[0] + l * self.e[:, 0], X[1] + l * self.e[:, 1]) - (X[2] + l * self.e[:, 2])
        l_vec = fsolve(func, self.L_current)
        return l_vec + np.random.normal(0, self.accuracy, l_vec.size)

    def __beamcoords(self, X, e, L):
        return X + L * e

    # def beamnet(self, X):
    #    L = np.zeros((self.Phi.size, self.Theta.size))
    #    r = np.empty((self.Phi.size, self.Theta.size), dtype=np.ndarray)
    #    dzdx = np.zeros((self.Phi.size, self.Theta.size))
    #    dzdy = np.zeros((self.Phi.size, self.Theta.size))
    #    M = np.zeros((self.Phi.size, self.Theta.size))

    #    for i in range(self.Phi.size):
    #        for j in range(self.Theta.size):
    #            e = self.__e(self.Phi[i], self.Theta[j])
    #            L[i,j] = self.__l(X, e)
    #            r[i,j] = self.__beamcoords(X, e, L[i,j])
    #            dzdx[i,j], dzdy[i,j] = Seabed.dz(r[i,j][0], r[i,j][1])
    #            M[i,j] = dzdx[i,j] * e[0] + dzdy[i,j] * e[1] - e[2] 

    #    return r, L, dzdx, dzdy, M

    # def step(self, X):
    #    _, self.L_net_current, self.dzdx_current, self.dzdy_current, self.M_current = self.beamnet(X)
    #    deltaL = np.zeros((self.Phi.size, self.Theta.size))
    #    if (self.X_estimate_history.size > X.size): #not the first step
    #        deltaL = self.L_net_current - self.L_net_previous
    #    A = np.concatenate((
    #        -np.reshape(self.dzdx_current, (self.dzdx_current.size,1)), 
    #        -np.reshape(self.dzdy_current, (self.dzdy_current.size,1)), 
    #        np.ones((self.dzdy_current.size,1))
    #        ), axis=1)
    #    B = np.reshape(deltaL, (deltaL.size,1)) * np.reshape(self.M_current, (self.M_current.size,1)) 

    #    #unregularized least squares solution
    #    #V = np.dot(np.linalg.inv(np.dot(np.transpose(A),A)), np.dot(np.transpose(A),B)) 

    #    #ridge regression (regularized solution)
    #    reg = lm.Ridge()
    #    reg.fit(A, B)
    #    V = reg.predict(np.eye(3))

    #    self.X_estimate = self.X_estimate + np.reshape(V,(1,V.size))
    #    self.X_estimate_history = np.vstack((self.X_estimate_history, self.X_estimate))

    #    self.L_net_previous[:,:] = self.L_net_current[:,:]

    def beamnet(self, X):
        L = self.__l(X)  # we use real position in order to calculate the measurements
        R = np.reshape(L, (L.size, 1)) * self.e
        # X+R - the points where the beams actually touch the seabed (if L is evaluated with zero noise)
        # hat{X}+R - the estimated points where the beams touch the seabed
        if self.estimateslope:
            # slope approximation
            self.sa.predict(R[:, 0:2], R[:, 2])  # , dZdx, dZdy)
            dZdx, dZdy = self.sa.partialdiffs(R[:, 0:2])
        else:
            # exact slope
            # seems that this method is practically useless, since we are trying to evaluate the slope at points we've calculated using the estimate \hat{X}
            # and not at the place where the beams actually touch the seabed
            dZdx, dZdy = self.seabed.dZ((X + R)[:, 0], (X + R)[:, 1])
            # but if we substitute \hat{X} with real X, we may use the estimate evaluated with exact slope as an unachievable "ideal"
        M = dZdx * self.e[:, 0] + dZdy * self.e[:, 1] - self.e[:, 2]

        return R + X, L, dZdx, dZdy, M

    def step(self, X):
        _, L, dZdx, dZdy, M = self.beamnet(X)
        deltaL = L - self.L_current
        A = np.transpose(np.vstack((-dZdx, -dZdy, np.ones(dZdx.shape))))
        B = deltaL * M
        # unregularized least squares solution
        # V = np.dot(np.linalg.inv(np.dot(np.transpose(A),A)), np.dot(np.transpose(A),B))

        # ridge regression (regularized solution)
        reg = lm.LinearRegression()
        reg.fit(A, B)
        V = reg.predict(np.eye(3))
        self.delta_X_estimate = V
        # self.delta_X_estimate = np.reshape(V,(1,V.size))
        self.X_estimate = self.X_estimate + self.delta_X_estimate
        # print(self.delta_X_estimate)
        # print(self.X_estimate)
        self.X_estimate_history = np.vstack((self.X_estimate_history, self.X_estimate))
        self.delta_X_estimate_history = np.vstack((self.delta_X_estimate_history, self.delta_X_estimate))

        self.L_current[:] = L
