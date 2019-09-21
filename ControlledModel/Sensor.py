import numpy as np
from sklearn import linear_model as lm
from scipy.optimize import fsolve
from Seabed.Profile import *
from Utils.SlopeApproximator import *

class Sensor():
    """Acoustic sensor model"""
    def __init__(self, X0, Xhat0, U0, accuracy, Gamma, Theta, seabed, estimateslope = True):
        self.accuracy = accuracy
        self.X_estimate = np.array(Xhat0)
        self.X_estimate_history = Xhat0      
        self.delta_X_estimate = np.array(self.X_estimate.shape) 
        self.delta_X_estimate_history = np.zeros(Xhat0.shape)      
        self.Gamma = np.array(Gamma)
        self.Theta = np.array(Theta)
        self.seabed = seabed
        self.estimateslope = estimateslope
        
        if (estimateslope):
            self.sa = SlopeApproximator()

        self.U_current = U0
        self.L_current = np.zeros(Gamma.size * Theta.size)
        _, self.L_current, _, _, _ = self.beamnet(X0, U0)


    def e(self, gamma, theta):
        return np.array([
            np.cos(gamma) * np.cos(theta), 
            np.cos(gamma) * np.sin(theta), 
            np.sin(gamma)
                        ])
    def de(self, gamma, dgamma, theta, dtheta):
        return np.array([
            -np.sin(gamma) * dgamma * np.cos(theta) - np.cos(gamma) * np.sin(theta) * dtheta, 
            -np.sin(gamma) * dgamma * np.sin(theta) + np.cos(gamma) * np.sin(theta) * dtheta, 
            np.cos(gamma) * dgamma
                        ])

    def __l(self, X, e):
        func = lambda l : self.seabed.Z(X[0] + l * e[:,0], X[1] + l * e[:,1]) - (X[2] + l * e[:,2])
        l_vec = fsolve(func, self.L_current)
        return l_vec + np.random.normal(0, self.accuracy, l_vec.size)
    def __beamcoords(self, X, e, L):
        return X + L * e

    def beamnet(self, X, U):

        e = np.transpose(np.reshape(np.fromfunction(lambda i, j: self.e(self.Gamma[i] + U[0], self.Theta[j] + U[1]), (self.Gamma.size, self.Theta.size), dtype=int), (3, self.Gamma.size * self.Theta.size)))
        L = self.__l(X, e) # we use real position in order to calculate the measurements
        R = np.reshape(L, (L.size, 1)) * e 
        # X+R - the points where the beams actually touch the seabed (if L is evaluated with zero noise)
        # hat{X}+R - the estimated points where the beams touch the seabed
        if self.estimateslope:
            # slope approximation
            self.sa.predict(R[:,0:2], R[:,2]) #, dZdx, dZdy)
            dZdx, dZdy = self.sa.partialdiffs(R[:,0:2])
        else:
            # exact slope
            # seems that this method is practically useless, since we are trying to evaluate the slope at points we've calculated using the estimate \hat{X}
            # and not at the place where the beams actually touch the seabed
            dZdx, dZdy = self.seabed.dZ((X+R)[:,0], (X+R)[:,1]) 
            # but if we substitute \hat{X} with real X, we may use the estimate evaluated with exact slope as an unachievable "ideal"
        M = dZdx * e[:,0] + dZdy * e[:,1] - e[:,2]  
    
        return R+X, L, dZdx, dZdy, M

    def step(self, X, U):
        dU = U - self.U_current
        self.U_current = U
        _, L, dZdx, dZdy, M = self.beamnet(X, U)
        deltaL = L - self.L_current
        de = np.transpose(np.reshape(np.fromfunction(lambda i, j: self.de(self.Gamma[i] + U[0], dU[0], self.Theta[j] + U[1], dU[1]), (self.Gamma.size, self.Theta.size), dtype=int), (3, self.Gamma.size * self.Theta.size)))
        A = np.transpose(np.vstack((-dZdx, -dZdy, np.ones(dZdx.shape))))
        B = deltaL * M + L * (dZdx * de[:,0] + dZdy * de[:,1] - de[:,2])        
        #unregularized least squares solution
        #V = np.dot(np.linalg.inv(np.dot(np.transpose(A),A)), np.dot(np.transpose(A),B)) 
        
        #ridge regression (regularized solution)
        reg = lm.LinearRegression()
        reg.fit(A, B)
        V = reg.predict(np.eye(3))
        self.delta_X_estimate = V 
        #self.delta_X_estimate = np.reshape(V,(1,V.size)) 
        self.X_estimate = self.X_estimate + self.delta_X_estimate 
        #print(self.delta_X_estimate)
        #print(self.X_estimate)
        self.X_estimate_history = np.vstack((self.X_estimate_history, self.X_estimate))
        self.delta_X_estimate_history = np.vstack((self.delta_X_estimate_history, self.delta_X_estimate))

        self.L_current[:] = L


    
