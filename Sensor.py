import numpy as np
from sklearn import linear_model as lm
from scipy.optimize import fsolve
from Seabed import *

class Sensor():
    """Acoustic sensor model"""
    def __init__(self, Xhat0, accuracy, Phi, Theta):
        self.accuracy = accuracy
        self.X_estimate = np.array(Xhat0)
        self.X_estimate_history = Xhat0      
        self.L_current = 0.0
        self.Phi = np.array(Phi)
        self.Theta = np.array(Theta)
        self.L_net_previous = np.zeros((self.Phi.size, self.Theta.size))
        self.L_net_current = np.zeros((self.Phi.size, self.Theta.size))
        self.dzdx_current = np.zeros((self.Phi.size, self.Theta.size))
        self.dzdy_current = np.zeros((self.Phi.size, self.Theta.size))
        self.M_current = np.zeros((self.Phi.size, self.Theta.size))
    def __e(self, phi, theta):
        return np.array([
            np.sin(phi)*np.cos(theta), 
            np.sin(phi)*np.sin(theta), 
            -np.cos(phi)])
    def __l(self, X, e):
        func = lambda l : Seabed.z(X[0] + e[0] * l, X[1] + e[1] * l) - X[2] - e[2] * l
        self.L_current = fsolve(func, self.L_current)
        return self.L_current + np.random.normal(0,self.accuracy)
    def __beamcoords(self, X, e, L):
        return X + L * e
    def beamnet(self, X):
        L = np.zeros((self.Phi.size, self.Theta.size))
        r = np.empty((self.Phi.size, self.Theta.size), dtype=np.ndarray)
        dzdx = np.zeros((self.Phi.size, self.Theta.size))
        dzdy = np.zeros((self.Phi.size, self.Theta.size))
        M = np.zeros((self.Phi.size, self.Theta.size))
        for i in range(self.Phi.size):
            for j in range(self.Theta.size):
                e = self.__e(self.Phi[i], self.Theta[j])
                L[i,j] = self.__l(X, e)
                r[i,j] = self.__beamcoords(X, e, L[i,j])
                dzdx[i,j], dzdy[i,j] = Seabed.dz(r[i,j][0], r[i,j][1])
                M[i,j] = dzdx[i,j] * e[0] + dzdy[i,j] * e[1] - e[2] 
        #np.fromfunction(lambda i, j: self.BeamCoords(self.X, 0,0), (len(self.Phi), len(self.Theta)), )
        return r, L, dzdx, dzdy, M
    def step(self, X):
        _, self.L_net_current, self.dzdx_current, self.dzdy_current, self.M_current = self.beamnet(X)
        deltaL = np.zeros((self.Phi.size, self.Theta.size))
        if (self.X_estimate_history.size > X.size): #not the first step
            deltaL = self.L_net_current - self.L_net_previous
        A = np.concatenate((
            -np.reshape(self.dzdx_current, (self.dzdx_current.size,1)), 
            -np.reshape(self.dzdy_current, (self.dzdy_current.size,1)), 
            np.ones((self.dzdy_current.size,1))
            ), axis=1)
        B = np.reshape(deltaL, (deltaL.size,1)) * np.reshape(self.M_current, (self.M_current.size,1)) 
        V = np.dot(np.linalg.inv(np.dot(np.transpose(A),A)), np.dot(np.transpose(A),B))
        
        #reg = lm.Ridge()
        #reg.fit(A, B)
        #V = reg.predict(np.eye(3))
        self.X_estimate = self.X_estimate + np.reshape(V,(1,V.size)) 
        self.X_estimate_history = np.vstack((self.X_estimate_history, self.X_estimate))

        self.L_net_previous[:,:] = self.L_net_current[:,:]


    
