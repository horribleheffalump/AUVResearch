import numpy as np
from math import *
#from sklearn import linear_model as lm
from scipy.optimize import fsolve
from Sensor import *


class AUVControlled():
    """AUV model"""
    def __init__(self, T, delta, X0, DW, UNominal, v):
        self.X = np.array(X0)
        self.sigmaW = np.sqrt(np.array(DW))
        self.UNominal = UNominal
        self.v = v
        self.t = 0.0
        self.delta = delta
        N = int(T / delta)
        self.delta_X = np.zeros(self.X.shape)

        self.t_history = np.arange(0.0, T+delta, delta)
        self.VNominal_history = np.array(list(map(lambda t: AUVControlled.V(v, UNominal(t)), self.t_history[:-1])))
        self.deltaXNominal_history = np.vstack((self.delta_X, delta * self.VNominal_history))
        self.XNominal_history = X0 + np.cumsum(self.deltaXNominal_history, axis = 0)

        self.VReal_history = np.zeros((N, self.X.shape[0]))                
        self.XReal_history = np.zeros((N+1, self.X.shape[0]))
        self.XReal_estimate_history = np.zeros((N+1, self.X.shape[0]))
        
        self.k = 0
        self.VReal_history[self.k,:] = AUVControlled.V(v, UNominal(0.0))
        self.XReal_history[self.k,:] = self.X 
        self.XReal_estimate_history[self.k,:] = self.X 


        self.Sensors = []

    def addsensor(self, accuracy, Phi, Theta, seabed, estimateslope):      
        self.Sensors.append(Sensor(self.X, self.X, accuracy, Phi, Theta, seabed, estimateslope))
        #print(Phi, Theta)

    def step(self):
        #print(self.Sensors[0].X_estimate)
        XRealEstimate = np.mean(list(map(lambda x: x.X_estimate, self.Sensors)), axis=0)
        #print(self.XNominal_history[self.k])
        #print(XRealEstimate)
        Uopt = AUVControlled.UOptimal(self.XNominal_history[self.k] - XRealEstimate, self.v, self.UNominal(self.t), self.delta)
        VReal = AUVControlled.V(self.v, Uopt)
        self.VReal_history[self.k,:] = VReal 
        self.delta_X = self.delta * VReal + self.sigmaW * np.array(np.random.normal(0,1,3))
        self.X = self.X + self.delta_X
        for s in self.Sensors:
             s.step(self.X)


        self.t = self.t + self.delta
        self.k = self.k + 1
        self.XReal_history[self.k,:] = self.X 
        self.XReal_estimate_history[self.k,:] = XRealEstimate

    @staticmethod
    def UOptimal(shift, v, UNominal, delta):
        #print(UNominal, shift)
        s0 = shift[0] + v * delta * cos(UNominal[0]) * cos(UNominal[1])
        s1 = shift[1] + v * delta * cos(UNominal[0]) * sin(UNominal[1])
        s2 = shift[2] + v * delta * sin(UNominal[0])
        theta = atan2(s1, s0)
        gamma = atan2(s2, cos(theta) * s0 + sin(theta) * s1)
        return np.array([gamma, theta])
    @staticmethod
    def V(v, U):
        return v * np.array([np.cos(U[0]) * np.cos(U[1]), np.cos(U[0]) * np.sin(U[1]), np.sin(U[0])])
        