import numpy as np
from math import *
#from sklearn import linear_model as lm
from scipy.optimize import fsolve
from ControlledModel.Sensor import *


class AUV():
    """AUV model"""
    def __init__(self, T, delta, X0, DW, UNominal):
        self.X = np.array(X0)
        self.sigmaW = np.sqrt(np.array(DW))
        self.UNominal = UNominal
        self.t = 0.0
        self.delta = delta
        self.N = int(T / delta)
        self.delta_X = np.zeros(self.X.shape)

        self.t_history = np.arange(0.0, T+delta, delta)
        self.VNominal_history = np.array(list(map(lambda t: AUV.V(UNominal(t)), self.t_history[:-1])))
        self.deltaXNominal_history = np.vstack((self.delta_X, delta * self.VNominal_history))
        self.XNominal_history = X0 + np.cumsum(self.deltaXNominal_history, axis = 0)

        self.VReal_history = np.zeros((self.N+1, self.X.shape[0]))                
        self.XReal_history = np.zeros((self.N+1, self.X.shape[0]))
        self.XReal_estimate_history = np.zeros((self.N+1, self.X.shape[0]))
        
        self.k = 0
        self.VReal_history[self.k,:] = AUV.V(UNominal(0.0))
        self.XReal_history[self.k,:] = self.X 
        self.XReal_estimate_history[self.k,:] = self.X 

        self.ControlError = 0.0

        self.Sensors = []


    def addsensor(self, accuracy, Phi, Theta, seabed, estimateslope):      
        Gamma = np.pi / 2.0 - Phi
        self.Sensors.append(Sensor(self.X, self.X, self.UNominal(0.0), accuracy, Gamma, Theta, seabed, estimateslope))

    def step(self, XHat):

        dX, VReal = self.staterecalc(self.k, XHat) + self.sigmaW * np.array(np.random.normal(0,1,3))

        self.XReal_estimate_history[self.k,:] = XHat
        self.VReal_history[self.k,:] = VReal       
        self.X = self.X + dX

        for s in self.Sensors:
             s.step(self.X, self.UOptimal(self.k, self.X))
             
        self.t = self.t + self.delta
        self.k = self.k + 1
        self.XReal_history[self.k,:] = self.X 



        self.ControlError += 1.0 / self.N * np.linalg.norm(self.XNominal_history[self.k] - self.XReal_history[self.k])

    
    def UOptimal(self, k, X):
        shift = self.XNominal_history[k] - X
        Un = self.UNominal(k * self.delta)
        s0 = shift[0] + Un[2] * self.delta * cos(Un[0]) * cos(Un[1])
        s1 = shift[1] + Un[2] * self.delta * cos(Un[0]) * sin(Un[1])
        s2 = shift[2] + Un[2] * self.delta * sin(Un[0])
        theta = atan2(s1, s0)
        gamma = atan2(s2, s0 / cos(theta))
        
        v = np.linalg.norm(self.XNominal_history[k+1] - X) / self.delta 

        #gamma = atan2(s2, cos(theta) * s0 + sin(theta) * s1)
        return np.array([gamma, theta, v])
    
    def staterecalc(self, k, XHat):       
        U = self.UOptimal(k, XHat)
        V = AUV.V(U)
        #V = AUV.V(self.v, self.UOptimal(k, XHat))
        deltaX = self.delta * V
        return deltaX, V

    @staticmethod
    def V(U):
        return U[2] * np.array([np.cos(U[0]) * np.cos(U[1]), np.cos(U[0]) * np.sin(U[1]), np.sin(U[0])])
       