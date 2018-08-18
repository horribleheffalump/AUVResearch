import numpy as np
from sklearn import linear_model as lm
from scipy.optimize import fsolve
from Sensor import *


class AUV():
    """AUV model"""
    def __init__(self, X0, V, delta):
        self.X = np.array(X0)
        self.V = V
        self.t = 0.0
        self.delta = delta
        self.X_history = self.X
        self.X_estimate_history = self.X
        self.t_history = self.t
        self.Sensors = []
    def addsensor(self, accuracy, Phi, Theta):      
        self.Sensors.append(Sensor(self.X, accuracy, Phi, Theta))
    def step(self):
        for s in self.Sensors:
            s.stepvec(self.X)
        self.X = self.X + self.delta * self.V(self.t) #+ np.sqrt(0.1 * self.delta) * np.random.normal(0,1,3)
        self.t = self.t + self.delta
        
        self.X_history = np.vstack((self.X_history, self.X))
        self.X_estimate_history = np.vstack((self.X_estimate_history, np.mean(list(map(lambda x: x.X_estimate, self.Sensors)), axis=0)))
        self.t_history = np.vstack((self.t_history, self.t))


    