import numpy as np
# from sklearn import linear_model as lm
from scipy.optimize import fsolve
from BasicModel.Sensor import *


class AUV():
    """AUV model"""
    def __init__(self, X0, V, delta):
        self.X = np.array(X0)
        self.V = V
        self.t = 0.0
        self.delta = delta
        self.delta_X = np.zeros(self.X.shape)
        self.delta_X_history = np.zeros(self.X.shape)
        self.delta_X_estimate_history = np.zeros(self.X.shape)
        self.X_history = self.X
        self.X_estimate_history = self.X
        self.V_history = self.V(0.0)
        self.t_history = self.t
        self.Sensors = []

    def addsensor(self, accuracy, Phi, Theta, seabed, estimateslope):      
        self.Sensors.append(Sensor(self.X, self.X, accuracy, Phi, Theta, seabed, estimateslope))
        # print(Phi, Theta)

    def step(self):
        V_cur = self.V(self.t) + 0.5 * self.delta * np.array([0.03, 0.04, 0.05]) * np.random.normal(0,1,3)
        self.delta_X = self.delta * V_cur 
        self.X = self.X + self.delta_X
        self.t = self.t + self.delta       
        for s in self.Sensors:
            s.step(self.X)
        self.X_history = np.vstack((self.X_history, self.X))
        self.delta_X_history = np.vstack((self.delta_X_history, self.delta_X))
        self.X_estimate_history = np.vstack((self.X_estimate_history, np.mean(list(map(lambda x: x.X_estimate, self.Sensors)), axis=0)))
        self.delta_X_estimate_history = np.vstack((self.delta_X_estimate_history, np.mean(list(map(lambda x: x.delta_X_estimate, self.Sensors)), axis=0)))
        self.V_history = np.vstack((self.V_history, V_cur))
        self.t_history = np.vstack((self.t_history, self.t))