import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from Seabed import *

class SlopeApproximator():
    """Acoustic sensor model"""
    def __init__(self):
        degree = 2
        self.pf = PolynomialFeatures(degree)
        self.ridge = Ridge()
        self.model = Pipeline([('poly', self.pf), ('ridge', self.ridge)])
    def predict(self, X, y):
        self.model.fit(X, y)
        #return self.model.predict(X) 
        print(self.ridge.coef_)
    def partialdiffs(self, X):         
        return  self.ridge.coef_[1] + 2.0 * self.ridge.coef_[3] * X[:,0] + self.ridge.coef_[4] * X[:,1], self.ridge.coef_[2] + self.ridge.coef_[4] * X[:,0] + 2.0 * self.ridge.coef_[5] * X[:,1], self.model.predict(X) 
    #, self.pf.get_feature_names(), self.ridge.coef_
