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
        features = pf.get_feature_names()
        features.remove("1")
        self.x = np.array([a.count("x0") for a in features])
        self.y = np.array([a.count("x1") for a in features])
        self.xpows = np.array(list(map(max, zip(self.x, [sum([tryParseInt(b.replace("x0^","")) for b in a.split(" ")]) for a in features]))))
        self.ypows = np.array(list(map(max, zip(self.y, [sum([tryParseInt(b.replace("x1^","")) for b in a.split(" ")]) for a in features]))))
        self.xpows_1 = np.abs(xpows - 1)
        self.ypows_1 = np.abs(ypows - 1)


    def predict(self, X, y):
        self.model.fit(X, y)
        #return self.model.predict(X) 
        #print(self.ridge.coef_)
    def partialdiffs(self, X):
        Xxpows = np.fromfunction(lambda i,j: X[i,0]**xpows[j], (X.shape[0], xpows.size), dtype=int)
        Xxpows_1 = np.fromfunction(lambda i,j: X[i,0]**xpows_1[j], (X.shape[0], xpows_1.size), dtype=int)
        Yypows = np.fromfunction(lambda i,j: X[i,1]**ypows[j], (X.shape[0], ypows.size), dtype=int)
        Yypows_1 = np.fromfunction(lambda i,j: X[i,1]**ypows_1[j], (X.shape[0], ypows_1.size), dtype=int)
        dfdx = self.x0 * self.ridge.coef_ * X[:,0]
        self.ridge.coef_[1] + 2.0 * self.ridge.coef_[3] * X[:,0] + self.ridge.coef_[4] * X[:,1], self.ridge.coef_[2] + self.ridge.coef_[4] * X[:,0] + 2.0 * self.ridge.coef_[5] * X[:,1]

        return  self.ridge.coef_[1] + 2.0 * self.ridge.coef_[3] * X[:,0] + self.ridge.coef_[4] * X[:,1], self.ridge.coef_[2] + self.ridge.coef_[4] * X[:,0] + 2.0 * self.ridge.coef_[5] * X[:,1]#, self.model.predict(X) 
    #, self.pf.get_feature_names(), self.ridge.coef_
