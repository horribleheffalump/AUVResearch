import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from Seabed import *

class SlopeApproximator():
    """Acoustic sensor model"""
    def __init__(self):
        degree = 5
        self.pf = PolynomialFeatures(degree)
        self.ridge = Ridge(fit_intercept=True, normalize=True)
        self.model = Pipeline([('poly', self.pf), ('ridge', self.ridge)])
        self.features = []
    def predict(self, X, y):
        self.model.fit(X, y)
        if len(self.features)  == 0:
            self.features = self.pf.get_feature_names()
            self.features.remove("1")
            self.x = np.array([a.count("x0") for a in self.features])
            self.y = np.array([a.count("x1") for a in self.features])
            self.xpows = np.array(list(map(max, zip(self.x, [sum([self.__tryParseInt(b.replace("x0^","")) for b in a.split(" ")]) for a in self.features]))))
            self.ypows = np.array(list(map(max, zip(self.y, [sum([self.__tryParseInt(b.replace("x1^","")) for b in a.split(" ")]) for a in self.features]))))
            self.xpows_1 = np.abs(self.xpows - 1)
            self.ypows_1 = np.abs(self.ypows - 1)

        #return self.model.predict(X) 
        #print(self.ridge.coef_)
    def partialdiffs(self, X):
        Xxpows = np.fromfunction(lambda i,j: X[i,0]**self.xpows[j], (X.shape[0], self.xpows.size), dtype=int)
        Xxpows_1 = np.fromfunction(lambda i,j: X[i,0]**self.xpows_1[j], (X.shape[0], self.xpows_1.size), dtype=int)
        Yypows = np.fromfunction(lambda i,j: X[i,1]**self.ypows[j], (X.shape[0], self.ypows.size), dtype=int)
        Yypows_1 = np.fromfunction(lambda i,j: X[i,1]**self.ypows_1[j], (X.shape[0], self.ypows_1.size), dtype=int)
        dzdx_gen = np.sum(self.x * self.ridge.coef_[1:] * self.xpows * Xxpows_1 * Yypows, axis=1)
        dzdy_gen = np.sum(self.y * self.ridge.coef_[1:] * self.ypows * Xxpows * Yypows_1, axis=1)
        #dzdx_prt2 = self.ridge.coef_[1] + 2.0 * self.ridge.coef_[3] * X[:,0] + self.ridge.coef_[4] * X[:,1]
        #dzdy_prt2 = self.ridge.coef_[2] + self.ridge.coef_[4] * X[:,0] + 2.0 * self.ridge.coef_[5] * X[:,1]
 
        return dzdx_gen, dzdy_gen
    def __tryParseInt(self, val):
        if val.isnumeric():
            return int(val)
        else:
            return 0