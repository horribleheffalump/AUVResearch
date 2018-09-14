import numpy as np
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class SlopeApproximator():
    """Acoustic sensor model"""
    def __init__(self):
        degree = 4
        self.pf = PolynomialFeatures(degree)
        self.ridge = LinearRegression(fit_intercept=True, normalize=True, )
        self.model = Pipeline([('poly', self.pf), ('ridge', self.ridge)])
        self.features = []
    def predict(self, X, y):#, dzdx, dzdy):
        self.model.fit(X, y)
        if len(self.features)  == 0:
            self.features = self.pf.get_feature_names()
            self.features.remove("1")
            self.x = np.array([a.count("x0") for a in self.features])
            self.y = np.array([a.count("x1") for a in self.features])
            self.xpows = np.array(list(map(max, zip(self.x, [sum([self.__tryParseInt(b.replace("x0^","")) for b in a.split(" ")]) for a in self.features]))))
            self.ypows = np.array(list(map(max, zip(self.y, [sum([self.__tryParseInt(b.replace("x1^","")) for b in a.split(" ")]) for a in self.features]))))
            self.xpows_1 = np.maximum(self.xpows - 1, np.zeros(self.xpows.shape))
            self.ypows_1 = np.maximum(self.ypows - 1, np.zeros(self.ypows.shape))
            #print(self.features)
            #print(self.x)
            #print(self.y)
            #print(self.xpows)
            #print(self.ypows)
            #print(self.xpows_1)
            #print(self.ypows_1)
            
            #fig = plt.figure(figsize=(12, 6), dpi=200)
            #gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])     
            #gs.update(left=0.03, bottom=0.05, right=0.99, top=0.99, wspace=0.1, hspace=0.1)    

            #ax = fig.add_subplot(gs[0], projection='3d')
            #ax.view_init(50, 30)
            #surf = ax.scatter(X[:,0], X[:,1], y, color = 'blue')
            #surf = ax.scatter(X[:,0], X[:,1], self.model.predict(X), color = 'red')

            #est_dzdx, est_dzdy = self.partialdiffs(X)

            #axdx = fig.add_subplot(gs[1], projection='3d')
            #axdx.view_init(50, 30)
            #surf = axdx.scatter(X[:,0], X[:,1], dzdx, color = 'blue')
            ##surf = axdx.scatter(X[:-40,0], X[:-40,1], est_dzdx[:-40], color = 'red')
            #surf = axdx.scatter(X[:,0], X[:,1], est_dzdx[:], color = 'red')

            #axdy = fig.add_subplot(gs[2], projection='3d')
            #axdy.view_init(50, 30)
            #surf = axdy.scatter(X[:,0], X[:,1], dzdy, color = 'blue')
            #surf = axdy.scatter(X[:,0], X[:,1], est_dzdy, color = 'red')
                
            #plt.show()
        #return self.model.predict(X) 
        #print(self.ridge.coef_)
    def partialdiffs(self, X):
        Xxpows = np.fromfunction(lambda i,j: (self.x*X[i,0])**self.xpows[j], (X.shape[0], self.xpows.size), dtype=int)
        Xxpows_1 = np.fromfunction(lambda i,j: (self.x*X[i,0])**self.xpows_1[j], (X.shape[0], self.xpows_1.size), dtype=int)
        Yypows = np.fromfunction(lambda i,j: (self.y*X[i,1])**self.ypows[j], (X.shape[0], self.ypows.size), dtype=int)
        Yypows_1 = np.fromfunction(lambda i,j: (self.y*X[i,1])**self.ypows_1[j], (X.shape[0], self.ypows_1.size), dtype=int)
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