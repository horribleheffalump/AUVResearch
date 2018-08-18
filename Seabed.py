import numpy as np
class Seabed():
    """Defines seabead surface as a fuction of y,x, coordinates """
    #def z(x,y):
    #    return 0.4 * np.sin(8.0 * y) + 0.05 * np.cos(25.0 * y) + 0.5 * np.cos(6.0 * x) + 0.08 * np.sin(20.0 * x) + 0.001 * y ** 2 + 0.002 * x ** 2
    #def dzdx(x,y):
    #    return -0.5 * 6.0 * np.sin(6.0 * x) + 0.08 * 20.0 * np.cos(20.0 * x) + 2.0 * 0.002 * x
    #def dzdy(x,y):
    #    return 0.4 * 8.0 * np.cos(8.0 * y) - 0.05 * 25.0 * np.sin(25.0 * y) + 2.0 * 0.001 * y
    #def dz(x, y):
    #    return Seabed.dzdx(x,y), Seabed.dzdy(x,y)
    def z(x,y):
        return -20 - 0.3 * np.sin(2.5 * x) - 0.3 * np.sin(2.5 * y) + 0.001 * y ** 2 + 0.001 * x ** 2
        #return -20 - 2.0 * x - 3.0 * y + 4.0 * x * y + 5.0 * x ** 2 + 6.0  * y ** 2
    def Z(X):
        return np.reshape(-20 - 0.3 * np.sin(2.5 * X[:,0]) - 0.3 * np.sin(2.5 * X[:,1]) + 0.001 * X[:,0] ** 2 + 0.001 * X[:,1] ** 2, (X.shape[0], 1))
        #return -20 - 2.0 * X[:,0] - 3.0 * X[:,1] + 4.0 * X[:,0] * X[:,1] + 5.0 * X[:,0] ** 2 + 6.0 * X[:,1] ** 2
    def ZZ(X,Y):
        return -20 - 0.3 * np.sin(2.5 * X) - 0.3 * np.sin(2.5 * Y) + 0.001 * X ** 2 + 0.001 * Y ** 2
        #return -20 - 2.0 * X - 3.0 * Y + 4.0 * np.multiply(X, Y) + 5.0 * X ** 2 + 6.0 * Y ** 2
    def dzdx(x,y):
        return -0.3 * 2.5 * np.cos(2.5 * x) + 2.0 * 0.001 * x
        #return -2.0 + 4.0 * y + 2.0 * 5.0 * x
    def dzdy(x,y):
        return -0.3 * 2.5 * np.cos(2.5 * y) + 2.0 * 0.001 * y
        #return -3.0 + 4.0 * x + 2.0 * 6.0 * y
    def dz(x, y):
        return Seabed.dzdx(x,y), Seabed.dzdy(x,y)
    def dZ(X):
        return np.reshape(-0.3 * 2.5 * np.cos(2.5 * X[:,0]) + 2.0 * 0.001 * X[:,0], (X.shape[0], 1)), np.reshape(-0.3 * 2.5 * np.cos(2.5 * X[:,1]) + 2.0 * 0.001 * X[:,1], (X.shape[0], 1))
