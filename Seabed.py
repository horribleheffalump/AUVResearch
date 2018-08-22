import numpy as np
class Seabed():
    """Defines seabead surface as a fuction of y,x coordinates""" 
    # z = a0 + sum_i(a_i sin (2 pi i x / Px) + b_i cos (2 pi i x / Px)) + sum_j(c_i sin (2 pi i y / Py) + d_i cos (2 pi i y / Py))  

    def __init__(self):
        self.a0 = -30.0
        self.Px = 20.0
        self.Py = 20.0
        self.a = np.random.normal(0,1,6) / [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        self.b = np.random.normal(0,1,6) / [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        self.c = np.random.normal(0,1,6) / [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        self.d = np.random.normal(0,1,6) / [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]

        self.sin_ix_coeff = np.reshape(np.fromfunction(lambda i: 2.0 * np.pi / self.Px * i, (self.a.size,), dtype = int), (self.a.size,1))
        self.cos_ix_coeff = np.reshape(np.fromfunction(lambda i: 2.0 * np.pi / self.Px * i, (self.b.size,), dtype = int), (self.b.size,1))
        self.sin_jy_coeff = np.reshape(np.fromfunction(lambda i: 2.0 * np.pi / self.Py * i, (self.c.size,), dtype = int), (self.c.size,1))
        self.cos_jy_coeff = np.reshape(np.fromfunction(lambda i: 2.0 * np.pi / self.Py * i, (self.d.size,), dtype = int), (self.d.size,1))


    def Z(self, X, Y):      
        sin_ix = np.array(list(map(lambda i: np.sin(self.sin_ix_coeff[i] * X), range(0, self.a.size))))
        cos_ix = np.array(list(map(lambda i: np.cos(self.cos_ix_coeff[i] * X), range(0, self.b.size))))
        sin_jy = np.array(list(map(lambda i: np.sin(self.sin_jy_coeff[i] * Y), range(0, self.c.size))))
        cos_jy = np.array(list(map(lambda i: np.cos(self.cos_jy_coeff[i] * Y), range(0, self.d.size))))        
        if len(X.shape) == 1:
            return self.a0 + np.sum(sin_ix * np.reshape(self.a, (self.a.size,1)), axis = 0) + np.sum(cos_ix * np.reshape(self.b, (self.b.size,1)), axis = 0) + np.sum(sin_jy * np.reshape(self.c, (self.c.size,1)), axis = 0) + np.sum(cos_jy * np.reshape(self.d, (self.d.size,1)), axis = 0)
        elif len(X.shape) == 2:
            return self.a0 + np.sum(sin_ix * np.reshape(self.a, (self.a.size,1,1)), axis = 0) + np.sum(cos_ix * np.reshape(self.b, (self.b.size,1,1)), axis = 0) + np.sum(sin_jy * np.reshape(self.c, (self.c.size,1,1)), axis = 0) + np.sum(cos_jy * np.reshape(self.d, (self.d.size,1,1)), axis = 0)
        else:
            raise NotImplemented()

    def dZ(self, X, Y):      
        cos_ix = self.sin_ix_coeff * np.array(list(map(lambda i: np.cos(self.sin_ix_coeff[i] * X), range(0, self.a.size))))
        sin_ix = self.cos_ix_coeff * np.array(list(map(lambda i: np.sin(self.cos_ix_coeff[i] * X), range(0, self.b.size))))
        cos_jy = self.sin_jy_coeff * np.array(list(map(lambda i: np.cos(self.sin_jy_coeff[i] * Y), range(0, self.c.size))))
        sin_jy = self.cos_jy_coeff * np.array(list(map(lambda i: np.sin(self.cos_jy_coeff[i] * Y), range(0, self.d.size))))        
        if len(X.shape) == 1:
            return np.sum(cos_ix * np.reshape(self.a, (self.a.size,1)), axis = 0) - np.sum(sin_ix * np.reshape(self.b, (self.b.size,1)), axis = 0), np.sum(cos_jy * np.reshape(self.c, (self.c.size,1)), axis = 0) - np.sum(sin_jy * np.reshape(self.d, (self.d.size,1)), axis = 0)
        else:
            raise NotImplemented()

    #def z(x,y):
    #    return -20 - 0.3 * np.sin(2.5 * x) - 0.3 * np.sin(2.5 * y) + 0.001 * y ** 2 + 0.001 * x ** 2
        #return -20 - 2.0 * x - 3.0 * y + 4.0 * x * y + 5.0 * x ** 2 + 6.0  * y ** 2
    #def Z(X):
    #    return np.reshape(-20 - 0.3 * np.sin(2.5 * X[:,0]) - 0.3 * np.sin(2.5 * X[:,1]) + 0.001 * X[:,0] ** 2 + 0.001 * X[:,1] ** 2, (X.shape[0], 1))
    #    #return -20 - 2.0 * X[:,0] - 3.0 * X[:,1] + 4.0 * X[:,0] * X[:,1] + 5.0 * X[:,0] ** 2 + 6.0 * X[:,1] ** 2
    #def Z(X,Y):
    #    return -20 - 0.3 * np.sin(2.5 * X) - 0.3 * np.sin(2.5 * Y) + 0.001 * X ** 2 + 0.001 * Y ** 2
    #    #return -20 - 2.0 * X - 3.0 * Y + 4.0 * np.multiply(X, Y) + 5.0 * X ** 2 + 6.0 * Y ** 2
    #def dzdx(x,y):
    #    return -0.3 * 2.5 * np.cos(2.5 * x) + 2.0 * 0.001 * x
    #    #return -2.0 + 4.0 * y + 2.0 * 5.0 * x
    #def dzdy(x,y):
    #    return -0.3 * 2.5 * np.cos(2.5 * y) + 2.0 * 0.001 * y
    #    #return -3.0 + 4.0 * x + 2.0 * 6.0 * y
    #def dz(x, y):
    #    return Seabed.dzdx(x,y), Seabed.dzdy(x,y)
    #def dZ(X):
    #    dzdx = -0.3 * 2.5 * np.cos(2.5 * X[:,0]) + 2.0 * 0.001 * X[:,0]
    #    dzdy = -0.3 * 2.5 * np.cos(2.5 * X[:,1]) + 2.0 * 0.001 * X[:,1]
    #    return dzdx, dzdy