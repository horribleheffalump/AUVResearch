import matplotlib.pyplot as plt
from matplotlib import gridspec

from Filters.KalmanFilter import *
from math import *
#np.random.seed(2213123)
T = 300.0
delta = 1.0
N = int(T / delta)

t_history = np.arange(0.0, T + delta, delta)

 
X0 = np.array([0.001,-0.0002,-10.0003])

def Psi1(auv,k,X,y):
    return X

def Psi2(auv,k,X,y):
    return np.eye(X0.shape[0])

def Phi1(auv, k, X):
    return X

def Phi2(auv, k, X):   
    return np.eye(X0.shape[0])


DW =  np.array([1.0,1.0,1.0])
sigmaW = np.sqrt(DW)

DNu = 1.0 * np.array([1.0,1.0,1.0])
sigmaNu = np.sqrt(DNu)
kalman = KalmanFilter(Phi1, lambda m, k, x: np.eye(X0.shape[0]), Phi2, Psi1, lambda m, k, x, y: np.eye(X0.shape[0]), Psi2, np.array([0.0,0.0,0.0]), np.diag(DW), np.array([0.0,0.0,0.0]), np.diag(DNu))

M = 100

EstimateError = np.zeros((M,N+1,X0.shape[0]))
ControlError = np.zeros((M,N+1,X0.shape[0]))

KHat = []
X = []
XHat = []
for m in range(0,M):
    print('Estimate m=', m)
    X = [X0]
    XHat = [X0]
    KHat = [np.diag(DW)]
    for i in range(0, N):
        X.append(Phi1([],[],X[i]) + Phi2([],[],X[i]) @ sigmaW * np.array(np.random.normal(0,1,3)))
        y = Psi1([],[],X[i+1],[])  + Psi2([],[],X[i+1],[]) @  sigmaNu * np.array(np.random.normal(0,1,3))
        XHat_, KHat_ = kalman.Step([], i+1, y, XHat[i], KHat[i])
        XHat.append(XHat_)
        KHat.append(KHat_)
    XHat = np.array(XHat)
    KHat = np.array(KHat)
    X = np.array(X)
    EstimateError[m,:,:] =  X - XHat

MEstimateError = np.mean(EstimateError, axis = 0)
stdEstimateError = np.std(EstimateError, axis = 0)

f = plt.figure(num=None, figsize=(15,5), dpi=200, facecolor='w', edgecolor='k')
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])     
gs.update(left=0.03, bottom=0.05, right=0.99, top=0.99, wspace=0.1, hspace=0.1)    


for k in range(0,3):
    ax = plt.subplot(gs[k])
    ax.plot(t_history, MEstimateError[:,k], color='black', linewidth=2.0)
    ax.plot(t_history, stdEstimateError[:,k], color='red', linewidth=2.0)
    ax.plot(t_history, np.sqrt(KHat)[:,k,k], color='blue', linewidth=2.0)
plt.show()

for k in range(0,3):
    ax = plt.subplot(gs[k])
    ax.plot(t_history, X[:,k], color='black', linewidth=2.0)
    ax.plot(t_history, XHat[:,k], color='red', linewidth=2.0)
plt.show()

