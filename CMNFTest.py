import matplotlib.pyplot as plt
from matplotlib import gridspec

from ControlledModel.AUV import *
from Filters.CMNFFilter import *
from math import *
#np.random.seed(2213123)

TT = 360.0
T = 360.0
delta = 1.0
N = int(T / delta)

X0 = np.array([0.001 + 0.1 * np.random.normal(0,1),-0.0002 + 0.1 * np.random.normal(0,1),-10.0003 + 0.1 * np.random.normal(0,1)])
DW = np.array([0.5,0.5,0.5])
U = lambda t: np.pi * np.array([1 / 100.0 * np.cos(1.0 * np.pi * t / TT), 1 / 3.0 * np.cos(4.0 * np.pi * t / TT)])
#U = lambda t: np.array([np.pi/2 - np.pi  / 100.0 * np.cos(1.0 * np.pi * t / TT), np.pi / 3.0 * np.cos(4.0 * np.pi * t / TT)])
v = 2.0 # 5.0
t_history = np.arange(0.0, T + delta, delta)
VNominal_history = np.array(list(map(lambda t: AUV.V(v, U(t)), t_history[:-1])))
deltaXNominal_history = np.vstack((np.zeros(X0.shape), delta * VNominal_history))
XNominal_history = X0 + np.cumsum(deltaXNominal_history, axis = 0)

auv = AUV(T, delta, X0, DW, U, v)
#test = testenvironmentControlled(T, delta, NBeams, accuracy, PhiBounds, ThetaBounds, auv, seabed, estimateslope)



def Psi(k,X):
    Xb = [100,40,0]
    tanphi = (X[1] - Xb[1]) / (X[0] - Xb[0])
    tanlambda = (X[2] - Xb[2]) / sqrt((X[0] - Xb[0]) * (X[0] - Xb[0]) + (X[1] - Xb[1]) * (X[1] - Xb[1]))
    return np.array([tanphi, tanlambda]) 

DNu = np.array([0.0003,0.0003])
def Angles(X):
    return Psi([],X) + DNu * np.array(np.random.normal(0.0,1.0,2))

def Phi(k, X, XHat):
    shift = XNominal_history[k] - XHat
    Un = U(k * delta)
    s0 = shift[0] + v * delta * cos(Un[0]) * cos(Un[1])
    s1 = shift[1] + v * delta * cos(Un[0]) * sin(Un[1])
    s2 = shift[2] + v * delta * sin(Un[0])
    #s0 = shift[0] + v * delta * sin(Un[0]) * cos(Un[1])
    #s1 = shift[1] + v * delta * sin(Un[0]) * sin(Un[1])
    #s2 = shift[2] + v * delta * cos(Un[0])
    theta = atan2(s1, s0)
    gamma = atan2(s2, cos(theta) * s0 + sin(theta) * s1)
    #gamma = atan2(cos(theta) * s0 + sin(theta) * s1, s2)
    return X + v * np.array([np.cos(gamma) * np.cos(theta), np.cos(gamma) * np.sin(theta), np.sin(gamma)])
    #return X + v * np.array([np.sin(gamma) * np.cos(theta), np.sin(gamma) * np.sin(theta), np.cos(gamma)])

def Xi(k, XHat):
    return Phi(k, XHat, XHat)

def Zeta(k, X, y):
    return y - Psi(k,X)


States = []
Observations = []

M = 1000
#for m in range(0, M):
#    auv = AUV(T, delta, X0, DW, U, v)
#    for i in range(0, N):
#        t = delta * i
#        auv.step(auv.X)
#    Obs = np.array(list(map(Angles, auv.XReal_history)))   
#    States.append(auv.XReal_history)
#    Observations.append(Obs)   
#    #x = [X0]
#    #for i in range(0, N):
#    #    x.append(Xi(i, x[i]) + DW * np.array(np.random.normal(0.0,1.0,3)))
#    #x = np.array(x)
#    #Obs = np.array(list(map(Angles, x)))
#    #States.append(x)
#    #Observations.append(Obs)
#States = np.array(States)
#Observations = np.array(Observations)


cmnf = CMNFFilter(Phi, Psi, DW, DNu, Xi, Zeta)
cmnf.EstimateParameters(X0, X0, N, M)

auv = AUV(T, delta, X0, DW, U, v)
XHat = [X0]
for i in range(0, N):
    t = delta * i
    auv.step(XHat[i])
    #auv.step()
    y = Angles(auv.X)
    XHat.append(cmnf.Step(i, y, XHat[i]))

XHat = np.array(XHat)

f = plt.figure(num=None, figsize=(15,5), dpi=200, facecolor='w', edgecolor='k')
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])     
gs.update(left=0.03, bottom=0.05, right=0.99, top=0.99, wspace=0.1, hspace=0.1)    

for k in range(0,3):
    ax = plt.subplot(gs[k])
    ax.plot(auv.t_history, auv.XNominal_history[:,k], color='grey', linewidth=4.0)
    ax.plot(auv.t_history, auv.XReal_history[:,k], color='black', linewidth=2.0)
    #ax.plot(auv.t_history, XHat[:,k], color='red', linewidth=2.0)
    #ax.plot(auv.t_history, auv.XReal_estimate_history[:,k], color='red',
    #linewidth=2.0)
plt.show()

#f = plt.figure(num=None, figsize=(15,5), dpi=200, facecolor='w', edgecolor='k')
#gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])     
#gs.update(left=0.03, bottom=0.05, right=0.99, top=0.99, wspace=0.1, hspace=0.1)    

#for k in range(0,2):
#    ax = plt.subplot(gs[k])
#    ax.plot(auv.t_history, Obs[:,k], color='blue', linewidth=4.0)
#plt.show()


#plt.savefig(path + 'pathsample.png')

