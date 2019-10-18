import matplotlib.pyplot as plt
from matplotlib import gridspec

from ControlledModel.AUV import *
from Filters.CMNFFilter import *
from math import *
#np.random.seed(2213123)
TT = 300.0
T = 300.0
delta = 1.0
N = int(T / delta)

sigmaW = np.array([1.0,1.0,1.0])
DW = np.power(sigmaW, 2.0)

mX0 = np.array([0.001,-0.0002,-10.0003]) 
v = 2.0 # 5.0
U = lambda t: np.array([np.pi / 100.0 * np.cos(1.0 * np.pi * t / TT), np.pi / 3.0 * np.cos(4.0 * np.pi * t / TT), v])
t_history = np.arange(0.0, T + delta, delta)
VNominal_history = np.array(list(map(lambda t: AUV.V(U(t)), t_history[:-1])))
deltaXNominal_history = np.vstack((np.zeros(mX0.shape), delta * VNominal_history))
XNominal_history = mX0 + np.cumsum(deltaXNominal_history, axis = 0)

maxX = np.max(XNominal_history, axis = 0)
minX = np.min(XNominal_history, axis = 0)
Xb = np.array([[maxX[0] + 10, maxX[1] + 10, 0.0], [maxX[0] + 10, minX[1] - 10, 0.0], [minX[0] - 10, maxX[1] + 10, 0.0], [minX[0] - 10, minX[1] - 10, 0.0]])

sigmaNu0 = np.tan(5 * np.pi / 180.0 / 60.0) # 5 arc minutes
sigmaNu = sigmaNu0 * np.ones(2 * Xb.shape[0])
DNu = np.power(sigmaNu, 2.0)

pc=28.0
pr=20.0
tr=20.0

NBeams = 8
#accuracy = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
#PhiBounds = np.array([[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr]])
#ThetaBounds = np.array([[10.0+tr,10.0-tr],   [100.0+tr,100.0-tr],  [190.0+tr,190.0-tr],  [280.0+tr,280.0-tr], [55.0+tr,55.0-tr],   [145.0+tr,145.0-tr],  [235.0+tr,235.0-tr],  [325.0+tr,325.0-tr]])
#accuracy = np.array([0.1, 0.1])
#PhiBounds = np.array([[pc-pr,pc+pr],[pc-pr,pc+pr]])
#ThetaBounds = np.array([[10.0+tr,10.0-tr],[100.0+tr,100.0-tr]])
accuracy = np.array([0.1])
PhiBounds = np.array([[pc-pr,pc+pr]])
ThetaBounds = np.array([[10.0+tr,10.0-tr]])
seabed = Profile()

def createAUV(X0):
    auv = AUV(T, delta, X0, DW, U)
    #for i in range(0, accuracy.size):
    #    ph = PhiBounds[i,:]
    #    th = ThetaBounds[i,:]
    #    PhiGrad = np.append(np.arange(ph[0], ph[1], (ph[1] - ph[0]) / NBeams), ph[1])
    #    ThetaGrad = np.append(np.arange(th[0], th[1], (th[1] - th[0]) / NBeams), th[1])
    #    auv.addsensor(accuracy[i], PhiGrad / 180.0 * np.pi, ThetaGrad / 180.0 * np.pi, seabed, estimateslope = True)
    return auv

def Psi(auv,k,X,y):
    tanphi = (X[1] - Xb[:,1]) / (X[0] - Xb[:,0])
    tanlambda = (X[2] - Xb[:,2]) / np.sqrt((X[0] - Xb[:,0]) * (X[0] - Xb[:,0]) + (X[1] - Xb[:,1]) * (X[1] - Xb[:,1]))
    return np.hstack((tanphi,tanlambda)) #np.array([tanphi, tanlambda])

def Angles(X):
    return Psi([], [], X, []) + sigmaNu * np.array(np.random.normal(0.0,1.0, DNu.shape[0]))

def Phi(auv, k, X, XHat):
    deltaX, _ = auv.staterecalc(k, XHat)
    return X + deltaX

def Xi(auv, k, XHat):
    if (len(auv.Sensors) > 0):
        #for CMNF acoustic
        X = XHat + np.mean(list(map(lambda x: x.delta_X_estimate, auv.Sensors)), axis=0)
        return X
    else:  
        #for CMNF with predict by virtue of the system
        return Phi(auv, k, XHat, XHat)

def Zeta(auv, k, X, y):
    return y - Psi(auv, k, X, y)



M = 100

X0all = np.array(list(map(lambda i: mX0 + sigmaW * np.array(np.random.normal(0,1,3)), range(0, M) )))
auvs = np.array(list(map(lambda i: createAUV(X0all[i]), range(0, M) )))

cmnf = CMNFFilter(Phi, Psi, DW, DNu, Xi, Zeta)
#cmnf.EstimateParameters(auvs, X0all, mX0, N, M)
#cmnf.SaveParameters("D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\[param].npy")
cmnf.LoadParameters("D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\[param].npy")

M = 3

EstimateError = np.zeros((M,N+1,mX0.shape[0]))
ControlError = np.zeros((M,N+1,mX0.shape[0]))

for m in range(0,M):
    print('Estimate m=', m)
    X0 = mX0 + sigmaW * np.array(np.random.normal(0,1,3))   
    auv = createAUV(X0)
    X = [X0]
    XHat = [mX0]
    for i in range(0, N):
        auv.step(XHat[i])   
        y = Angles(auv.X)
        X.append(auv.X)
        #XHat.append(cmnf.Step(auv, i, y, XHat[i]))
        XHat_, _ = cmnf.Step(auv, i+1, y, XHat[i], [])
        XHat.append(XHat_)
        #EstimateError += 1.0 / N * np.linalg.norm(auv.X - XHat[i + 1])
    XHat = np.array(XHat)
    X = np.array(X)
    EstimateError[m,:,:] =  X - XHat
    ControlError[m,:,:] =   X - auv.XNominal_history

MEstimateError = np.mean(EstimateError, axis = 0)
stdEstimateError = np.std(EstimateError, axis = 0)

MControlError = np.mean(ControlError, axis = 0)
stdControlError = np.std(ControlError, axis = 0)

f = plt.figure(num=None, figsize=(15,5), dpi=200, facecolor='w', edgecolor='k')
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])     
gs.update(left=0.03, bottom=0.05, right=0.99, top=0.99, wspace=0.1, hspace=0.1)    


for k in range(0,3):
    ax = plt.subplot(gs[k])
    ax.plot(auv.t_history, MEstimateError[:,k], color='black', linewidth=2.0)
    ax.plot(auv.t_history, stdEstimateError[:,k], color='red', linewidth=2.0)
plt.show()

for k in range(0,3):
    ax = plt.subplot(gs[k])
    ax.plot(auv.t_history, MControlError[:,k], color='black', linewidth=2.0)
    ax.plot(auv.t_history, stdControlError[:,k], color='red', linewidth=2.0)
plt.show()

for k in range(0,3):
    ax = plt.subplot(gs[k])
    #ax.plot(auv.t_history, auv.XNominal_history[:,k], color='grey',
    #linewidth=4.0)
    #ax.plot(auv.t_history, auv.XReal_history[:,k], color='black',
    #linewidth=2.0)
    ax.plot(auv.t_history, auv.XReal_history[:,k] - auv.XNominal_history[:,k], color='black', linewidth=2.0)
    #ax.plot(auv.t_history, XHat[:,k], color='red', linewidth=2.0)
    #ax.plot(auv.t_history, auv.XReal_estimate_history[:,k], color='red',
    #linewidth=2.0)
plt.show()

for k in range(0,3):
    ax = plt.subplot(gs[k])
    #ax.plot(auv.t_history, auv.XNominal_history[:,k], color='grey',
    #linewidth=4.0)
    ax.plot(auv.t_history, auv.XReal_history[:,k] - XHat[:,k], color='black', linewidth=2.0)
    #ax.plot(auv.t_history, XHat[:,k], color='red', linewidth=2.0)
    #ax.plot(auv.t_history, auv.XReal_estimate_history[:,k], color='red',
    #linewidth=2.0)
plt.show()


for k in range(0,3):
    ax = plt.subplot(gs[k])
    ax.plot(auv.t_history, auv.XNominal_history[:,k], color='grey', linewidth=4.0)
    ax.plot(auv.t_history, auv.XReal_history[:,k], color='black', linewidth=2.0)
    #ax.plot(auv.t_history, XHat[:,k], color='red', linewidth=2.0)
    #ax.plot(auv.t_history, auv.XReal_estimate_history[:,k], color='red', linewidth=2.0)
plt.show()




#print(auv.ControlError)
#print(EstimateError)

#f = plt.figure(num=None, figsize=(15,5), dpi=200, facecolor='w', edgecolor='k')
#gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])     
#gs.update(left=0.03, bottom=0.05, right=0.99, top=0.99, wspace=0.1, hspace=0.1)    

#for k in range(0,2):
#    ax = plt.subplot(gs[k])
#    ax.plot(auv.t_history, Obs[:,k], color='blue', linewidth=4.0)
#plt.show()


#plt.savefig(path + 'pathsample.png')

