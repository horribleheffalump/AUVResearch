import matplotlib.pyplot as plt
from matplotlib import gridspec

from ControlledModel.AUV import *
from Filters.CMNFFilter import *
from Filters.KalmanFilter import *
from math import *
#np.random.seed(2213123)
TT = 100.0
T = 5.0
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
#sigmaNu0 = np.tan(0.5 * np.pi / 180.0) # 0.5 degree
sigmaNu = sigmaNu0 * np.ones(2 * Xb.shape[0])
DNu = np.power(sigmaNu, 2.0)

dX = mX0 - Xb
pdX = dX[:,:-1]
sign_sin_phi = np.sign(pdX[:,1] / np.linalg.norm(pdX, axis = 1))
sign_cos_phi = np.sign(pdX[:,0] / np.linalg.norm(pdX, axis = 1))
sign_sin_lambda = np.sign(dX[:,2] / np.linalg.norm(dX, axis = 1))
sign_cos_lambda = np.sign(np.linalg.norm(pdX, axis = 1) / np.linalg.norm(dX, axis = 1))

def tan2sin(tan):
    sin = tan / np.sqrt(1.0 + tan * tan)
    return sin

def tan2cos(tan):
    cos = 1.0 / np.sqrt(1.0 + tan * tan)
    return cos

def obs2sincos(y):
    [tanphi, tanlambda] = np.split(y,2)
    sin_phi = np.abs(tan2sin(tanphi)) * sign_sin_phi
    cos_phi = np.abs(tan2cos(tanphi)) * sign_cos_phi
    sin_lambda = np.abs(tan2sin(tanlambda)) * sign_sin_lambda
    cos_lambda = np.abs(tan2cos(tanlambda)) * sign_cos_lambda
    return sin_phi, cos_phi, sin_lambda, cos_lambda 

pc=28.0
pr=20.0
tr=20.0

NBeams = 8
#accuracy = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
#PhiBounds = np.array([[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr],[pc-pr,pc+pr]])
#ThetaBounds = np.array([[10.0+tr,10.0-tr],   [100.0+tr,100.0-tr],  [190.0+tr,190.0-tr],  [280.0+tr,280.0-tr], [55.0+tr,55.0-tr],   [145.0+tr,145.0-tr],  [235.0+tr,235.0-tr],  [325.0+tr,325.0-tr]])
accuracy = np.array([0.1, 0.1])
PhiBounds = np.array([[pc-pr,pc+pr],[pc-pr,pc+pr]])
ThetaBounds = np.array([[10.0+tr,10.0-tr],[100.0+tr,100.0-tr]])
#accuracy = np.array([0.1])
#PhiBounds = np.array([[pc-pr,pc+pr]])
#ThetaBounds = np.array([[10.0+tr,10.0-tr]])
seabed = Profile()

def createAUV(X0):
    auv = AUV(T, delta, X0, DW, U)
    for i in range(0, accuracy.size):
        ph = PhiBounds[i,:]
        th = ThetaBounds[i,:]
        PhiGrad = np.append(np.arange(ph[0], ph[1], (ph[1] - ph[0]) / NBeams), ph[1])
        ThetaGrad = np.append(np.arange(th[0], th[1], (th[1] - th[0]) / NBeams), th[1])
        auv.addsensor(accuracy[i], PhiGrad / 180.0 * np.pi, ThetaGrad / 180.0 * np.pi, seabed, estimateslope = True)
    return auv

def Psi1(auv, k, X, y):
    tanphi = (X[1] - Xb[:,1]) / (X[0] - Xb[:,0])
    tanlambda = (X[2] - Xb[:,2]) / np.sqrt((X[0] - Xb[:,0]) * (X[0] - Xb[:,0]) + (X[1] - Xb[:,1]) * (X[1] - Xb[:,1]))
    return np.hstack((tanphi,tanlambda)) #np.array([tanphi, tanlambda])

def dPsi1(auv, k, X, y):
    d_tanphi_dX = -(X[1] - Xb[:,1]) / np.power(X[0] - Xb[:,0],2)
    d_tanphi_dY = 1.0 / (X[0] - Xb[:,0])
    d_tanphi_dZ = np.zeros_like(Xb[:,2])
    d_tanlambda_dX = -(X[2] - Xb[:,2]) * (X[0] - Xb[:,0]) / np.power(np.power(X[0] - Xb[:,0], 2) + np.power(X[1] - Xb[:,1], 2), 1.5)
    d_tanlambda_dY = -(X[2] - Xb[:,2]) * (X[1] - Xb[:,1]) / np.power(np.power(X[0] - Xb[:,0], 2) + np.power(X[1] - Xb[:,1], 2), 1.5)
    d_tanlambda_dZ = 1.0 / np.sqrt(np.power(X[0] - Xb[:,0], 2) + np.power(X[1] - Xb[:,1], 2))
    return np.vstack((np.array([d_tanphi_dX, d_tanphi_dY, d_tanphi_dZ]).T, np.array([d_tanlambda_dX, d_tanlambda_dY, d_tanlambda_dZ]).T));

def Psi2(auv, k, X, y):
    return np.eye(2 * Xb.shape[0])

def Psi1Pseudo(auv,k,X,y):
    sin_phi, cos_phi, sin_lambda, cos_lambda = obs2sincos(y)
    return np.vstack((np.array([-sin_phi, cos_phi, np.zeros(sin_phi.shape[0])]).T,np.array([-sin_lambda, np.zeros(sin_phi.shape[0]), cos_phi*cos_lambda]).T)) @ X

def dPsi1Pseudo(auv,k,X,y):
    sin_phi, cos_phi, sin_lambda, cos_lambda = obs2sincos(y)
    return np.vstack((np.array([-sin_phi, cos_phi, np.zeros(sin_phi.shape[0])]).T,np.array([-sin_lambda, np.zeros(sin_phi.shape[0]), cos_phi*cos_lambda]).T))

def Psi2Pseudo(auv,k,X,y):
    sin_phi, cos_phi, sin_lambda, cos_lambda = obs2sincos(y)
    return np.hstack(((X0[0] - Xb[:,0]) * cos_phi, (X0[0] - Xb[:,0]) * cos_lambda));


def Angles(X):
    return Psi1([], [], X, []) + Psi2([], [], X, []) @ sigmaNu * np.array(np.random.normal(0.0,1.0, DNu.shape[0]))

def PseudoMeasurements(y):
    sin_phi, cos_phi, sin_lambda, cos_lambda = obs2sincos(y)
    return np.hstack((Xb[:,1] * cos_phi - Xb[:,0] * sin_phi, Xb[:,2] * cos_phi*cos_lambda - Xb[:,0] * sin_lambda));


def Phi1(auv, k, X, XHat = []):
    if (len(XHat)) == 0:
        XHat = X
    if (len(auv.Sensors) > 0):
        #for acoustic
        deltaX =  np.mean(list(map(lambda x: x.delta_X_estimate, auv.Sensors)), axis=0)
    else:  
        #by virtue of the system
        deltaX, _ = auv.staterecalc(k, XHat)
    return X + deltaX

def dPhi1(auv, k, X):
    return np.eye(mX0.shape[0])

def Phi2(auv, k, X):   
    return np.eye(mX0.shape[0])

def Xi(auv, k, XHat):
    if (len(auv.Sensors) > 0):
        #for CMNF acoustic
        X = XHat + np.mean(list(map(lambda x: x.delta_X_estimate, auv.Sensors)), axis=0)
        return X
    else:  
        #for CMNF with predict by virtue of the system
        return Phi1(auv, k, XHat, XHat) + Phi2(auv, k, XHat) @ sigmaW

def Zeta(auv, k, X, y):
    return y - Psi1(auv, k, X, y) - Psi2(auv, k, X, y) @ sigmaNu



Mtrain = 1000

X0all = np.array(list(map(lambda i: mX0 + sigmaW * np.array(np.random.normal(0,1,3)), range(0, Mtrain) )))
auvs = np.array(list(map(lambda i: createAUV(X0all[i]), range(0, Mtrain) )))

cmnf = CMNFFilter(Phi1, Psi1, DW, DNu, Xi, Zeta)
cmnf.EstimateParameters(auvs, X0all, mX0, N, Mtrain)
cmnf.SaveParameters("D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\acoustic\\_[param].npy")
#cmnf.LoadParameters("D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\acoustic\\_[param].npy")

kalman = KalmanFilter(Phi1, dPhi1, Phi2, Psi1, dPsi1, Psi2, np.array([0.0,0.0,0.0]), np.diag(DW), np.zeros(2 * Xb.shape[0]), np.diag(DNu))

pseudo = KalmanFilter(Phi1, dPhi1, Phi2, Psi1Pseudo, dPsi1Pseudo, Psi2Pseudo, np.array([0.0,0.0,0.0]), np.diag(DW), np.zeros(2 * Xb.shape[0]), np.diag(DNu))

M = 1000

#filters = [cmnf, kalman, pseudo]
#needsPseudoMeasurements = [False, False, True]
#colors = ['red', 'green', 'blue']
#names = ['cmnf', 'kalman', 'pseudo']

filters = [cmnf]
needsPseudoMeasurements = [False]
colors = ['red']
names = ['cmnf']


EstimateError = [None] * len(filters)
ControlError = [None] * len(filters)

EstimateErrorFileNameTemplate = "D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\acoustic\\estimate\\estimate_error_[filter]_[pathnum].txt"
ControlErrorFileNameTemplate = "D:\\Наука\\_Статьи\\__в работе\\2019 - Sensors - Navigation\\data\\acoustic\\control\\control_error_[filter]_[pathnum].txt"

for k in range(0, len(filters)):
    EstimateError[k] = np.zeros((M, N+1, mX0.shape[0]))
    ControlError[k] = np.zeros((M, N+1, mX0.shape[0]))

for m in range(0,M):
    print('Sample path m=', m)
    X0 = mX0 + sigmaW * np.array(np.random.normal(0,1,3))   
    
    auvs = [None] * len(filters)
    Xs = [None] * len(filters)
    XHats = [None] * len(filters)
    KHats = [None] * len(filters)

    for k in range(0, len(filters)):
        auvs[k] = createAUV(X0)
        Xs[k] = [X0]
        XHats[k] = [mX0]
        KHats[k] = [np.diag(DW)]
        for i in range(0, N):
            auvs[k].step(XHats[k][i])   
            y = Angles(auvs[k].X)
            Xs[k].append(auvs[k].X)
            if needsPseudoMeasurements[k]:
                Y = PseudoMeasurements(y)
                XHat_, KHat_ = filters[k].Step(auvs[k], i+1, y, XHats[k][i], KHats[k][i], Y)
            else:
                XHat_, KHat_ = filters[k].Step(auvs[k], i+1, y, XHats[k][i], KHats[k][i])
            XHats[k].append(XHat_)
            KHats[k].append(KHat_)
        XHats[k] = np.array(XHats[k])
        Xs[k] = np.array(Xs[k])
        EstimateError[k][m,:,:] = Xs[k] - XHats[k]
        ControlError[k][m,:,:] = Xs[k] - XNominal_history
        np.savetxt(EstimateErrorFileNameTemplate.replace('[filter]',names[k]).replace('[pathnum]', str(m).zfill(int(np.log10(M)))),  EstimateError[k][m,:,:], fmt='%f')
        np.savetxt(ControlErrorFileNameTemplate.replace('[filter]',names[k]).replace('[pathnum]', str(m).zfill(int(np.log10(M)))),  ControlError[k][m,:,:], fmt='%f')

mEstimateError = [None] * len(filters)
stdEstimateError = [None] * len(filters)
mControlError = [None] * len(filters)
stdControlError = [None] * len(filters)

for k in range(0, len(filters)):
    mEstimateError[k] = np.mean(EstimateError[k], axis = 0)
    stdEstimateError[k] = np.std(EstimateError[k], axis = 0)
    np.savetxt(EstimateErrorFileNameTemplate.replace('[filter]',names[k]).replace('[pathnum]', 'mean'),  mEstimateError[k], fmt='%f')
    np.savetxt(EstimateErrorFileNameTemplate.replace('[filter]',names[k]).replace('[pathnum]', 'std'),  stdEstimateError[k], fmt='%f')

for k in range(0, len(filters)):
    mControlError[k] = np.mean(ControlError[k], axis = 0)
    stdControlError[k] = np.std(ControlError[k], axis = 0)
    np.savetxt(ControlErrorFileNameTemplate.replace('[filter]',names[k]).replace('[pathnum]', 'mean'),  mControlError[k], fmt='%f')
    np.savetxt(ControlErrorFileNameTemplate.replace('[filter]',names[k]).replace('[pathnum]', 'std'),  stdControlError[k], fmt='%f')


