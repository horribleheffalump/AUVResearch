"""
Setting for simulation experiments for the AUV position estimation [1]
The AUV dynamic model is provided in AUV.py
The estimation is made by
- conditionnaly minimax nonlinear filter
- extended Kalman filter
- Kalman fiter for linear system with pseud measurements
The measurements are the DOA of the acoustic signals from a set of beacons with known positions
The prediction of all tree filters may be defined by virtue of the system (== the actual AUV speed is known)
or by means of acoustic seabed sensing algorithm from [2,3] (== external speed predictor)

[1] to be announced :)
[2]	A. Miller, B. Miller, G. Miller, AUV navigation with seabed acoustic sensing // 2018 Australian & New Zealand Control Conference (ANZCC), Melbourne, Australia, 7-8 Dec. 2018
https://ieeexplore.ieee.org/document/8606561 
DOI: 10.1109/ANZCC.2018.8606561
[3] A. Miller, B. Miller, G. Miller, AUV position estimation via acoustic seabed profile measurements // 2018 IEEE/OES Autonomous Underwater Vehicle Symposium - AUV 2018, University of Porto, Porto, Portugal, 6-9 Nov. 2018
https://ieeexplore.ieee.org/document/8729708 
DOI: 10.1109/AUV.2018.8729708

The CMNF filter is defined in CMNFFilter.py
The Kalman filter is defined in KalmanFilter.py
"""

import matplotlib.pyplot as plt
from matplotlib import gridspec

from ControlledModel.AUV import *
from Filters.CMNFFilter import *
from Filters.KalmanFilter import *
from math import *
#np.random.seed(2213123)


# path to save results at 
# [TODO]: use os package, not the path+file concatenation!!!
path = "Z:\\Наука - Data\\2019 - Sensors - AUV\\data\\acoustic_new\\"

########### AUV model definition ###################

TT = 100.0              # defines the nominal path oscilation periods
T = 300.0               # simulation time limit
delta = 1.0             # simulation discretization step
N = int(T / delta)      # number of time instants after discretization

# standard variation and covariance matrix of the noise in system dynamics and of the initial condition
sigmaW = np.array([1.0,1.0,1.0])    
DW = np.power(sigmaW, 2.0)          

mX0 = np.array([0.001,-0.0002,-10.0003]) # initial condition mean
v = 2.0 # 5.0                            # constant AUV speed absolute value for nominal (desired) path calculation
U = lambda t: np.array([np.pi / 100.0 * np.cos(1.0 * np.pi * t / TT), np.pi / 3.0 * np.cos(4.0 * np.pi * t / TT), v]) # controls' sequences for the nominal path calculation
t_history = np.arange(0.0, T + delta, delta) # array of time instants
VNominal_history = np.array(list(map(lambda t: AUV.V(U(t)), t_history[:-1]))) # array of speed vectors at time instants t_history for the nominal path
deltaXNominal_history = np.vstack((np.zeros(mX0.shape), delta * VNominal_history))  # array of position increments at time instants t_history for the nominal path
XNominal_history = mX0 + np.cumsum(deltaXNominal_history, axis = 0) # nominal path

# array of acoustic beacons positions
#Xb = np.array([[-100, 100, 0.0]])
Xb = np.array([[500, 100, 0.0], [500, -100, 0.0], [-100.0, 100.0, 0.0], [-100.0, -100.0, 0.0]])
#Xb = np.array([[500, 100, 0.0]])
#Xb = np.array([[maxX[0] + 100, maxX[1] + 100, 0.0], [maxX[0] + 100, minX[1] - 100, 0.0], [minX[0] - 100, maxX[1] + 100, 0.0], [minX[0] - 100, minX[1] - 100, 0.0]])
#Xb = np.array([[maxX[0] + 100, maxX[1] + 100, 0.0]])

# standard variation and the covariance matrix of the noise in observations
#sigmaNu0 = np.tan(5 * np.pi / 180.0 / 60.0) # ~5 arc minutes
sigmaNu0 = np.tan(0.5 * np.pi / 180.0) # ~0.5 degree
sigmaNu = sigmaNu0 * np.ones(2 * Xb.shape[0])
DNu = np.power(sigmaNu, 2.0)

# trigonometrix functions for pseudo measurements recalculation
def tan2sin(tan):
    sin = tan / np.sqrt(1.0 + tan * tan)
    return sin

def tan2cos(tan):
    cos = 1.0 / np.sqrt(1.0 + tan * tan)
    return cos

def obs2sincos(y,X):
    # we need X here just to calculate the proper signs of the sines and cosines
    dX = X - Xb
    pdX = dX[:,:-1]
    sign_sin_phi = np.sign(pdX[:,1] / np.linalg.norm(pdX, axis = 1))
    sign_cos_phi = np.sign(pdX[:,0] / np.linalg.norm(pdX, axis = 1))
    sign_sin_lambda = np.sign(dX[:,2] / np.linalg.norm(dX, axis = 1))
    sign_cos_lambda = np.sign(np.linalg.norm(pdX, axis = 1) / np.linalg.norm(dX, axis = 1))

    [tanphi, tanlambda] = np.split(y,2)
    sin_phi = np.abs(tan2sin(tanphi)) * sign_sin_phi
    cos_phi = np.abs(tan2cos(tanphi)) * sign_cos_phi
    sin_lambda = np.abs(tan2sin(tanlambda)) * sign_sin_lambda
    cos_lambda = np.abs(tan2cos(tanlambda)) * sign_cos_lambda
    return sin_phi, cos_phi, sin_lambda, cos_lambda 


######## acoustic sensors for seabed profile sensing based estimation ########

# each array member defines a square array of [NBeams by NBeams] sensors with certain accuracy.
# the angle bounds PhiBounds, ThetaBounds define the aiming angles of the sensors.
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

# the seabed profile
seabed = Profile()

# define the AUV
def createAUV(X0):
    auv = AUV(T, delta, X0, DW, U)
    # if the predicion is intended to be by virtue of the system, then the following loop has to be commented
    #for i in range(0, accuracy.size):
    #    ph = PhiBounds[i,:]
    #    th = ThetaBounds[i,:]
    #    PhiGrad = np.append(np.arange(ph[0], ph[1], (ph[1] - ph[0]) / NBeams), ph[1])
    #    ThetaGrad = np.append(np.arange(th[0], th[1], (th[1] - th[0]) / NBeams), th[1])
    #    auv.addsensor(accuracy[i], PhiGrad / 180.0 * np.pi, ThetaGrad / 180.0 * np.pi, seabed, estimateslope = True)
    return auv

######## structure functions of the system dynamics and observations   ########


def Psi1(auv, k, X, y):
    # original observation transformation 
    tanphi = (X[1] - Xb[:,1]) / (X[0] - Xb[:,0])
    tanlambda = (X[2] - Xb[:,2]) / np.sqrt((X[0] - Xb[:,0]) * (X[0] - Xb[:,0]) + (X[1] - Xb[:,1]) * (X[1] - Xb[:,1]))
    return np.hstack((tanphi,tanlambda)) #np.array([tanphi, tanlambda])

def dPsi1(auv, k, X, y):
    # original observation transformation Jacobian
    d_tanphi_dX = -(X[1] - Xb[:,1]) / np.power(X[0] - Xb[:,0],2)
    d_tanphi_dY = 1.0 / (X[0] - Xb[:,0])
    d_tanphi_dZ = np.zeros_like(Xb[:,2])
    d_tanlambda_dX = -(X[2] - Xb[:,2]) * (X[0] - Xb[:,0]) / np.power(np.power(X[0] - Xb[:,0], 2) + np.power(X[1] - Xb[:,1], 2), 1.5)
    d_tanlambda_dY = -(X[2] - Xb[:,2]) * (X[1] - Xb[:,1]) / np.power(np.power(X[0] - Xb[:,0], 2) + np.power(X[1] - Xb[:,1], 2), 1.5)
    d_tanlambda_dZ = 1.0 / np.sqrt(np.power(X[0] - Xb[:,0], 2) + np.power(X[1] - Xb[:,1], 2))
    return np.vstack((np.array([d_tanphi_dX, d_tanphi_dY, d_tanphi_dZ]).T, np.array([d_tanlambda_dX, d_tanlambda_dY, d_tanlambda_dZ]).T));


def Psi2(auv, k, X, y):
    # observations noise multiplier
    return np.eye(2 * Xb.shape[0])


def Psi1Pseudo(auv,k,X,y):
    # pseudo observations transformation
    sin_phi, cos_phi, sin_lambda, cos_lambda = obs2sincos(y, X)
    return np.vstack((np.array([-sin_phi, cos_phi, np.zeros(sin_phi.shape[0])]).T,np.array([-sin_lambda, np.zeros(sin_phi.shape[0]), cos_phi*cos_lambda]).T)) @ X


def dPsi1Pseudo(auv,k,X,y):
    # pseudo observations transformation Jacobian
    sin_phi, cos_phi, sin_lambda, cos_lambda = obs2sincos(y, X)
    return np.vstack((np.array([-sin_phi, cos_phi, np.zeros(sin_phi.shape[0])]).T,np.array([-sin_lambda, np.zeros(sin_phi.shape[0]), cos_phi*cos_lambda]).T))


def Psi2Pseudo(auv,k,X,y):
    # pseudo observations noise multiplier (depends on state!)
    sin_phi, cos_phi, sin_lambda, cos_lambda = obs2sincos(y, X)
    return np.diag(np.hstack((
        (X[0] - Xb[:,0]) * cos_phi,  
        (X[0] - Xb[:,0]) * cos_lambda
        )));

 
def Angles(X):
    # original observations
    return Psi1([], [], X, []) + Psi2([], [], X, []) @ sigmaNu * np.array(np.random.normal(0.0,1.0, DNu.shape[0]))

 
def PseudoMeasurements(y, X):
    # pseudo observations
    # sin_phi, cos_phi, sin_lambda, cos_lambda = obs2sincos(y, X)
    return np.hstack((Xb[:,1] * cos_phi - Xb[:,0] * sin_phi, Xb[:,2] * cos_phi*cos_lambda - Xb[:,0] * sin_lambda));


def Phi(auv, k, X, XHat):
    # state transformation in the original system (calculated via AUV model)
    deltaX, _ = auv.staterecalc(k, XHat)
    return X + deltaX


def Phi1(auv, k, X):
    # state transformation for the prediction
    if (len(auv.Sensors) > 0):
        #for acoustic
        deltaX =  np.mean(list(map(lambda x: x.delta_X_estimate, auv.Sensors)), axis=0)
    else:  
        #by virtue of the system
        deltaX, _ = auv.staterecalc(k, X)
    return X + deltaX

def dPhi1(auv, k, X):
    # state transformation Jacobian
    return np.eye(mX0.shape[0])

def Phi2(auv, k, X):   
    # state nois multiplier
    return np.eye(mX0.shape[0])

def Xi(auv, k, XHat):
    # CMNF basic prediction
    return Phi1(auv, k, XHat)

def Zeta(auv, k, X, y):
    # CMNF basic correction
    return y - Psi1(auv, k, X, y)



############ FILTERS #########################

# CMNF filter definition

#cmnf = CMNFFilter(Phi, Psi1, DW, DNu, Xi, Zeta)

# uncomment if parameters are estimated anew
#Mtrain = 1000 # number of sample paths for CMNF parameters estimation (train set)
#X0all = np.array(list(map(lambda i: mX0 + sigmaW * np.array(np.random.normal(0,1,3)), range(0, Mtrain) ))) # initial point for the training sample paths
#auvs = np.array(list(map(lambda i: createAUV(X0all[i]), range(0, Mtrain) )))  # AUV models for the training sample paths
#cmnf.EstimateParameters(auvs, X0all, mX0, N, Mtrain)
#cmnf.SaveParameters(path + "_[param].npy")

# uncomment to load precalculated parameters
#cmnf.LoadParameters(path + "_[param].npy")

# EKF definition

kalman = KalmanFilter(Phi1, dPhi1, Phi2, Psi1, dPsi1, Psi2, np.array([0.0,0.0,0.0]), np.diag(DW), np.zeros(2 * Xb.shape[0]), np.diag(DNu))

# pseudo measurements Kalman filter definition

pseudo = KalmanFilter(Phi1, dPhi1, Phi2, Psi1Pseudo, dPsi1Pseudo, Psi2Pseudo, np.array([0.0,0.0,0.0]), np.diag(DW), np.zeros(2 * Xb.shape[0]), np.diag(DNu))

############ estimation and control samples calculation ##############

M = 1000 # number of samples

# set of filters for position estimation, their names and do they need the pseudomeasurements

#filters = [cmnf, kalman, pseudo]
#names = ['cmnf', 'kalman', 'pseudo']
#needsPseudoMeasurements = [False, False, True]

filters = [kalman, pseudo] 
names = ['kalman', 'pseudo']
needsPseudoMeasurements = [False, True]

# initialization

Path = [None] * len(filters)  # array to store path samples
EstimateError = [None] * len(filters) # array to store position estimation error samples
ControlError = [None] * len(filters)  # array to store the differences between the nominal and actual positions
ControlErrorNorm = [None] * len(filters) # array to store the distance between the nominal and actual positions

PathFileNameTemplate = path + "path\\path_[filter]_[pathnum].txt"
EstimateErrorFileNameTemplate = path + "\\estimate\\estimate_error_[filter]_[pathnum].txt"
ControlErrorFileNameTemplate = path + "\\control\\control_error_[filter]_[pathnum].txt"


for k in range(0, len(filters)):
    Path[k] = np.zeros((M, N+1, mX0.shape[0]))
    EstimateError[k] = np.zeros((M, N+1, mX0.shape[0]))
    ControlError[k] = np.zeros((M, N+1, mX0.shape[0]))
    ControlErrorNorm[k] = np.zeros((M, N+1))

# samples calculation
for m in range(0,M):
    print('Sample path m=', m)
    X0 = mX0 + sigmaW * np.array(np.random.normal(0,1,3))   # initial position for sample m
    
    auvs = [None] * len(filters)    # auv model for each filter
    Xs = [None] * len(filters)      # real position for each filter
    XHats = [None] * len(filters)    # position estimate for each filter
    KHats = [None] * len(filters)   # estimate error covariance (or its estimate) for each filter

    # do the same for every filter
    for k in range(0, len(filters)):
        # init a sample path
        auvs[k] = createAUV(X0)
        Xs[k] = [X0]
        XHats[k] = [mX0]
        KHats[k] = [np.diag(DW)]

        # calculate a sample path and estimate step-by-step
        for i in range(0, N):
            auvs[k].step(XHats[k][i])   # real current position calculation
            y = Angles(auvs[k].X)       # observation calculation
            Xs[k].append(auvs[k].X)     # store the current position
            # calculate the position estimate
            if needsPseudoMeasurements[k]:
                # if the filter requires pseudomeasurements
                Y = PseudoMeasurements(y, XHats[k][i])
                XHat_, KHat_ = filters[k].Step(auvs[k], i+1, y, XHats[k][i], KHats[k][i], Y)
            else:
                # if the filter uses only original measurements
                XHat_, KHat_ = filters[k].Step(auvs[k], i+1, y, XHats[k][i], KHats[k][i])
            XHats[k].append(XHat_) # store the current estimate
            KHats[k].append(KHat_) # store the current estimate error covariance estimate
        # calculate the estimate error and 
        XHats[k] = np.array(XHats[k])
        Xs[k] = np.array(Xs[k])
        Path[k][m,:,:] = Xs[k]
        EstimateError[k][m,:,:] = Xs[k] - XHats[k]
        ControlError[k][m,:,:] = Xs[k] - XNominal_history
        ControlErrorNorm[k][m,:] = np.power(np.linalg.norm(Xs[k] - XNominal_history, axis = 1), 2)
        # uncomment to save each path, estimate error and position deviation from the nominal path in separate files 
        np.savetxt(PathFileNameTemplate.replace('[filter]',names[k]).replace('[pathnum]', str(m).zfill(int(np.log10(M)))),  Path[k][m,:,:], fmt='%f')
        np.savetxt(EstimateErrorFileNameTemplate.replace('[filter]',names[k]).replace('[pathnum]', str(m).zfill(int(np.log10(M)))),  EstimateError[k][m,:,:], fmt='%f')
        np.savetxt(ControlErrorFileNameTemplate.replace('[filter]',names[k]).replace('[pathnum]', str(m).zfill(int(np.log10(M)))),  ControlError[k][m,:,:], fmt='%f')

# calculate the mean and std for the estimate error and position deviation
# this may be done later by GatherStats.py script
mEstimateError = [None] * len(filters)
stdEstimateError = [None] * len(filters)
mControlError = [None] * len(filters)
stdControlError = [None] * len(filters)
mControlErrorNorm = [None] * len(filters)

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
    mControlErrorNorm[k] = np.max(np.mean(ControlErrorNorm[k], axis = 0))
    print(names[k], mControlErrorNorm[k])

