# Navigation
./_Navigation

AUV position estimation with seabed acoustic sensing and DOA measurements (extended Kalman filter and conditionnaly minimax nonlinear filter)

Setting for simulation experiments for the AUV position estimation [1] are defined in ./_Navigation/TestNavigation.py

The AUV dynamic model is provided in ./ControlledModel/AUV.py

The estimation is made by
- conditionnaly minimax nonlinear filter
- extended Kalman filter
- Kalman fiter for linear system with pseudo measurements

The measurements are the DOA of the acoustic signals from a set of beacons with known positions.

The prediction of all three filters may be defined 
- by virtue of the system (== the actual AUV speed is known)
- by means of acoustic seabed sensing algorithm from [2,3] (== external speed predictor)

# Tracking
./_Tracking

AUV position estimation with DOA and Doppler measurements (conditionnaly minimax nonlinear filter)

The dynamic model of an AUV performing a coordinate turn in a random plane and the observers, which provide the direction cosines and Doppler shift measurements is available in ./_Tracking/TrackingModel.py. The simulation settings are defined in the same file. The detailed description of the model can be found in [4].

The estimation in a static (AUV detection) is available in ./_Tracking/TestStatic.py. Available estimates:
- best linear estimate (conditionally minimax estimate on DOA measurements)
- least squares estimate given the observations of a single point
- Lasso regression estimate
- conditionally minimax estimate on combined DOA measurements and least squares estimate
- conditionally minimax estimate on combined DOA measurements and Lasso regression estimate

The estimation in a dynamic (AUV tracking) is avaliable ./_Tracking/TestTrackingFast.py. The version in ./_Tracking/TestTrackingSlow.py does not use precompiled prediction and correction functions and hence is much slower. Available filters/estimates:
- least squares estimate given the observations of a single point
- Lasso regression estimate
- Conditionally minimax nonlinear filter given the DOA measurements only
- Conditionally minimax nonlinear filter given the DOA and Doppler shift measurements
- Conditionally minimax nonlinear filter given the DOA measurements plus the least square estimate 
- Conditionally minimax nonlinear filter given the DOA measurements plus the Lasso regression estimate 
- Conditionally minimax nonlinear filter given the DOA/Doppler measurements plus the least square estimate 
- Conditionally minimax nonlinear filter given the DOA/Doppler measurements plus the Lasso regression estimate 


## Conditionnaly minimax nonlinear filter Python implementation

The CMNF filter is implemented in ./Filters/CMNFFilter.py 

Conditionnaly minimax nonlinear filter
for a nonlinear stchastic dicret-time model:

x(t) = Phi(t-1, x(t-1), xHat(t-1)) + W(t)   - state dynamics

y(t) = Psi(t, x(t)) + Nu(t)                 - observations

with 
- t in [0, N]
- W, N - Gaussian white noise with zero mean and covariances DW, DNu
- Xi, Zeta - basic prediction and correction functions, in general case can be chosen as follows:

Xi = Phi                                    - by virtue of the system

Zeta = y - Psi                              - residual

if the structure functions Phi, Psi can not be defined in the 
inline manner or require some history, an external object may be used: Phi = Phi(model, ...), Psi = Psi(model, ...)

./Filters/SimpleCMNFFilter.py version is less general and may be used for faster calcultaions

## Extended Kalman filter Python implementation
The Kalman filter is implemented in ./Filters/KalmanFilter.py

Extended Kalman filter
for a nonlinear stchastic dicrete-time model:

x(t) = Phi1(t-1, x(t-1), xHat(t-1)) + Phi2(t-1, x(t-1), xHat(t-1)) W(t)   - state dynamics

y(t) = Psi1(t, x(t)) + Psi1(t, x(t)) Nu(t)                                - observations

with 
- t in [0, N]
- W, N - Gaussian white noise with means MW, MNu and covariances DW, DNu

Extended Kalman filter requires the matrices of partial derivatives dPhi1/dx and dPsi1/dx

### Can be used for linear systems:

x(t) = Phi(t-1, xHat(t-1)) x(t-1) + Phi2(t-1, x(t-1), xHat(t-1)) W(t)

y(t) = Psi(t) x(t) + Psi1(t, x(t)) Nu(t),

then Phi1 = Phi * x, dPhi1/dx = Phi and Psi1 = Psi * x, dPsi1/dx = Psi

### Can be used for systems with linear pseudo measurements Y, which depend on real measurements y:

x(t) = Phi(t-1, xHat(t-1)) x(t-1) + Phi2(t-1, x(t-1), xHat(t-1)) W(t)

Y(t) = Psi(t, x(t), y(t)) x(t) + Psi1(t, x(t), y(t)) Nu(t)

if the structure functions Phi, Psi can not be defined in the 
inline manner or require some history, an external object may be used: Phi1,2 = Phi1,2(model, ...), Psi1,2 = Psi1,2(model, ...)

## References
[[1]](https://www.mdpi.com/1424-8220/19/24/5520) A. Miller, B. Miller, G. Miller, On AUV Control with the Aid of Position Estimation Algorithms Based on Acoustic Seabed Sensing and DOA Measurements // Sensors 2019, 19, 5520.
DOI: 10.3390/s19245520

[[2]](https://ieeexplore.ieee.org/document/8606561)	A. Miller, B. Miller, G. Miller, AUV navigation with seabed acoustic sensing // 2018 Australian & New Zealand Control Conference (ANZCC), Melbourne, Australia, 7-8 Dec. 2018
DOI: 10.1109/ANZCC.2018.8606561

[[3]](https://ieeexplore.ieee.org/document/8729708) A. Miller, B. Miller, G. Miller, AUV position estimation via acoustic seabed profile measurements // 2018 IEEE/OES Autonomous Underwater Vehicle Symposium - AUV 2018, University of Porto, Porto, Portugal, 6-9 Nov. 2018
DOI: 10.1109/AUV.2018.8729708

[[4]]() Tracking unpublished
