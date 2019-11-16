## Conditionnaly minimax nonlinear filter Python implementation

The CMNF filter is implemented in Filters/CMNFFilter.py

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

## Extended Kalman filter Python implementation
The Kalman filter is implemented in Filters/KalmanFilter.py

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
