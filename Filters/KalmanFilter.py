
import numpy as np

class KalmanFilter():
    """
    Extended Kalman filter
    for a nonlinear stchastic dicret-time model:
    x(t) = Phi1(t-1, x(t-1), xHat(t-1)) + Phi2(t-1, x(t-1), xHat(t-1)) W(t)   - state dynamics
    y(t) = Psi1(t, x(t)) + Psi1(t, x(t)) Nu(t)                                - observations
    with t in [0, N]
    W, N - Gaussian white noise with means MW, MNu and covariances DW, DNu
    requires the matrices of partial derivatives dPhi1/dx and dPsi1/dx

    Can be used for linear systems:
    x(t) = Phi(t-1, xHat(t-1)) x(t-1) + Phi2(t-1, x(t-1), xHat(t-1)) W(t)
    y(t) = Psi(t) x(t) + Psi1(t, x(t)) Nu(t),
    then Phi1 = Phi * x, dPhi1/dx = Phi and Psi1 = Psi * x, dPsi1/dx = Psi

    Can be used for systems with linear pseudo measurements Y, which depend on real measurements y:
    x(t) = Phi(t-1, xHat(t-1)) x(t-1) + Phi2(t-1, x(t-1), xHat(t-1)) W(t)
    Y(t) = Psi(t, x(t), y(t)) x(t) + Psi1(t, x(t), y(t)) Nu(t)

    if the structure functions Phi, Psi can not be defined in the 
    inline manner or require some history, an external object may be used: Phi1,2 = Phi1,2(model, ...), Psi1,2 = Psi1,2(model, ...)

    """
    def __init__(self, Phi1, dPhi1, Phi2, Psi1, dPsi1, Psi2, MW, DW, MNu, DNu):
        self.Phi1 = Phi1                    #Phi1 = Phi1(model, k-1, xHat)
        self.dPhi1 = dPhi1                  #dPhi1 = dPhi1(model, k-1, xHat) = dPhi1/dx (Jacobian)
        self.Phi2 = Phi2                    #Phi2 = Phi2(model, k-1, xHat)
        self.Psi1 = Psi1                    #Psi1 = Psi1(model, k, x, y)
        self.dPsi1 = dPsi1                  #dPsi1 = dPsi1(model, k, x, y) = dPsi1/dx (Jacobian)
        self.Psi2 = Psi2                    #Psi2 = Psi2(model, k, x, y)
        self.MW = MW                        #Noise in state dynamics mean
        self.DW = DW                        #Noise in state dynamics covariance
        self.MNu = MNu                      #Noise in observations mean
        self.DNu = DNu                      #Noise in observations covariance

    def Step(self, model, k, y, xHat_, kHat_, Y = []):
        """
        One step estimate xHat(t) = Step(model, t, y(t), xHat(t-1), Y(t)),
        where y - original measurements, Y - pseudo measurements        
        """
        if len(Y) == 0:
            # if no pseudomeasurements provided, then use the original ones
            Y = y
        F = self.dPhi1(model, k-1, xHat_)
        Q = self.Phi2(model, k-1, xHat_) @ self.DW @ self.Phi2(model, k-1, xHat_).T
        xTilde = self.Phi1(model, k-1, xHat_) + self.Phi2(model, k-1, xHat_) @ self.MW;
        kTilde = F @ kHat_ @ F.T + Q; 

        H = self.dPsi1(model, k, xTilde, y);
        R = self.Psi2(model, k, xTilde, y) @ self.DNu @ self.Psi2(model, k, xTilde, y).T;
        I = np.eye(xTilde.shape[0]);
        K = kTilde @ H.T @ np.linalg.pinv(H @ kTilde @ H.T + R);
        xHat__ = xTilde + K @ (Y - self.Psi1(model, k, xTilde, y) - self.Psi2(model, k, xTilde, y) @ self.MNu);
        kHat = (I - K @ H) @ kTilde;

        return xHat__, kHat


