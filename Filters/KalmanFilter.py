
import numpy as np

class KalmanFilter():
    """Kalman filter"""
    def __init__(self, Phi1, dPhi1, Phi2, Psi1, dPsi1, Psi2, MW, DW, MNu, DNu):
        self.Phi1 = Phi1
        self.dPhi1 = dPhi1
        self.Phi2 = Phi2
        self.Psi1 = Psi1
        self.dPsi1 = dPsi1
        self.Psi2 = Psi2
        self.MW = MW
        self.DW = DW
        self.MNu = MNu
        self.DNu = DNu

    def Step(self, model, k, y, xHat_, kHat_):
        F = self.dPhi1(model, k, xHat_)
        Q = self.Phi2(model, k, xHat_) @ self.DW @ self.Phi2(model, k, xHat_).T
        xTilde = self.Phi1(model, k, xHat_) + self.Phi2(model, k, xHat_) @ self.MW;
        kTilde = F @ kHat_ @ F.T + Q; 

        H = self.dPsi1(model, k, xTilde, y);
        R = self.Psi2(model, k, y, xTilde) @ self.DNu @ self.Psi2(model, k, y, xTilde).T;
        I = np.eye(xTilde.shape[0]);
        K = kTilde @ H.T @ np.linalg.pinv(H @ kTilde @ H.T + R);
        xHat__ = xTilde + K @ (y - self.Psi1(model, k, xTilde, y) - self.Psi2(model, k, xTilde, y) @ self.MNu);
        kHat = (I - K @ H) @ kTilde;

        return xHat__, kHat


