import numpy as np
from Filters.CMNFFilter import CMNFFilter
from numba import jit
import time

class SimpleCMNFFilter():
    """
    Conditionnaly minimax nonlinear filter
    for a nonlinear stchastic dicret-time model:
    x(t) = Phi(t-1, x(t-1), xHat(t-1)) + W(t)   - state dynamics
    y(t) = Psi(t, x(t)) + Nu(t)                 - observations
    with t in [0, N]
    W, N - Gaussian white noise with zero mean and covariances DW, DNu
    Xi, Zeta - basic prediction and correction functions,
    in general case can be chosen as follows:
    Xi = Phi                                    - by virtue of the system
    Zeta = y - Psi                              - residual

    if the structure functions Phi, Psi can not be defined in the
    inline manner or require some history, an external object may be used: Phi = Phi(model, ...), Psi = Psi(model, ...)
    """

    def __init__(self, Phi, Psi, DW, DNu, Xi, Zeta):
        self.Phi = Phi  # Nonlinear function of the state dynamics. Phi = Phi(model, t-1, x, xHat)
        self.Psi = Psi  # Nonlinear function of the observations. Phi = Psi(model, t, x, _) (last parameter is not used here, it is to pretend some other duck)
        self.DW = DW  # Covariation of the Gaussian noise in the state equation
        self.sigmaW = np.sqrt(DW)  # Standard deviation of the Gaussian noise in state equation
        self.m_W = np.zeros_like(self.sigmaW)
        self.DNu = DNu  # Covariation of the Gaussian noise in the observations
        self.sigmaNu = np.sqrt(DNu)  # Standard deviation of the Gaussian noise in the observations
        self.m_Nu = np.zeros_like(self.sigmaNu)
        self.Xi = Xi  # CMNF basic predictor function. Xi = Xi(model, t, x)
        self.Zeta = Zeta  # CMNF basic correction function. Zeta = Zeta(model, t, x, y)
        self.tol = 1e-20  # tolerance for the matrix inverse calculation

    def EstimateParameters(self, x, y, XHat0):
        """
        This function calculates the parameters of the CMNF with Monte-Carlo sampling: we generate a
        bunch of sample paths and calculate the sampled covariances of the state, prediction and estimate

        X0all - array of initial conditions for the sample paths

        XHat0 - array of initial estimates

        N - time limit

        M - number of samples to generate

        filename_template - file template to save the filter parameters
        """
        M = x.shape[0]
        N = x.shape[1]

        zeta_test = self.Zeta(x[0, 0,:], y[0, 0, :])

        self.FHat = np.zeros((N, x.shape[2], x.shape[2]))
        self.fHat = np.zeros((N, x.shape[2]))
        self.HHat = np.zeros((N, x.shape[2], zeta_test.shape[0]))
        self.hHat = np.zeros((N, x.shape[2]))
        self.KTilde = np.zeros((N, x.shape[2], x.shape[2]))
        self.KHat = np.zeros((N, x.shape[2], x.shape[2]))

        xHat = np.tile(XHat0, (M,1))
        epsilon = 0.1  # regularization for the initial step (otherwise CovXiHat is zero)
        xHat = xHat + epsilon * np.random.normal(size=xHat.shape)
        start = time.time()
        for t in range(0, N):
            if t % 10 == 0:
                end = time.time()
                print(f"estimate params CMNF t={t}, elapsed {end - start}")
                start = time.time()

            xiHat = np.apply_along_axis(self.Xi, 1, xHat)
            CovXiHat = np.cov(xiHat, rowvar=False)

            InvCovXiHat = SimpleCMNFFilter.inverse(CovXiHat)
            F = SimpleCMNFFilter.cov(x[:, t, :], xiHat) @ InvCovXiHat
            f = x[:, t, :].mean(axis=0) - np.dot(F, xiHat.mean(axis=0))
            kTilde = np.cov(x[:, t, :], rowvar=False) - np.dot(F, SimpleCMNFFilter.cov(x[:, t, :], xiHat).T)

            xTilde = np.apply_along_axis(lambda x: F @ x + f, 1, xiHat)
            zetaTilde = np.array(list(map(lambda i: self.Zeta(xTilde[i, :], y[i, t, :]), range(0, M))))
            delta_x_xTilde = x[:, t, :] - xTilde

            CovZetaTilde = np.cov(zetaTilde, rowvar=False)

            InvCovZetaTilde = np.linalg.pinv(CovZetaTilde)
            H = SimpleCMNFFilter.cov(delta_x_xTilde, zetaTilde) @ InvCovZetaTilde
            h = np.dot(-H, zetaTilde.mean(axis=0))
            delta_x = x[:, t, :] - xTilde
            kHat = kTilde - np.dot(SimpleCMNFFilter.cov(delta_x, zetaTilde), H.T)

            xHat = xTilde + zetaTilde @ H.T + h

            self.FHat[t, :, :] = F
            self.fHat[t, :] = f
            self.HHat[t, :, :] = H
            self.hHat[t, :] = h
            self.KTilde[t, :, :] = kTilde
            self.KHat[t, :, :] = kHat

    def Filter(self, y, XHat0):
        M = y.shape[0]
        N = y.shape[1]

        xHat = np.zeros((M, N, XHat0.shape[0]))

        xHat[:, 0, :] = np.tile(XHat0, (M,1))
        start = time.time()
        for t in range(1, N):
            if t % 10 == 0:
                end = time.time()
                print(f"filter t={t}, elapsed {end - start}")
                start = time.time()

            xiHat = np.apply_along_axis(self.Xi, 1, xHat[:, t-1, :])
            xTilde = np.apply_along_axis(lambda x: self.FHat[t, :, :] @ x + self.fHat[t, :], 1, xiHat)
            zetaTilde = np.array(list(map(lambda i: self.Zeta(xTilde[i, :], y[i, t, :]), range(0, M))))
            xHat[:, t, :] = xTilde + zetaTilde @ self.HHat[t, :, :].T + self.hHat[t, :]
        return(xHat)

    def SaveParameters(self, filename_template):
        """
        Saves the CMNF parameters calculated by EstimateParameters(...) in files
        """
        np.save(filename_template.replace('[param]', 'FMultHat'), self.FHat)
        np.save(filename_template.replace('[param]', 'FAddHat'), self.fHat)
        np.save(filename_template.replace('[param]', 'HMultHat'), self.HHat)
        np.save(filename_template.replace('[param]', 'HAddHat'), self.hHat)
        np.save(filename_template.replace('[param]', 'KTilde'), self.KTilde)
        np.save(filename_template.replace('[param]', 'KHat'), self.KHat)

    def LoadParameters(self, filename_template):
        """
        Loads the pre-estimated CMNF parameters
        """
        self.FHat = np.load(filename_template.replace('[param]', 'FMultHat'))
        self.fHat = np.load(filename_template.replace('[param]', 'FAddHat'))
        self.HHat = np.load(filename_template.replace('[param]', 'HMultHat'))
        self.hHat = np.load(filename_template.replace('[param]', 'HAddHat'))
        self.KTilde = np.load(filename_template.replace('[param]', 'KTilde'))
        self.KHat = np.load(filename_template.replace('[param]', 'KHat'))

    def Step(self, k, y, xHat_):
        """
        One step estimate xHat(t) = Step(model, t, y(t), xHat(t-1))
        kHat is for compatibility, not required here
        """
        if (k == len(self.FHat)):
            # OMG!! Here comes a dirty trick to make the CMNF time scale in line with Kalman filter timescale.
            #  Otherwise we need to calculate CMNF params on one additional step.
            #  Note that this affects the quality of the estimate on the final step!!!
            k -= 1
        xTilde = np.dot(self.FHat[k], self.Xi(xHat_)) + self.fHat[k]
        xCorr = np.dot(self.HHat[k], self.Zeta(xTilde, y)) + self.hHat[k]
        xHat = xTilde + xCorr
        return xHat, self.KHat[k], xTilde, xCorr

    # sampled covariation of two sequences
    @staticmethod
    def cov(X, Y):
        n = X.shape[0]
        cX = X - np.mean(X, axis=0)
        cY = Y - np.mean(Y, axis=0)
        return np.dot(cX.T, cY) / (n - 1.)

    # inverse with svd decomposition
    @staticmethod
    def inverse(A):
        tol = 1e-2
        u, s, vh = np.linalg.svd(A)
        nonzero = np.abs(s) > tol
        inv_s = 1.0 / (s + np.invert(nonzero)) * (nonzero)
        return u @ np.diag(inv_s) @ vh

    @staticmethod
    def inverseSVD(A):
        u, s, vh = np.linalg.svd(A)
        # zero = s == 0
        inv_s = 1.0 / s  # (s + zero) * np.invert(zero)
        return u, np.diag(inv_s), vh



