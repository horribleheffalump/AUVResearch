import numpy as np

class CMNFFilter():
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
        self.Phi = Phi                  # Nonlinear function of the state dynamics. Phi = Phi(model, t-1, x, xHat)
        self.Psi = Psi                  # Nonlinear function of the observations. Phi = Psi(model, t, x, _) (last parameter is not used here, it is to pretend some other duck)
        self.DW = DW                    # Covariation of the Gaussian noise in the state equation
        self.sigmaW = np.sqrt(DW)       # Standard deviation of the Gaussian noise in state equation
        self.DNu = DNu                  # Covariation of the Gaussian noise in the observations
        self.sigmaNu = np.sqrt(DNu)     # Standard deviation of the Gaussian noise in the observations
        self.Xi = Xi                    # CMNF basic predictor function. Xi = Xi(model, t, x)
        self.Zeta = Zeta                # CMNF basic correction function. Zeta = Zeta(model, t, x, y)
        self.tol = 1e-20                # tolerance for the matrix inverse calculation 

    def EstimateParameters(self, models, X0all, XHat0, N, M, filename_template: str):
        """
        This function calculates the parameters of the CMNF with Monte-Carlo sampling: we generate a
        bunch of sample paths and calculate the sampled covariances of the state, prediction and estimate 
        
        models - if the structure functions Phi, Psi can not be defined in the 
        inline manner or require some history, an external object may be used for each path sample.
        models parameters is a list of such objects. Should implement step(self, XHat)
        
        X0all - array of initial conditions for the sample paths
        
        XHat0 - array of initial estimates
        
        N - time limit
        
        M - number of samples to generate

        filename_template - file template to save the filter parameters
        """
        self.FHat = []
        self.fHat = []
        self.HHat = []
        self.hHat = []
        self.KTilde = []
        self.KHat = []
        x = X0all
        epsilon = 0.1 # regularization for the initial step (otherwise CovXiHat is zero)
        xHat = np.array(list(map(lambda i: XHat0 + epsilon * (np.random.normal(0.0, 1.0, XHat0.shape[0])), range(0, M))))
        for t in range(1, N + 1):
            print('estimate params CMNF t=', t)
            for i in range(0, M):
                models[i].step(xHat[i])
            x = np.array(list(map(lambda i: self.Phi(models[i], t-1, x[i], xHat[i]) + self.sigmaW * np.array(np.random.normal(0.0,1.0, self.DW.shape[0])), range(0, M) )))
            y = np.array(list(map(lambda i: self.Psi(models[i], t, x[i], []) + self.sigmaNu * np.array(np.random.normal(0.0,1.0, self.DNu.shape[0])), range(0, M) )))
            xiHat = np.array(list(map(lambda i : self.Xi(models[i], t-1, xHat[i]), range(0, M))))
            CovXiHat = CMNFFilter.COV(xiHat, xiHat)

            #InvCovXiHatU, InvCovXiHatS, InvCovXiHatVH  = CMNFFilter.inverseSVD(CovXiHat)
            #F = (CMNFFilter.COV(x, xiHat) @ InvCovXiHatU) 
            #F = F @ InvCovXiHatS @ InvCovXiHatVH
            #InvCovXiHat = InvCovXiHatU @ InvCovXiHatS @ InvCovXiHatVH
            
            InvCovXiHat = CMNFFilter.inverse(CovXiHat)
            F = CMNFFilter.COV(x, xiHat) @ InvCovXiHat 
            f = x.mean(axis=0) - np.dot(F, xiHat.mean(axis=0))
            kTilde = CMNFFilter.COV(x, x) - np.dot(F, CMNFFilter.COV(x, xiHat).T)


            xTilde = np.array(list(map(lambda i : np.dot(F, xiHat[i]) + f, range(0,M))))
            zetaTilde = np.array(list(map(lambda i : self.Zeta(models[i], t, xTilde[i], y[i]), range(0,M))))
            delta_x_xTilde = np.array(list(map(lambda i : x[i] - xTilde[i], range(0,M))))
            delta_by_zetaTilde = np.array(list(map(lambda i : np.outer(delta_x_xTilde[i], zetaTilde[i]), range(0,M))))

            CovZetaTilde = CMNFFilter.COV(zetaTilde, zetaTilde)

            #InvCovZetaTildeU, InvCovZetaTildeS, InvCovZetaTildeVH = CMNFFilter.inverseSVD(CovZetaTilde)
            #H = (delta_by_zetaTilde.mean(axis=0) @ InvCovZetaTildeU) 
            #H = H @ InvCovZetaTildeS @ InvCovZetaTildeVH

            #InvCovZetaTilde = CMNFFilter.inverse(CovZetaTilde)
            InvCovZetaTilde = np.linalg.pinv(CovZetaTilde)
            H = delta_by_zetaTilde.mean(axis=0) @ InvCovZetaTilde 
            h = np.dot(-H, zetaTilde.mean(axis=0))
            delta_x = x-xTilde
            kHat = kTilde - np.dot(CMNFFilter.COV(delta_x, zetaTilde), H.T)

            xHat = np.array(list(map(lambda i : xTilde[i] + np.dot(H, zetaTilde[i]) + h, range(0,M))))

            self.FHat.append(F)
            self.fHat.append(f)
            self.HHat.append(H)
            self.hHat.append(h)
            self.KTilde.append(kTilde)
            self.KHat.append(kHat)    

            np.savetxt(filename_template.replace('[param]', 'CovXiHat'), CovXiHat, fmt='%f')
            np.savetxt(filename_template.replace('[param]', 'InvCovXiHat'), InvCovXiHat, fmt='%f')
            np.savetxt(filename_template.replace('[param]', 'CovXiHat_by_InvCovXiHat'), CovXiHat @ InvCovXiHat, fmt='%f')
            np.savetxt(filename_template.replace('[param]', 'InvCovXiHat_by_CovXiHat'), InvCovXiHat @ CovXiHat, fmt='%f')

            np.savetxt(filename_template.replace('[param]', 'CovZetaTilde'), CovZetaTilde, fmt='%f')
            np.savetxt(filename_template.replace('[param]', 'InvCovZetaTilde'), InvCovZetaTilde, fmt='%f')
            np.savetxt(filename_template.replace('[param]', 'CovZetaTilde_by_InvCovZetaTilde'),  CovZetaTilde @ InvCovZetaTilde, fmt='%f')
            np.savetxt(filename_template.replace('[param]', 'InvCovZetaTilde_by_CovZetaTilde'),  InvCovZetaTilde @ CovZetaTilde, fmt='%f')

        self.FHat = np.array(self.FHat)
        self.fHat = np.array(self.fHat)
        self.HHat = np.array(self.HHat)
        self.hHat = np.array(self.hHat)
        self.KTilde = np.array(self.KTilde)
        self.KHat = np.array(self.KHat)

    def SaveParameters(self, filename_template):
        """
        Saves the CMNF parameters calculated by EstimateParameters(...) in files     
        """
        np.save(filename_template.replace('[param]','FMultHat'), self.FHat)
        np.save(filename_template.replace('[param]','FAddHat'), self.fHat)
        np.save(filename_template.replace('[param]','HMultHat'), self.HHat)
        np.save(filename_template.replace('[param]','HAddHat'), self.hHat)
        np.save(filename_template.replace('[param]','KTilde'), self.KTilde)
        np.save(filename_template.replace('[param]','KHat'), self.KHat)

    def LoadParameters(self, filename_template):
        """
        Loads the pre-estimated CMNF parameters 
        """
        self.FHat = np.load(filename_template.replace('[param]','FMultHat'))
        self.fHat = np.load(filename_template.replace('[param]','FAddHat'))
        self.HHat = np.load(filename_template.replace('[param]','HMultHat'))
        self.hHat = np.load(filename_template.replace('[param]','HAddHat'))
        self.KTilde = np.load(filename_template.replace('[param]','KTilde'))
        self.KHat = np.load(filename_template.replace('[param]','KHat'))

    def Step(self, model, k, y, xHat_, kHat_):
        """
        One step estimate xHat(t) = Step(model, t, y(t), xHat(t-1))
        kHat is for compatibility, not required here
        """
        if (k == len(self.FHat)): 
            # OMG!! Here comes a dirty trick to make the CMNF time scale in line with Kalman filter timescale. 
            #  Otherwise we need to calculate CMNF params on one additional step. 
            #  Note that this affects the quality of the estimate on the final step!!!
            k -= 1 
        xTilde = np.dot(self.FHat[k], self.Xi(model, k-1, xHat_)) + self.fHat[k]
        xCorr = np.dot(self.HHat[k], self.Zeta(model, k, xTilde, y)) + self.hHat[k]
        xHat = xTilde + xCorr
        #if (np.linalg.norm(xHat)) > 1e10:
        #    print("diverged")
        return xHat, self.KHat[k], xTilde, xCorr

    # sampled covariation of two sequences
    @staticmethod
    def COV(X,Y):
        n = X.shape[0]
        cX = X - X.mean(axis=0)[np.newaxis,:] 
        cY = Y - Y.mean(axis=0)[np.newaxis,:] 
        return np.dot(cX.T, cY)/(n-1.) 

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
        #zero = s == 0
        inv_s = 1.0 / s #(s + zero) * np.invert(zero)
        return u, np.diag(inv_s), vh

