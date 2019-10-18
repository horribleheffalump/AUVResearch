import numpy as np

class CMNFFilter():
    """Conditionnaly minimax nonlinear filter"""
    def __init__(self, Phi, Psi, DW, DNu, Xi, Zeta):
        self.Phi = Phi
        self.Psi = Psi
        self.DW = DW
        self.sigmaW = np.sqrt(DW)
        self.DNu = DNu
        self.sigmaNu = np.sqrt(DNu)
        self.Xi = Xi
        self.Zeta = Zeta
        self.tol = 1e-20
    def EstimateParameters(self, models, X0all, XHat0, N, M):
        #M = States.shape[0] #number of samples
        self.FHat = [];
        self.fHat = [];
        self.HHat = [];
        self.hHat = [];
        self.KTilde = [];
        self.KHat = [];
        #x = np.array(list(map(lambda i: X0, range(0, M) )))
        x = X0all
        xHat = np.array(list(map(lambda i: XHat0, range(0, M) )))
        for t in range(1, N + 1):
            print('estimate params CMNF t=',t)
            for i in range(0, M):
                models[i].step(xHat[i])
            x = np.array(list(map(lambda i: self.Phi(models[i], t-1, x[i], xHat[i]) + self.sigmaW * np.array(np.random.normal(0.0,1.0, self.DW.shape[0])), range(0, M) )))
            y = np.array(list(map(lambda i: self.Psi(models[i], t, x[i], []) + self.sigmaNu * np.array(np.random.normal(0.0,1.0, self.DNu.shape[0])), range(0, M) )))
            xiHat = np.array(list(map(lambda i : self.Xi(models[i], t-1, xHat[i]), range(0, M))))
            CovXiHat = CMNFFilter.COV(xiHat, xiHat)
            InvCovXiHat = np.zeros_like(CovXiHat)
            if (np.linalg.norm(CovXiHat) > self.tol):
                InvCovXiHat = np.linalg.pinv(CovXiHat)

            F = np.dot(CMNFFilter.COV(x, xiHat), InvCovXiHat)
            f = x.mean(axis=0) - np.dot(F, xiHat.mean(axis=0))
            kTilde = CMNFFilter.COV(x, x) - np.dot(F, CMNFFilter.COV(x, xiHat).T)

            xTilde = np.array(list(map(lambda i : np.dot(F, xiHat[i]) + f, range(0,M))))
            zetaTilde = np.array(list(map(lambda i : self.Zeta(models[i], t, xTilde[i], y[i]), range(0,M))))
            delta_x_xTilde = np.array(list(map(lambda i : x[i] - xTilde[i], range(0,M))))
            delta_by_zetaTilde = np.array(list(map(lambda i : np.outer(delta_x_xTilde[i], zetaTilde[i]), range(0,M))))

            CovZetaTilde = CMNFFilter.COV(zetaTilde, zetaTilde)
            InvCovZetaTilde = np.zeros_like(CovZetaTilde)
            if (np.linalg.norm(CovZetaTilde) > self.tol):
                InvCovZetaTilde = np.linalg.pinv(CovZetaTilde)

            delta_x = x-xTilde
            H = np.dot(delta_by_zetaTilde.mean(axis=0), InvCovZetaTilde)
            h = np.dot(-H, zetaTilde.mean(axis=0))
            kHat = kTilde - np.dot(CMNFFilter.COV(delta_x, zetaTilde), H.T)

            xHat = np.array(list(map(lambda i : xTilde[i] + np.dot(H, zetaTilde[i]) + h, range(0,M))))

            self.FHat.append(F)
            self.fHat.append(f)
            self.HHat.append(H)
            self.hHat.append(h)
            self.KTilde.append(kTilde)
            self.KHat.append(kHat)          
        self.FHat = np.array(self.FHat)
        self.fHat = np.array(self.fHat)
        self.HHat = np.array(self.HHat)
        self.hHat = np.array(self.hHat)
        self.KTilde = np.array(self.KTilde)
        self.KHat = np.array(self.KHat)

    def SaveParameters(self, filename_template):
        np.save(filename_template.replace('[param]','FMultHat'), self.FHat)
        np.save(filename_template.replace('[param]','FAddHat'), self.fHat)
        np.save(filename_template.replace('[param]','HMultHat'), self.HHat)
        np.save(filename_template.replace('[param]','HAddHat'), self.hHat)
        np.save(filename_template.replace('[param]','KTilde'), self.KTilde)
        np.save(filename_template.replace('[param]','KHat'), self.KHat)

    def LoadParameters(self, filename_template):
        self.FHat = np.load(filename_template.replace('[param]','FMultHat'))
        self.fHat = np.load(filename_template.replace('[param]','FAddHat'))
        self.HHat = np.load(filename_template.replace('[param]','HMultHat'))
        self.hHat = np.load(filename_template.replace('[param]','HAddHat'))
        self.KTilde = np.load(filename_template.replace('[param]','KTilde'))
        self.KHat = np.load(filename_template.replace('[param]','KHat'))

    def Step(self, model, k, y, xHat_, kHat_):
        if (k == len(self.FHat)): ## OMG!! This is a dirty trick to make the CMNF time scale in line with Kalman filter timescale. Otherwise we need to calculate CMNF params on one additional step.
            k -= 1 
        xTilde = np.dot(self.FHat[k], self.Xi(model, k-1, xHat_)) + self.fHat[k]
        xHat = xTilde + np.dot(self.HHat[k], self.Zeta(model, k, xTilde, y)) + self.hHat[k]
        return xHat, self.KHat[k]

    @staticmethod
    def COV(X,Y):
        n = X.shape[0]
        cX = X - X.mean(axis=0)[np.newaxis,:] 
        cY = Y - Y.mean(axis=0)[np.newaxis,:] 
        return np.dot(cX.T, cY)/(n-1.) 

