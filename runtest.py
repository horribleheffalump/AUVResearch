from testenvironment import *
from SlopeApproximator import *

# Make data.
#T = 30.0
#delta = 0.05
#NBeams = 15
#PhiBounds = [9.0,11.0]
#ThetaBounds = [-3.0,3.0]
#X0 = [0.001,-0.0002,13.0003]
#V = lambda t: np.array([1.0 + 0.2 * np.cos(1.0 * t), np.cos(0.1 * t), 0.1 * np.sin(2.0 * t)])
def tryParseInt(val):
    if val.isnumeric():
        return int(val)
    else:
        return 0


T = 10.0
delta = 0.1
NBeams = 10
accuracy = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
PhiBounds =     [[15.0,30.0],   [20.0, 35.0],  [10.0,25.0],  [10.0, 35.0], [15.0,30.0],   [20.0, 35.0],  [10.0,25.0],  [10.0, 35.0]]
ThetaBounds =   [[10.0+15.0,10.0-15.0],   [100.0+15.0,100.0-15.0],  [190.0+15.0,190.0-15.0],  [280.0+15.0,280.0-15.0], [55.0+15.0,55.0-15.0],   [145.0+15.0,145.0-15.0],  [235.0+15.0,235.0-15.0],  [325.0+15.0,325.0-15.0]]
#PhiBounds =     [[5.0,6.0],   [10.0, 11.0],  [10.0,11.0],  [10.0, 11.0], [15.0,16.0],   [10.0, 11.0],  [8.0,9.0],  [10.0, 11.0]]
#ThetaBounds =   [[10.0+2.0,10.0-2.0],   [100.0+2.0,100.0-2.0],  [190.0+2.0,190.0-2.0],  [280.0+2.0,280.0-2.0], [55.0+2.0,55.0-2.0],   [145.0+2.0,145.0-2.0],  [235.0+2.0,235.0-2.0],  [325.0+2.0,325.0-2.0]]
X0 = [0.001,-0.0002,-10.0003]
V = lambda t: np.array([1.0 + 0.2 * np.cos(1.0 * t), np.cos(0.1 * t), 0.1 * np.sin(2.0 * t)])
estimateslope = False
seabed = Seabed()



#ph = [15.0,30.0]
#th = [10.0+15.0,10.0-15.0]
#PhiGrad     = np.append(np.arange(ph[0], ph[1], (ph[1] - ph[0])/ NBeams), ph[1])
#ThetaGrad   = np.append(np.arange(th[0], th[1], (th[1] - th[0])/ NBeams), th[1])

#Phi = PhiGrad / 180.0 * np.pi
#Theta = ThetaGrad / 180.0 * np.pi
#X = [0.001,-0.0002,-10.0003]
#func = lambda l : Seabed.z(X[0] + e[0] * l, X[1] + e[1] * l) - X[2] - e[2] * l
#L_current = 0.0 

#L = np.zeros((Phi.size, Theta.size))
#r = np.empty((Phi.size, Theta.size), dtype=np.ndarray)
#dzdx = np.zeros((Phi.size, Theta.size))
#dzdy = np.zeros((Phi.size, Theta.size))
#for i in range(Phi.size):
#    for j in range(Theta.size):
#        e = np.array([
#            np.sin(Phi[i])*np.cos(Theta[j]), 
#            np.sin(Phi[i])*np.sin(Theta[j]), 
#            -np.cos(Phi[i])])
#        L_current = fsolve(func, L_current)    
#        L[i,j] = L_current #+ np.random.normal(0,0.1)
#        r[i,j] = X + L[i,j] * e
#        dzdx[i,j], dzdy[i,j] = Seabed.dz(r[i,j][0], r[i,j][1])

#r_plain = np.reshape(r, (r.size,1))
#dzdx_plain = np.reshape(dzdx, (1, dzdx.size))
#dzdy_plain = np.reshape(dzdy, (1, dzdy.size))

#X, Z = np.array([p[0][0:2] for p in r_plain]), np.array([p[0][2] for p in r_plain])
##print(X)
##print(Z)
#est = SlopeApproximator()
#est.predict(X,Z)
#est.partialdiffs(X)

##print(est.partialdiffs(X)[0])
##print(dzdx_plain)
##print(est.partialdiffs(X)[1])
##print(dzdy_plain)
##print(est.partialdiffs(X)[2] - Z)



        #np.fromfunction(lambda i, j: self.BeamCoords(self.X, 0,0), (len(self.Phi), len(self.Theta)), )


test = testenvironment(T, delta, NBeams, accuracy, PhiBounds, ThetaBounds, X0, V, seabed, estimateslope)

test.showradar()
test.plottrajectory('D:\\projects.git\\NavigationResearch\\results\\')
#test.plotseabedsequence('D:\\projects.git\\NavigationResearch\\results\\')