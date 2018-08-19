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
accuracy = [0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
PhiBounds =     [[15.0,30.0],   [20.0, 35.0],  [10.0,25.0],  [10.0, 35.0], [15.0,30.0],   [20.0, 35.0],  [10.0,25.0],  [10.0, 35.0]]
ThetaBounds =   [[10.0+15.0,10.0-15.0],   [100.0+15.0,100.0-15.0],  [190.0+15.0,190.0-15.0],  [280.0+15.0,280.0-15.0], [55.0+15.0,55.0-15.0],   [145.0+15.0,145.0-15.0],  [235.0+15.0,235.0-15.0],  [325.0+15.0,325.0-15.0]]
X0 = [0.001,-0.0002,-10.0003]
V = lambda t: np.array([1.0 + 0.2 * np.cos(1.0 * t), np.cos(0.1 * t), 0.1 * np.sin(2.0 * t)])
#V = lambda t: np.array([0.3, -0.02*np.cos(0.05 * t), 0.0])


ph = [15.0,30.0]
th = [10.0+15.0,10.0-15.0]
PhiGrad     = np.append(np.arange(ph[0], ph[1], (ph[1] - ph[0])/ NBeams), ph[1])
ThetaGrad   = np.append(np.arange(th[0], th[1], (th[1] - th[0])/ NBeams), th[1])

Phi = PhiGrad / 180.0 * np.pi
Theta = ThetaGrad / 180.0 * np.pi
X = [0.001,-0.0002,-10.0003]
func = lambda l : Seabed.z(X[0] + e[0] * l, X[1] + e[1] * l) - X[2] - e[2] * l
L_current = 0.0 

L = np.zeros((Phi.size, Theta.size))
r = np.empty((Phi.size, Theta.size), dtype=np.ndarray)
dzdx = np.zeros((Phi.size, Theta.size))
dzdy = np.zeros((Phi.size, Theta.size))
for i in range(Phi.size):
    for j in range(Theta.size):
        e = np.array([
            np.sin(Phi[i])*np.cos(Theta[j]), 
            np.sin(Phi[i])*np.sin(Theta[j]), 
            -np.cos(Phi[i])])
        L_current = fsolve(func, L_current)    
        L[i,j] = L_current #+ np.random.normal(0,0.1)
        r[i,j] = X + L[i,j] * e
        dzdx[i,j], dzdy[i,j] = Seabed.dz(r[i,j][0], r[i,j][1])

r_plain = np.reshape(r, (r.size,1))
dzdx_plain = np.reshape(dzdx, (1, dzdx.size))
dzdy_plain = np.reshape(dzdy, (1, dzdy.size))

X, Z = np.array([p[0][0:2] for p in r_plain]), np.array([p[0][2] for p in r_plain])
#print(X)
#print(Z)
est = SlopeApproximator()
est.predict(X,Z)
features = est.pf.get_feature_names()
features.remove("1")
print(features)
x0 = [a.count("x0") for a in features]
x1 = [a.count("x1") for a in features]
x0pows = np.array(list(map(max, zip(x0, [sum([tryParseInt(b.replace("x0^","")) for b in a.split(" ")]) for a in features]))))
x1pows = np.array(list(map(max, zip(x1, [sum([tryParseInt(b.replace("x1^","")) for b in a.split(" ")]) for a in features]))))
x0pows_1 = np.abs(x0pows - 1.0)
x1pows_1 = np.abs(x1pows - 1.0)
print(x0)
print(x1)
print(x0pows)
print(x1pows)
print(x0pows_1)
print(x1pows_1)
#print(est.partialdiffs(X)[0])
#print(dzdx_plain)
#print(est.partialdiffs(X)[1])
#print(dzdy_plain)
#print(est.partialdiffs(X)[2] - Z)



        #np.fromfunction(lambda i, j: self.BeamCoords(self.X, 0,0), (len(self.Phi), len(self.Theta)), )


test = testenvironment(T, delta, NBeams, accuracy, PhiBounds, ThetaBounds, X0, V)

#test.showradar()
#test.plottrajectory('C:\\projects.git\\NavigationResearch\\results\\')
#test.plotseabedsequence('D:\\projects.git\\NavigationResearch\\results\\')