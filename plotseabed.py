from AUV import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
NBeams = 8
PhiBounds = [7.0,13.0]
PhiGrad = np.append(np.arange(PhiBounds[0], PhiBounds[1], (PhiBounds[1] - PhiBounds[0])/NBeams), PhiBounds[1])

ThetaBounds = [-15.0,15.0]
ThetaGrad = np.append(np.arange(ThetaBounds[0], ThetaBounds[1], (ThetaBounds[1] - ThetaBounds[0])/NBeams), ThetaBounds[1])

print(PhiGrad)
print(ThetaGrad)

auv = AUV([0.001,-0.0002,5.0003], lambda t: np.array([2.0, 2.0 * np.cos(0.1 * t), 0.0]), 0.01)
auv.initAcousticSensor(PhiGrad / 180.0 * np.pi, ThetaGrad / 180.0 * np.pi)
print(type(PhiGrad))
#print(auv.L(auv.X, 0.0, 0.0))
#print(auv.BeamCoords(auv.X, 0.0, 0.0))


bn, _, _, _, _ = auv.CurrentBeamNet()
bn_plain = np.reshape(bn, (bn.size,1))
bn1_X, bn1_Y, bn1_Z = np.array([p[0][0] for p in bn_plain]), np.array([p[0][1] for p in bn_plain]), np.array([p[0][2] for p in bn_plain])
scatter1 = ax.scatter(bn1_X, bn1_Y, bn1_Z, color = 'black', s = 10)

auv.step()

bn, _, _, _, _ = auv.CurrentBeamNet()
bn_plain = np.reshape(bn, (bn.size,1))
bn2_X, bn2_Y, bn2_Z = np.array([p[0][0] for p in bn_plain]), np.array([p[0][1] for p in bn_plain]), np.array([p[0][2] for p in bn_plain])
scatter2 = ax.scatter(bn2_X, bn2_Y, bn2_Z, color = 'blue', s = 15)

auv.step()

bn, _, _, _, _ = auv.CurrentBeamNet()
bn_plain = np.reshape(bn, (bn.size,1))
bn3_X, bn3_Y, bn3_Z = np.array([p[0][0] for p in bn_plain]), np.array([p[0][1] for p in bn_plain]), np.array([p[0][2] for p in bn_plain])
scatter3 = ax.scatter(bn3_X, bn3_Y, bn3_Z, color = 'green', s = 15)

auv.step()

bn, _, _, _, _ = auv.CurrentBeamNet()
bn_plain = np.reshape(bn, (bn.size,1))
bn4_X, bn4_Y, bn4_Z = np.array([p[0][0] for p in bn_plain]), np.array([p[0][1] for p in bn_plain]), np.array([p[0][2] for p in bn_plain])
scatter4 = ax.scatter(bn4_X, bn4_Y, bn4_Z, color = 'red', s = 15)

auv.step()

bn, _, _, _, _ = auv.CurrentBeamNet()
bn_plain = np.reshape(bn, (bn.size,1))
bn5_X, bn5_Y, bn5_Z = np.array([p[0][0] for p in bn_plain]), np.array([p[0][1] for p in bn_plain]), np.array([p[0][2] for p in bn_plain])
scatter5 = ax.scatter(bn5_X, bn5_Y, bn5_Z, color = 'cyan', s = 15)


bndX = [np.min([bn1_X, bn2_X, bn3_X, bn4_X, bn5_X]), np.max([bn1_X, bn2_X, bn3_X, bn4_X, bn5_X])]
bndY = [np.min([bn1_Y, bn2_Y, bn3_Y, bn4_Y, bn5_Y]), np.max([bn1_Y, bn2_Y, bn3_Y, bn4_Y, bn5_Y])]

X = np.arange(bndX[0], bndX[1], (bndX[1] - bndX[0])/ 100.0)
Y = np.arange(bndY[0], bndY[1], (bndY[1] - bndY[0])/ 100.0)
X, Y = np.meshgrid(X, Y)
Z = Seabed.z(X,Y)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.3)


#print(np.array([p[0][0] for p in bn_plain]))
#print(np.array([p[0][1] for p in bn_plain]))
#print(np.array([p[0][2] for p in bn_plain]))

# Customize the z axis.
#ax.set_xlim(-0.5, 0.5)
#ax.set_ylim(-0.5, 0.5)
#ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

for i in range(0,100):
    auv.step

bn2_X, bn2_Y, bn2_Z = np.array([p[0][0] for p in bn_plain]), np.array([p[0][1] for p in bn_plain]), np.array([p[0][2] for p in bn_plain])
scatter2 = ax.scatter(bn2_X, bn2_Y, bn2_Z, color = 'blue', s = 5)

bndX = [np.min([bn1_X, bn2_X]), np.max([bn1_X, bn2_X])]
bndY = [np.min([bn1_Y, bn2_Y]), np.max([bn1_Y, bn2_Y])]

X = np.arange(bndX[0], bndX[1], (bndX[1] - bndX[0])/ 100.0)
Y = np.arange(bndY[0], bndY[1], (bndY[1] - bndY[0])/ 100.0)
X, Y = np.meshgrid(X, Y)
Z = Seabed.z(X,Y)


plt.show()
