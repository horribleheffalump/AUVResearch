from AUV import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


# Make data.

T = 30.0
delta = 0.05
Npoints = int(T / delta)

NBeams = 15
PhiBounds = [7.0,13.0]
PhiGrad = np.append(np.arange(PhiBounds[0], PhiBounds[1], (PhiBounds[1] - PhiBounds[0])/NBeams), PhiBounds[1])

ThetaBounds = [-10.0,10.0]
ThetaGrad = np.append(np.arange(ThetaBounds[0], ThetaBounds[1], (ThetaBounds[1] - ThetaBounds[0])/NBeams), ThetaBounds[1])

auv = AUV([0.001,-0.0002,3.0003], lambda t: np.array([1.0 + 0.2 * np.cos(1.0 * t), np.cos(0.1 * t), 0.1 * np.sin(2.0 * t)]), delta)

auv.initAcousticSensor(PhiGrad / 180.0 * np.pi, ThetaGrad / 180.0 * np.pi)

print(PhiGrad)
print(ThetaGrad)

print(type(PhiGrad))
#print(auv.L(auv.X, 0.0, 0.0))
#print(auv.BeamCoords(auv.X, 0.0, 0.0))

fig = plt.figure(figsize=(10, 6), dpi=200)
ax = fig.gca(projection='3d')
ax.view_init(50, 30)

bn, _, _, _, _ = auv.CurrentBeamNet()
bn_plain = np.reshape(bn, (bn.size,1))
bn1_X, bn1_Y, bn1_Z = np.array([p[0][0] for p in bn_plain]), np.array([p[0][1] for p in bn_plain]), np.array([p[0][2] for p in bn_plain])


for i in range(0,Npoints):
    print(delta * i)
    auv.step()
    scatter1 = ax.scatter(bn1_X, bn1_Y, bn1_Z, color = 'black', s = 30)

    bn, _, _, _, _ = auv.CurrentBeamNet()
    bn_plain = np.reshape(bn, (bn.size,1))
    bn2_X, bn2_Y, bn2_Z = np.array([p[0][0] for p in bn_plain]), np.array([p[0][1] for p in bn_plain]), np.array([p[0][2] for p in bn_plain])
    scatter2 = ax.scatter(bn2_X, bn2_Y, bn2_Z, color = 'blue', s = 30)

    bndX = [np.min([bn1_X, bn2_X]), np.max([bn1_X, bn2_X])]
    bndY = [np.min([bn1_Y, bn2_Y]), np.max([bn1_Y, bn2_Y])]

    X = np.arange(bndX[0], bndX[1], (bndX[1] - bndX[0])/ 100.0)
    Y = np.arange(bndY[0], bndY[1], (bndY[1] - bndY[0])/ 100.0)
    X, Y = np.meshgrid(X, Y)
    Z = Seabed.z(X,Y)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, alpha=0.3)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(str(i) + '.jpg')

    bn1_X, bn1_Y, bn1_Z = bn2_X, bn2_Y, bn2_Z 
   
    fig = plt.figure(figsize=(10, 6), dpi=200)
    ax = fig.gca(projection='3d')
    ax.view_init(50, 30)
