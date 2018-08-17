from AUV import *
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import datetime

class testenvironment():
    """AUV model"""
    def __init__(self, T, delta, NBeams, accuracy, PhiBounds, ThetaBounds, X0, V):
        self.T = T
        self.delta = delta
        self.Npoints = int(T / delta)
        self.NBeams = NBeams
        self.accuracy = np.array(accuracy)
        self.PhiBounds = np.array(PhiBounds)
        self.ThetaBounds = np.array(ThetaBounds)
        self.auv = AUV(X0, V, delta)
        for i in range(0, self.accuracy.size):
            ph = self.PhiBounds[i,:]
            th = self.ThetaBounds[i,:]
            PhiGrad     = np.append(np.arange(ph[0], ph[1], (ph[1] - ph[0])/self.NBeams), ph[1])
            ThetaGrad   = np.append(np.arange(th[0], th[1], (th[1] - th[0])/self.NBeams), th[1])
            self.auv.addsensor(accuracy[i], PhiGrad / 180.0 * np.pi, ThetaGrad / 180.0 * np.pi)
        self.colors = ['lavenderblush', 'pink', 'plum', 'palevioletred', 'mediumvioletred', 'mediumorchid', 'darkorchid', 'purple']
    def run(self):
        for i in range(0, self.Npoints):
            self.auv.step()
            print(self.delta * i)
    def plottrajectory(self, path):
        self.run()
        now = datetime.datetime.now()
        f = plt.figure(num=None, figsize=(5,5), dpi=200, facecolor='w', edgecolor='k')
        for k in range(0,3):
            f = plt.figure(num=None, figsize=(5,5), dpi=200, facecolor='w', edgecolor='k')
            plt.plot(self.auv.t_history, self.auv.X_history[:,k], color='black')
            for i,s in enumerate(self.auv.Sensors):
                plt.plot(self.auv.t_history, s.X_estimate_history[:,k], color=self.colors[i], label=str(i))
            plt.plot(self.auv.t_history, self.auv.X_estimate_history[:,k], color='blue')
            plt.legend()
            plt.savefig(path + now.strftime(str(k) + "___%Y-%m-%d %H-%M-%S-%f")+'.jpg')
        with open(path + "results.txt", "a") as myfile:
            myfile.write(
                now.strftime("%Y-%m-%d %H-%M-%S") + " " + 
                str(self.T) + " " + 
                str(self.delta) + " " + 
                str(self.NBeams) + " " + 
                np.array2string(
                    np.mean(list(map(lambda x: x.X_estimate_history[self.Npoints-1,:] - self.auv.X_history[self.Npoints-1,:], self.auv.Sensors)), axis=0), 
                    formatter={'float_kind':lambda x: "%.5f" % x}
                    ) + " " +
                np.array2string(
                    self.auv.X_estimate_history[self.Npoints-1,:] - self.auv.X_history[self.Npoints-1,:], 
                    formatter={'float_kind':lambda x: "%.5f" % x}
                    ) + " " +
                "\n"
                )
    #def plotseabedsequence(self, path):
    #    fig = plt.figure(figsize=(10, 6), dpi=200)
    #    ax = fig.gca(projection='3d')
    #    ax.view_init(50, 30)

    #    bn, _, _, _, _ = self.auv.beamnet()
    #    bn_plain = np.reshape(bn, (bn.size,1))
    #    bn1_X, bn1_Y, bn1_Z = np.array([p[0][0] for p in bn_plain]), np.array([p[0][1] for p in bn_plain]), np.array([p[0][2] for p in bn_plain])


    #    for i in range(0, self.Npoints):
    #        print(self.delta * i)
    #        self.auv.step()
    #        scatter1 = ax.scatter(bn1_X, bn1_Y, bn1_Z, color = 'black', s = 30)

    #        bn, _, _, _, _ = self.auv.beamnet()
    #        bn_plain = np.reshape(bn, (bn.size,1))
    #        bn2_X, bn2_Y, bn2_Z = np.array([p[0][0] for p in bn_plain]), np.array([p[0][1] for p in bn_plain]), np.array([p[0][2] for p in bn_plain])
    #        scatter2 = ax.scatter(bn2_X, bn2_Y, bn2_Z, color = 'blue', s = 30)

    #        bndX = [np.min([bn1_X, bn2_X]), np.max([bn1_X, bn2_X])]
    #        bndY = [np.min([bn1_Y, bn2_Y]), np.max([bn1_Y, bn2_Y])]

    #        X = np.arange(bndX[0], bndX[1], (bndX[1] - bndX[0])/ 100.0)
    #        Y = np.arange(bndY[0], bndY[1], (bndY[1] - bndY[0])/ 100.0)
    #        X, Y = np.meshgrid(X, Y)
    #        Z = Seabed.z(X,Y)

    #        # Plot the surface.
    #        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
    #                               linewidth=0, antialiased=False, alpha=0.3)

    #        ax.zaxis.set_major_locator(LinearLocator(10))
    #        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    #        # Add a color bar which maps values to colors.
    #        fig.colorbar(surf, shrink=0.5, aspect=5)
    #        plt.savefig(path + '\\' + str(i) + '.jpg')

    #        bn1_X, bn1_Y, bn1_Z = bn2_X, bn2_Y, bn2_Z 
   
    #        fig = plt.figure(figsize=(10, 6), dpi=200)
    #        ax = fig.gca(projection='3d')
    #        ax.view_init(50, 30)
    def showradar(self):
        fig = plt.figure(figsize=(10, 6), dpi=200)
        ax = fig.gca(projection='3d')
        ax.view_init(50, 30)

        bndX = [self.auv.X[0], self.auv.X[0]]
        bndY = [self.auv.X[1], self.auv.X[1]]

        for s in self.auv.Sensors:
            bn, _, _, _, _ = s.beamnet(self.auv.X)
            bn_plain = np.reshape(bn, (bn.size,1))
            bn1_X, bn1_Y, bn1_Z = np.array([p[0][0] for p in bn_plain]), np.array([p[0][1] for p in bn_plain]), np.array([p[0][2] for p in bn_plain])
            #_ = ax.scatter(bn1_X, bn1_Y, bn1_Z, color = 'black', s = 30)
            bndX = [np.min(np.hstack((bn1_X, bndX[0]))), np.max(np.hstack((bn1_X, bndX[1])))]
            bndY = [np.min(np.hstack((bn1_Y, bndY[0]))), np.max(np.hstack((bn1_Y, bndY[1])))]
        
        print('')
        self.auv.step()
        print('')

        for i, s in enumerate(self.auv.Sensors):
            bn, _, _, _, _ = s.beamnet(self.auv.X)
            bn_plain = np.reshape(bn, (bn.size,1))
            bn2_X, bn2_Y, bn2_Z = np.array([p[0][0] for p in bn_plain]), np.array([p[0][1] for p in bn_plain]), np.array([p[0][2] for p in bn_plain])
            _ = ax.scatter(bn2_X, bn2_Y, bn2_Z, color = self.colors[i], s = 30)
            bndX = [np.min(np.hstack((bn2_X, bndX[0]))), np.max(np.hstack((bn2_X, bndX[1])))]
            bndY = [np.min(np.hstack((bn2_Y, bndY[0]))), np.max(np.hstack((bn2_Y, bndY[1])))]

        print('')

        X = np.arange(bndX[0], bndX[1], (bndX[1] - bndX[0])/ 100.0)
        Y = np.arange(bndY[0], bndY[1], (bndY[1] - bndY[0])/ 100.0)
        X, Y = np.meshgrid(X, Y)
        Z = Seabed.z(X,Y)

        _ = ax.scatter(self.auv.X[0], self.auv.X[1], self.auv.X[2], color = 'red', s = 100)

        # Plot the surface.
        #surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.3)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        #fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        
