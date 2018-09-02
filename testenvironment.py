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
    def __init__(self, T, delta, NBeams, accuracy, PhiBounds, ThetaBounds, X0, V, seabed, estimateslope):
        self.T = T
        self.delta = delta
        self.Npoints = int(T / delta)
        self.NBeams = NBeams
        self.accuracy = np.array(accuracy)
        self.PhiBounds = np.array(PhiBounds)
        self.ThetaBounds = np.array(ThetaBounds)
        self.auv = AUV(X0, V, delta)
        self.seabed = seabed;
        for i in range(0, self.accuracy.size):
            ph = self.PhiBounds[i,:]
            th = self.ThetaBounds[i,:]
            PhiGrad     = np.append(np.arange(ph[0], ph[1], (ph[1] - ph[0])/self.NBeams), ph[1])
            ThetaGrad   = np.append(np.arange(th[0], th[1], (th[1] - th[0])/self.NBeams), th[1])
            self.auv.addsensor(accuracy[i], PhiGrad / 180.0 * np.pi, ThetaGrad / 180.0 * np.pi, seabed, estimateslope)
        self.colors = ['lavenderblush', 'pink', 'plum', 'palevioletred', 'mediumvioletred', 'mediumorchid', 'darkorchid', 'purple']
    def run(self):
        for i in range(0, self.Npoints):
            self.auv.step()
            print(self.delta * i)
    def plottrajectory(self, path):
        start = datetime.datetime.now()
        self.run()
        finish = datetime.datetime.now()
        f = plt.figure(num=None, figsize=(15,5), dpi=200, facecolor='w', edgecolor='k')
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])     
        gs.update(left=0.03, bottom=0.05, right=0.99, top=0.99, wspace=0.1, hspace=0.1)    

        for k in range(0,3):
            #f = plt.figure(num=None, figsize=(5,5), dpi=200, facecolor='w', edgecolor='k')
            ax = plt.subplot(gs[k])
            ax.plot(self.auv.t_history, self.auv.X_history[:,k], color='black')
            for i,s in enumerate(self.auv.Sensors):
                ax.plot(self.auv.t_history, s.X_estimate_history[:,k], color=self.colors[i], label=str(i))
            ax.plot(self.auv.t_history, self.auv.X_estimate_history[:,k], color='blue')
            ax.legend()
        plt.savefig(path + finish.strftime("%Y-%m-%d %H-%M-%S-%f")+'.jpg')
        with open(path + "results.txt", "a") as myfile:
            myfile.write(
                finish.strftime("%Y-%m-%d %H-%M-%S") + " " + 
                str(self.T) + " " + 
                str(self.delta) + " " + 
                str(self.NBeams) + " " + 
                #np.array2string(
                #    np.mean(list(map(lambda x: x.X_estimate_history[self.Npoints-1,:] - self.auv.X_history[self.Npoints-1,:], self.auv.Sensors)), axis=0), 
                #    formatter={'float_kind':lambda x: "%.5f" % x}
                #    ) + " " +
                np.array2string(
                    self.auv.X_estimate_history[self.Npoints-1,:] - self.auv.X_history[self.Npoints-1,:], 
                    formatter={'float_kind':lambda x: "%.5f" % x}
                    ) + " " +
                "elapsed seconds: " + str((finish-start).total_seconds()) + " " +
                "\n"
                )
    def plotseabedsequence(self, path, sideplots):
        for t in range(0, self.Npoints):
            print(self.delta * t)
            self.auv.step()

            fig = plt.figure(figsize=(10, 6), dpi=200)
            gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1], height_ratios=[1, 1, 1])     
            gs.update(left=0.03, bottom=0.05, right=0.99, top=0.99, wspace=0.1, hspace=0.1)    

            if sideplots == 'X':
                ax = fig.add_subplot(gs[:,0], projection='3d')
                for k in range(0,3):
                    axp = fig.add_subplot(gs[k,1])
                    axp.plot(self.auv.t_history, self.auv.X_history[:,k], color='black')
                    for i,s in enumerate(self.auv.Sensors):
                        axp.plot(self.auv.t_history, s.X_estimate_history[:,k], color=self.colors[i], label=str(i))
                    axp.plot(self.auv.t_history, self.auv.X_estimate_history[:,k], color='blue')
            elif sideplots == 'deltaX':
                ax = fig.add_subplot(gs[:,0], projection='3d')
                for k in range(0,3):
                    axp = fig.add_subplot(gs[k,1])
                    axp.plot(self.auv.t_history, self.auv.delta_X_history[:,k], color='black')
                    for i,s in enumerate(self.auv.Sensors):
                        axp.plot(self.auv.t_history, s.delta_X_estimate_history[:,k], color=self.colors[i], label=str(i))
                    axp.plot(self.auv.t_history, self.auv.delta_X_estimate_history[:,k], color='blue')
            else:
                ax = fig.add_subplot(gs[:], projection='3d')

            ax.view_init(30, 30)



            bndX = [self.auv.X[0], self.auv.X[0]]
            bndY = [self.auv.X[1], self.auv.X[1]]

            for i, s in enumerate(self.auv.Sensors):
                bn, _, _, _, _ = s.beamnet(self.auv.X)
                bn_X, bn_Y, bn_Z = bn[:,0], bn[:,1], bn[:,2]
                _ = ax.scatter(bn_X, bn_Y, bn_Z, color = self.colors[i], s = 30)
                bndX = [np.min(np.hstack((bn_X, bndX[0]))), np.max(np.hstack((bn_X, bndX[1])))]
                bndY = [np.min(np.hstack((bn_Y, bndY[0]))), np.max(np.hstack((bn_Y, bndY[1])))]

            X = np.arange(bndX[0], bndX[1], (bndX[1] - bndX[0])/ 100.0)
            Y = np.arange(bndY[0], bndY[1], (bndY[1] - bndY[0])/ 100.0)
            X, Y = np.meshgrid(X, Y)
            Z = self.seabed.Z(X,Y)

            v_scale = 50.0
            Vplot = np.vstack((self.auv.X, self.auv.X + v_scale * self.delta * self.auv.V(t))) 
            
            _ = ax.scatter(Vplot[0,0], Vplot[0,1], Vplot[0,2], color = 'red', s = 100)
            _ = ax.scatter(Vplot[1,0], Vplot[1,1], Vplot[1,2], color = 'red', s = 10)
            _ = ax.plot(Vplot[:,0], Vplot[:,1], Vplot[:,2], color = 'red')
    
            # Plot the surface.
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.3)

            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

            #fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.savefig(path + '\\' + str(t) + '.jpg')

    def showradar(self):
        fig = plt.figure(figsize=(10, 6), dpi=200)
        ax = fig.gca(projection='3d')
        ax.view_init(50, 30)

        bndX = [self.auv.X[0], self.auv.X[0]]
        bndY = [self.auv.X[1], self.auv.X[1]]

        for s in self.auv.Sensors:
            bn, _, _, _, _ = s.beamnet(self.auv.X)
            bn1_X, bn1_Y, bn1_Z = bn[:,0], bn[:,1], bn[:,2]
            _ = ax.scatter(bn1_X, bn1_Y, bn1_Z, color = 'black', s = 10)
            bndX = [np.min(np.hstack((bn1_X, bndX[0]))), np.max(np.hstack((bn1_X, bndX[1])))]
            bndY = [np.min(np.hstack((bn1_Y, bndY[0]))), np.max(np.hstack((bn1_Y, bndY[1])))]
        
        print('')
        self.auv.step()
        print('')

        for i, s in enumerate(self.auv.Sensors):
            bn, _, _, _, _ = s.beamnet(self.auv.X)
            bn2_X, bn2_Y, bn2_Z = bn[:,0], bn[:,1], bn[:,2]
            _ = ax.scatter(bn2_X, bn2_Y, bn2_Z, color = self.colors[i], s = 30)
            bndX = [np.min(np.hstack((bn2_X, bndX[0]))), np.max(np.hstack((bn2_X, bndX[1])))]
            bndY = [np.min(np.hstack((bn2_Y, bndY[0]))), np.max(np.hstack((bn2_Y, bndY[1])))]

        print('')

        X = np.arange(bndX[0], bndX[1], (bndX[1] - bndX[0])/ 100.0)
        Y = np.arange(bndY[0], bndY[1], (bndY[1] - bndY[0])/ 100.0)
        X, Y = np.meshgrid(X, Y)
        Z = self.seabed.Z(X,Y)

        _ = ax.scatter(self.auv.X[0], self.auv.X[1], self.auv.X[2], color = 'red', s = 100)
    
        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.3)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        #fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        
