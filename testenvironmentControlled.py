from AUV import *
import matplotlib.pyplot as plt
from matplotlib import gridspec

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import datetime


from mpl_toolkits.mplot3d import Axes3D


from DrawHelper import *

from matplotlib import rc
rc('font',**{'family':'serif'})
rc('text', usetex=True)
rc('text.latex',unicode=True)
rc('text.latex',preamble=r'\usepackage[T2A]{fontenc}')
rc('text.latex',preamble=r'\usepackage[utf8]{inputenc}')
rc('text.latex',preamble=r'\usepackage[russian]{babel}')



class testenvironmentControlled():
    """AUV model"""
    def __init__(self, T, delta, NBeams, accuracy, PhiBounds, ThetaBounds, auv, seabed, estimateslope):

        self.T = T
        self.delta = delta
        self.Npoints = int(T / delta)
        self.NBeams = NBeams
        self.accuracy = np.array(accuracy)
        self.PhiBounds = np.array(PhiBounds)
        self.ThetaBounds = np.array(ThetaBounds)
        self.auv = auv
        self.seabed = seabed
        for i in range(0, self.accuracy.size):
            ph = self.PhiBounds[i,:]
            th = self.ThetaBounds[i,:]
            PhiGrad = np.append(np.arange(ph[0], ph[1], (ph[1] - ph[0]) / self.NBeams), ph[1])
            ThetaGrad = np.append(np.arange(th[0], th[1], (th[1] - th[0]) / self.NBeams), th[1])
            self.auv.addsensor(accuracy[i], PhiGrad / 180.0 * np.pi, ThetaGrad / 180.0 * np.pi, seabed, estimateslope)
        #self.colors = ['lavenderblush', 'pink', 'plum', 'palevioletred',
        #'mediumvioletred', 'mediumorchid', 'darkorchid', 'purple']
        self.colors = ['red', 'green', 'blue', 'cyan', 'pink', 'yellow', 'orange', 'purple']
    def run(self):
        for i in range(0, self.Npoints):
            t = self.delta * i
            self.auv.step()
            print(t)
    def crit(self):
        for i in range(0, self.Npoints):
            self.auv.step()
        err_cov = np.dot(np.transpose(self.auv.X_history - self.auv.X_estimate_history), (self.auv.X_history - self.auv.X_estimate_history))
        #print(err_cov)
        return(np.trace(err_cov))
    def plottrajectory(self, path):
        #start = datetime.datetime.now()
        #self.run()
        #finish = datetime.datetime.now()
        f = plt.figure(num=None, figsize=(15,5), dpi=200, facecolor='w', edgecolor='k')
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])     
        gs.update(left=0.03, bottom=0.05, right=0.99, top=0.99, wspace=0.1, hspace=0.1)    

        for k in range(0,3):
            #f = plt.figure(num=None, figsize=(5,5), dpi=200, facecolor='w',
            #edgecolor='k')
            ax = plt.subplot(gs[k])

            for i,s in enumerate(self.auv.Sensors):
                ax.plot(self.auv.t_history, s.X_estimate_history[:,k], color=self.colors[i], label='Estimate' + str(i), linewidth=0.5)
            ax.plot(self.auv.t_history, self.auv.XNominal_history[:,k], color='grey', linewidth=4.0, label='Nominal')
            ax.plot(self.auv.t_history, self.auv.XReal_history[:,k], color='black', linewidth=2.0, label='Real')
            ax.plot(self.auv.t_history, self.auv.XReal_estimate_history[:,k], color='red', linewidth=2.0, label='Estimate')
            ax.legend()
        #plt.savefig(path + finish.strftime("%Y-%m-%d %H-%M-%S-%f") + '.png')
        plt.savefig(path + 'pathsample.png')
        with open(path + "results.txt", "a") as myfile:
            #myfile.write(finish.strftime("%Y-%m-%d %H-%M-%S") + " " +
            #str(self.T) + " " + str(self.delta) + " " + str(self.NBeams) + " "
            #+ #np.array2string(
            myfile.write(" " + str(self.T) + " " + str(self.delta) + " " + str(self.NBeams) + " " + #np.array2string(
                #    np.mean(list(map(lambda x:
                #    x.X_estimate_history[self.Npoints-1,:] -
                #    self.auv.X_history[self.Npoints-1,:], self.auv.Sensors)),
                #    axis=0),
                #    formatter={'float_kind':lambda x: "%.5f" % x}
                #    ) + " " +
                np.array2string(self.auv.XReal_estimate_history[self.Npoints - 1,:] - self.auv.XReal_history[self.Npoints - 1,:], 
                    formatter={'float_kind':lambda x: "%.5f" % x}) + " " + "\n") #+ "elapsed seconds: " + str((finish - start).total_seconds()) + " " + "\n")
    def plotspeed(self, sonars_xyz, path):
        #start = datetime.datetime.now()
        #self.run()
        #finish = datetime.datetime.now()
        f = plt.figure(num=None, figsize=(15,5), dpi=200, facecolor='w', edgecolor='k')
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])     
        gs.update(left=0.03, bottom=0.05, right=0.99, top=0.99, wspace=0.15, hspace=0.1)    

        for k in range(0,3):
            ax = plt.subplot(gs[k])
            for i,s in enumerate(self.auv.Sensors):
                if (i == sonars_xyz[k]):
                    ax.plot(self.auv.t_history[:], s.delta_X_estimate_history[:,k] / self.delta, color = 'red', label='Estimate' + str(i), linewidth = 0.5) #color=self.colors[i]
            ax.plot(self.auv.t_history[:-1], self.auv.VNominal_history[:,k], color='grey', linewidth=4.0, label='Nominal')
            ax.plot(self.auv.t_history[:-1], self.auv.VReal_history[:,k], color='black', linewidth=2.0, label='Real')
            ax.legend()
                                                                                                                                         #color='black', linewidth=2.0)
                                                                                                                                         #ax.legend()
        #plt.show()
        f.savefig(path + 'speed_sample.pdf')

    def plotspeederror(self, sonars_xyz, path):
        #start = datetime.datetime.now()
        #self.run()
        #finish = datetime.datetime.now()
        f = plt.figure(num=None, figsize=(15,5), dpi=200, facecolor='w', edgecolor='k')
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])     
        gs.update(left=0.03, bottom=0.05, right=0.99, top=0.99, wspace=0.15, hspace=0.1)    

        for k in range(0,3):
            ax = plt.subplot(gs[k])
            #ax.plot(self.auv.t_history, self.auv.V_history[:,k],
            #color='black', linewidth=2.0)
            for i,s in enumerate(self.auv.Sensors):
                if (i == sonars_xyz[k]):
                    ax.plot(self.auv.t_history[:], s.delta_X_estimate_history[:,k] / self.delta - self.auv.V_history[:,k], color = 'red', label=str(i)) #color=self.colors[i]
            #ax.plot(self.auv.t_history, self.auv.X_estimate_history[:,k],
                                                                                                                                                                   #color='black', linewidth=2.0)
                                                                                                                                                                   #ax.legend()
        #plt.show()
        f.savefig(path + 'speed_error_sample.pdf')

    def stats(self, points, sonars_xyz, path):
        #start = datetime.datetime.now()
        #self.run()
        #finish = datetime.datetime.now()
        f = plt.figure(num=None, figsize=(15,5), dpi=200, facecolor='w', edgecolor='k')
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])     
        gs.update(left=0.03, bottom=0.05, right=0.99, top=0.99, wspace=0.1, hspace=0.1)    
        for p in points:
            with open(path + "results_" + str(p) + ".txt", "a") as myfile:
                #myfile.write(finish.strftime("%Y-%m-%d %H-%M-%S") + " " +
                #str(self.T) + " " + str(self.delta) + " " + str(self.NBeams) +
                #" " +
                #str(self.auv.Sensors[sonars_xyz[0]].X_estimate_history[p,0] -
                #self.auv.X_history[p,0]) + " " +
                #str(self.auv.Sensors[sonars_xyz[1]].X_estimate_history[p,1] -
                #self.auv.X_history[p,1]) + " " +
                #str(self.auv.Sensors[sonars_xyz[2]].X_estimate_history[p,2] -
                #self.auv.X_history[p,2]) + " " + "elapsed seconds: " +
                #str((finish - start).total_seconds()) + " " + "\n")
                myfile.write(" " + str(self.T) + " " + str(self.delta) + " " + str(self.NBeams) + " " + str(self.auv.Sensors[sonars_xyz[0]].X_estimate_history[p,0] - self.auv.X_history[p,0]) + " " + str(self.auv.Sensors[sonars_xyz[1]].X_estimate_history[p,1] - self.auv.X_history[p,1]) + " " + str(self.auv.Sensors[sonars_xyz[2]].X_estimate_history[p,2] - self.auv.X_history[p,2]) + " " + "elapsed seconds: " + str((finish - start).total_seconds()) + " " + "\n")

    def speedstats(self, sonars_xyz, path):
        #self.run()
        for k in range(0,3):
            with open(path + "results_speed_" + str(k) + ".txt", "a") as myfile:
                s = str(self.auv.V_history[0,k] - self.auv.Sensors[sonars_xyz[k]].delta_X_estimate_history[0,k] / self.delta)
                for i in range(1,self.auv.t_history.size):
                    s = s + ", " + str(self.auv.V_history[i,k] - self.auv.Sensors[sonars_xyz[k]].delta_X_estimate_history[i,k] / self.delta) 
                myfile.write(s + "\n")


    def plot3DtrajectoryByCoords(self, sonars_xyz, path):
        #start = datetime.datetime.now()
        #self.run()
        #finish = datetime.datetime.now()
        f = plt.figure(num=None, figsize=(15,5), dpi=200, facecolor='w', edgecolor='k')
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])     
        gs.update(left=0.03, bottom=0.05, right=0.99, top=0.99, wspace=0.1, hspace=0.1)    

        for k in range(0,3):
            #f = plt.figure(num=None, figsize=(5,5), dpi=200, facecolor='w',
            #edgecolor='k')
            ax = plt.subplot(gs[k])
            ax.plot(self.auv.t_history, self.auv.X_history[:,k], color='black', linewidth=2.0, linestyle=':')
            for i,s in enumerate(self.auv.Sensors):
                if (i == sonars_xyz[k]):
                    ax.plot(self.auv.t_history, s.X_estimate_history[:,k], color=self.colors[i], label=str(i))
        #plt.savefig(path + finish.strftime("%Y-%m-%d %H-%M-%S-%f") + '.jpg')
        plt.savefig(path + '3Dpathsample.png')
           
    def plot3Dtrajectory(self, sonarNumber, path):
        #start = datetime.datetime.now()
        #self.run()
        #finish = datetime.datetime.now()
        fig = plt.figure(num=None, figsize=(15,5), dpi=200, facecolor='w', edgecolor='k')     
        gs = gridspec.GridSpec(1, 1)     
        ax = fig.add_subplot(gs[:], projection='3d')
        _ = ax.plot(self.auv.XNominal_history[:,0], self.auv.XNominal_history[:,1], self.auv.XNominal_history[:,2], color = 'grey', linewidth=4.0, label = 'Nominal')
        _ = ax.plot(self.auv.XReal_history[:,0], self.auv.XReal_history[:,1], self.auv.XReal_history[:,2], color = 'black', linewidth=2.0, label = 'Real')
        for i,s in enumerate(self.auv.Sensors):
            if (i == sonarNumber):
                _ = ax.plot(s.X_estimate_history[:,0], s.X_estimate_history[:,1], s.X_estimate_history[:,2], color = 'red', label = 'estimate ' + str(i))
        ax.legend()
        #for k in range(0,3):
        #    #f = plt.figure(num=None, figsize=(5,5), dpi=200, facecolor='w',
        #    edgecolor='k')
        #    ax = plt.subplot(gs[k])
        #    ax.plot(self.auv.t_history, self.auv.X_history[:,k],
        #    color='black', linewidth=2.0, linestyle=':')
        #    for i,s in enumerate(self.auv.Sensors):
        #        if (i == sonars_xyz[k]):
        #            ax.plot(self.auv.t_history, s.X_estimate_history[:,k],
        #            color=self.colors[i], label=str(i))
        #plt.savefig(path + finish.strftime("%Y-%m-%d %H-%M-%S-%f")+'.jpg')
        plt.show()  


    def plotseabedsequence(self, path, sideplots):
        for t in range(0, self.Npoints):
            print(self.delta * t)
            self.auv.step()

            fig = plt.figure(figsize=(12, 6), dpi=200)
            gs = gridspec.GridSpec(3, 3, width_ratios=[3, 1, 1], height_ratios=[1, 1, 1])     
            gs.update(left=0.03, bottom=0.05, right=0.99, top=0.99, wspace=0.1, hspace=0.1)    

            if sideplots == 'both':
                ax = fig.add_subplot(gs[:,0], projection='3d')
                for k in range(0,3):
                    axp = fig.add_subplot(gs[k,1])
                    for i,s in enumerate(self.auv.Sensors):
                        axp.plot(self.auv.t_history, s.delta_X_estimate_history[:,k], color=self.colors[i], label=str(i))
                    axp.plot(self.auv.t_history, self.auv.delta_X_estimate_history[:,k], color='black', linewidth=2.0)
                    axp.plot(self.auv.t_history, self.auv.delta_X_history[:,k], color='black', linewidth=2.0, linestyle=':')
                for k in range(0,3):
                    axp = fig.add_subplot(gs[k,2])
                    for i,s in enumerate(self.auv.Sensors):
                        axp.plot(self.auv.t_history, s.X_estimate_history[:,k], color=self.colors[i], label=str(i))
                    axp.plot(self.auv.t_history, self.auv.X_estimate_history[:,k], color='black', linewidth=2.0)
                    axp.plot(self.auv.t_history, self.auv.X_history[:,k], color='black', linewidth=2.0, linestyle=':')
            elif sideplots == 'X':
                ax = fig.add_subplot(gs[:,0:2], projection='3d')
                for k in range(0,3):
                    axp = fig.add_subplot(gs[k,2])
                    for i,s in enumerate(self.auv.Sensors):
                        axp.plot(self.auv.t_history, s.X_estimate_history[:,k], color=self.colors[i], label=str(i))
                    axp.plot(self.auv.t_history, self.auv.X_estimate_history[:,k], color='black', linewidth=2.0)
                    axp.plot(self.auv.t_history, self.auv.X_history[:,k], color='black', linewidth=2.0, linestyle=':')
            elif sideplots == 'deltaX':
                ax = fig.add_subplot(gs[:,0:2], projection='3d')
                for k in range(0,3):
                    axp = fig.add_subplot(gs[k,2])
                    for i,s in enumerate(self.auv.Sensors):
                        axp.plot(self.auv.t_history, s.delta_X_estimate_history[:,k], color=self.colors[i], label=str(i))
                    axp.plot(self.auv.t_history, self.auv.delta_X_estimate_history[:,k], color='black', linewidth=2.0)
                    axp.plot(self.auv.t_history, self.auv.delta_X_history[:,k], color='black', linewidth=2.0, linestyle=':')
            else:
                ax = fig.add_subplot(gs[:], projection='3d')

            ax.view_init(30, 30)



            bndX = [self.auv.X[0], self.auv.X[0]]
            bndY = [self.auv.X[1], self.auv.X[1]]

            for i, s in enumerate(self.auv.Sensors):
                if isinstance(s, SensorControlled):
                    bn, _, _, _, _ = s.beamnet(self.auv.X, self.auv.U(self.auv.t))
                else:  
                    bn, _, _, _, _ = s.beamnet(self.auv.X)

                bn_X, bn_Y, bn_Z = bn[:,0], bn[:,1], bn[:,2]
                _ = ax.scatter(bn_X, bn_Y, bn_Z, color = self.colors[i], s = 30)
                bndX = [np.min(np.hstack((bn_X, bndX[0]))), np.max(np.hstack((bn_X, bndX[1])))]
                bndY = [np.min(np.hstack((bn_Y, bndY[0]))), np.max(np.hstack((bn_Y, bndY[1])))]

            X = np.arange(bndX[0], bndX[1], (bndX[1] - bndX[0]) / 100.0)
            Y = np.arange(bndY[0], bndY[1], (bndY[1] - bndY[0]) / 100.0)
            X, Y = np.meshgrid(X, Y)
            Z = self.seabed.Z(X,Y)

            v_scale = 50.0
            Vplot = np.vstack((self.auv.X, self.auv.X + v_scale * self.delta * self.auv.V_history[t,:])) 
            
            _ = ax.scatter(Vplot[0,0], Vplot[0,1], Vplot[0,2], color = 'red', s = 100)
            _ = ax.scatter(Vplot[1,0], Vplot[1,1], Vplot[1,2], color = 'red', s = 10)
            _ = ax.plot(Vplot[:,0], Vplot[:,1], Vplot[:,2], color = 'red')

            _ = ax.plot([self.auv.X[0], self.auv.X[0] + 5.0],[self.auv.X[1], self.auv.X[1] + 0.0],[self.auv.X[2], self.auv.X[2] + 0.0], color = 'red')
            _ = ax.plot([self.auv.X[0], self.auv.X[0] + 0.0],[self.auv.X[1], self.auv.X[1] + 5.0],[self.auv.X[2], self.auv.X[2] + 0.0], color = 'green')
            _ = ax.plot([self.auv.X[0], self.auv.X[0] + 0.0],[self.auv.X[1], self.auv.X[1] + 0.0],[self.auv.X[2], self.auv.X[2] + 5.0], color = 'blue')

    
            # Plot the surface.
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.3)

            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

            #fig.colorbar(surf, shrink=0.5, aspect=5)
            #if t % 10 == 0 and t > 29:
            #   plt.show()
            #else:
            plt.savefig(path + '\\' + str(t) + '.png')

    def showradar(self, sonars=[]):
        if sonars == []:
            sonars = range(0, len(self.auv.Sensors))
        fig = plt.figure(figsize=(10, 6), dpi=200)
        ax = fig.gca(projection='3d')
        ax.view_init(50, 30)

        bndX = [self.auv.X[0], self.auv.X[0]]
        bndY = [self.auv.X[1], self.auv.X[1]]

        for i, s in enumerate(self.auv.Sensors):
            bn, _, _, _, _ = s.beamnet(self.auv.X)
            bn1_X, bn1_Y, bn1_Z = bn[:,0], bn[:,1], bn[:,2]
            if i in sonars:
                _ = ax.scatter(bn1_X, bn1_Y, bn1_Z, color = 'black', s = 10)
            bndX = [np.min(np.hstack((bn1_X, bndX[0]))), np.max(np.hstack((bn1_X, bndX[1])))]
            bndY = [np.min(np.hstack((bn1_Y, bndY[0]))), np.max(np.hstack((bn1_Y, bndY[1])))]
        
        #print('')
        #self.auv.step()
        #print('')

        #for i, s in enumerate(self.auv.Sensors):
        #    bn, _, _, _, _ = s.beamnet(self.auv.X)
        #    bn2_X, bn2_Y, bn2_Z = bn[:,0], bn[:,1], bn[:,2]
        #    _ = ax.scatter(bn2_X, bn2_Y, bn2_Z, color = self.colors[i], s =
        #    30)
        #    bndX = [np.min(np.hstack((bn2_X, bndX[0]))),
        #    np.max(np.hstack((bn2_X, bndX[1])))]
        #    bndY = [np.min(np.hstack((bn2_Y, bndY[0]))),
        #    np.max(np.hstack((bn2_Y, bndY[1])))]

        #print('')

        X = np.arange(bndX[0], bndX[1], (bndX[1] - bndX[0]) / 100.0)
        Y = np.arange(bndY[0], bndY[1], (bndY[1] - bndY[0]) / 100.0)
        X, Y = np.meshgrid(X, Y)
        Z = self.seabed.Z(X,Y)

        _ = ax.scatter(self.auv.X[0], self.auv.X[1], self.auv.X[2], color = 'black', s = 30)
    
        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.3)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


        v_scale = 20.0
        Vplot = np.vstack((self.auv.X, self.auv.X + v_scale * self.delta * self.auv.V(0))) 

        arrow = Arrow3D(Vplot[:,0], Vplot[:,1], Vplot[:,2], mutation_scale=10, arrowstyle="-|>", color="black")
        ax.add_artist(arrow)
        
        #_ = ax.scatter(Vplot[0,0], Vplot[0,1], Vplot[0,2], color = 'red', s =
        #100)
        #_ = ax.scatter(Vplot[1,0], Vplot[1,1], Vplot[1,2], color = 'red', s =
        #10)
        #_ = ax.plot(Vplot[:,0], Vplot[:,1], Vplot[:,2], color = 'red')

        # Add a color bar which maps values to colors.
        #fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def plottrajectory3Dseabed(self):
        fig = plt.figure(figsize=(10, 6), dpi=200)
        ax = fig.gca(projection='3d')
        ax.view_init(50, 30)

        for t in range(0, self.Npoints):
            print(self.delta * t)
            self.auv.step()

        bndX = [self.auv.X_history[:,0].min(), self.auv.X_history[:,0].max()]
        bndY = [self.auv.X_history[:,1].min(), self.auv.X_history[:,1].max()]

        X = np.arange(bndX[0], bndX[1], (bndX[1] - bndX[0]) / 100.0)
        Y = np.arange(bndY[0], bndY[1], (bndY[1] - bndY[0]) / 100.0)
        X, Y = np.meshgrid(X, Y)
        Z = self.seabed.Z(X,Y)

        _ = ax.plot(self.auv.X_history[:,0], self.auv.X_history[:,1], self.auv.X_history[:,2], color = 'black')
    
        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.3)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


        # Add a color bar which maps values to colors.
        #fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


    def showbearing(self, path, show=False): # model scheme with basic notation
        fig = plt.figure(figsize=(10, 6), dpi=200)
        ax = fig.gca(projection='3d')
        ax.view_init(31, 26)

        # set good viewpoint
        drawPoint(ax, [1.1, 1.1, 1.1], color = 'white', s = 0)


        s = 0.1
        Xk = np.array([0.7,0.2,0.4])
        XB = np.array([0.2,0.5,0.81])
        xx = np.array([XB[0], Xk[1], 0.0])

        #points
        drawPoint(ax, Xk)
        drawPoint(ax, XB)
        textAtPoint(ax, Xk, '${\mathbf{X}}_{k}$', [0,-0.7*s,0.2*s])
        textAtPoint(ax, XB, '${\mathbf{X}}_{B}$ (beacon)', [0,-0.7*s,0])


        #Xk projections
        drawLine(ax, Xk, PZ(Xk), linestyle=':', linewidth=0.5)
        drawLine(ax, PZ(Xk), Y(Xk), linestyle=':', linewidth=0.5)
        drawLine(ax, PZ(Xk), X(Xk), linestyle=':', linewidth=0.5)

        #XB projections
        drawLine(ax, XB, PZ(XB), linestyle=':', linewidth=0.5)
        drawLine(ax, PZ(XB), Y(XB), linestyle=':', linewidth=0.5)
        drawLine(ax, PZ(XB), X(XB), linestyle=':', linewidth=0.5)

        #Xk to XB and projections
        drawLine(ax, Xk, XB, linestyle=':', linewidth=0.5)
        drawLine(ax, PZ(Xk), PZ(XB), linestyle=':', linewidth=0.5)
        drawLine(ax, Xk, [XB[0],XB[1],Xk[2]], linestyle=':', linewidth=0.5)

        #deltas
        drawLine(ax, PZ(Xk), xx, linestyle='-', linewidth=0.5)
        drawLine(ax, PZ(XB), xx, linestyle='-', linewidth=0.5)
        drawLine(ax, XB, [XB[0],XB[1],Xk[2]], linestyle='-', linewidth=0.5)
        textAtPoint(ax, 0.5*(PZ(Xk) + xx), '$X_B - X_k$', [0,-1.3*s,0])
        textAtPoint(ax, 0.5*(PZ(XB) + xx), '$Y_B - Y_k$', [0.0,-0.4*s,0])
        textAtPoint(ax, 0.5*(PZ(XB) + XB), '$Z_B - Z_k$', [0,0.2*s,1.3*s])


        sX = np.arange(-0.1, 1.1, 1.0 / 100.0)
        sY = np.arange(-0.1, 1.05, 1.0 / 100.0)
        sX, sY = np.meshgrid(sX, sY)
        sZ = XB[2] * np.ones_like(sX)
        # Plot the surface.
        surf = ax.plot_surface(sX, sY, sZ, cmap=cm.Blues, linewidth=0, antialiased=False, alpha=0.1)
        ax.text(-0.1, 0.6, XB[2], 'sea surface', (0,1,0.1))


        #angles
        drawArcScale(ax, Xk, XB, [XB[0],XB[1],Xk[2]], scale = 0.10)
        textAtPoint(ax, Xk, '$\lambda_k$', [-1.1*s, 0.4*s, 0.0])

        drawArcScale(ax, PZ(Xk), PZ(XB),  [XB[0], Xk[1], 0], scale = 0.15)
        drawArcScale(ax, PZ(Xk), PZ(XB),  [XB[0], Xk[1], 0], scale = 0.18)
        textAtPoint(ax, PZ(Xk), '$\\varphi_k$', [0.3*s, 0.8*s, 0])

        #axes
        drawArrow(ax, [-0.01, 0.0, 0.0], [1.0, 0.0, 0.0])
        drawArrow(ax, [0.0, -0.01, 0.0], [0.0, 1.0, 0.0])
        drawArrow(ax, [0.0, 0.0, -0.01], [0.0, 0.0, 1.0])
        textAtPoint(ax, [1.0, 0.05, 0.0], '$x$')
        textAtPoint(ax, [0.1, 1.0, 0.0], '$y$')
        textAtPoint(ax, [0.05, 0.05, 1.0], '$z$')


        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.set_axis_off()
        fig.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(path)

        
    def showsonarmodel(self, path, show=False): # model scheme with basic notation
        fig = plt.figure(figsize=(10, 6), dpi=200)
        ax = fig.gca(projection='3d')
        ax.view_init(31, 26)

        # set good viewpoint
        drawPoint(ax, [1.1, 1.1, 1.1], color = 'white', s = 0)


        s = 0.1
        Xk = np.array([0.2,0.3,0.7])
        v = np.array([-0.05,0.1,0.05])
        x = np.array([0.7,0.8,0.2])

        #points
        drawPoint(ax, Xk)
        drawPoint(ax, x)
        textAtPoint(ax, Xk, '${\mathbf{X}}_{k}$', [0,-0.7*s,0])
        textAtPoint(ax, (x+Xk)*0.5, '$L_{k}^{ij}$', [0,0.2*s,0])
        textAtPoint(ax, x, '$\mathbf{x}_{k}^{ij}$', [0,0.2*s,0])
        for i in range(-2,3):
            for j in range(-2,3):
                drawPoint(ax, x+np.array([i*0.05, j*0.05,0]),'black', 10, alpha=0.1)
                drawLine(ax, Xk, x+np.array([i*0.05, j*0.05,0]), 'black', linestyle=':', linewidth=0.5, alpha = 0.2)


        #Xk projections
        drawLine(ax, Xk, PZ(Xk), linestyle=':', linewidth=0.5)
        drawLine(ax, [x[0], Xk[1], 0], Y(Xk), linestyle=':', linewidth=0.5)
        drawLine(ax, PZ(Xk), X(Xk), linestyle=':', linewidth=0.5)

        #x projections
        drawLine(ax, x, PZ(x), linestyle=':', linewidth=0.5)
        drawLine(ax, PZ(x), Y(x), linestyle=':', linewidth=0.5)
        drawLine(ax, PZ(x), X(x), linestyle=':', linewidth=0.5)

        #Xk to x and projections
        drawLine(ax, PZ(Xk), PZ(x), linestyle=':', linewidth=0.5)
        drawLine(ax, Xk, [x[0],x[1],Xk[2]], linestyle=':', linewidth=0.5)

        sX = np.arange(-0.1, 1.1, 1.0 / 100.0)
        sY = np.arange(-0.1, 1.05, 1.0 / 100.0)
        sX, sY = np.meshgrid(sX, sY)
        sZ = x[2] * np.ones_like(sX)
        # Plot the surface.
        surf = ax.plot_surface(sX, sY, sZ, cmap=cm.copper, linewidth=0, antialiased=False, alpha=0.1)
        ax.text(-0.1, 0.6, x[2], 'seabed', (0,1,0.1))

        #vectors
        drawArrow(ax, Xk, Xk+v)
        textAtPoint(ax, Xk+v, '${\mathbf{V}}_k^*$', [0.0, -0.5*s, 0.0])
        drawLine(ax, Xk, x)

        #angles
        drawArcScale(ax, Xk, x, [x[0],x[1],Xk[2]], scale = 0.10)
        textAtPoint(ax, Xk, '$\gamma_k + \gamma^i$', [s, s, 0.0])

        drawArcScale(ax, PZ(Xk), PZ(x),  [x[0], Xk[1], 0], scale = 0.15)
        drawArcScale(ax, PZ(Xk), PZ(x),  [x[0], Xk[1], 0], scale = 0.18)
        textAtPoint(ax, PZ(Xk), '$\\theta_k + \\theta^j$', [2.5*s, 0.5*s, 0])

        #axes
        drawArrow(ax, [-0.01, 0.0, 0.0], [1.0, 0.0, 0.0])
        drawArrow(ax, [0.0, -0.01, 0.0], [0.0, 1.0, 0.0])
        drawArrow(ax, [0.0, 0.0, -0.01], [0.0, 0.0, 1.0])
        textAtPoint(ax, [1.0, 0.05, 0.0], '$x$')
        textAtPoint(ax, [0.1, 1.0, 0.0], '$y$')
        textAtPoint(ax, [0.05, 0.05, 1.0], '$z$')


        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.set_axis_off()
        fig.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(path)

    def showoptimalcontrol(self, path, show=False): # model scheme with basic notation
        fig = plt.figure(figsize=(10, 6), dpi=200)
        ax = fig.gca(projection='3d')
        ax.view_init(31, 26)

        # set good viewpoint
        drawPoint(ax, [1.1, 1.1, 1.1], color = 'white', s = 0)


        s = 0.1
        XNk = np.array([0.5,0.3,0.5])
        XNk1 = np.array([0.2,0.7,0.8])
        Xk = np.array([0.7,0.5,0.2])

        #points
        drawPoint(ax, XNk)
        drawPoint(ax, XNk1)
        drawPoint(ax, Xk)
        textAtPoint(ax, XNk, '$\mathring{\mathbf{X}}_k$', [0,0,s])
        textAtPoint(ax, XNk1, '$\mathring{\mathbf{X}}_{k+1}$', [0,0,s])
        textAtPoint(ax, Xk, '${\mathbf{X}}_{k}$', [0,-s,-s])

        #XNk1 projections
        drawLine(ax, XNk1, PZ(XNk1), linestyle=':', linewidth=0.5)
        drawLine(ax, PZ(XNk1), Y(XNk1), linestyle=':', linewidth=0.5)
        drawLine(ax, PZ(XNk1), X(XNk1), linestyle=':', linewidth=0.5)

        #Xk projections
        drawLine(ax, Xk, PZ(Xk), linestyle=':', linewidth=0.5)
        drawLine(ax, PZ(Xk), Y(Xk), linestyle=':', linewidth=0.5)
        drawLine(ax, PZ(Xk), X(Xk), linestyle=':', linewidth=0.5)

        #Xk to XNk1 and projections
        drawLine(ax, PZ(Xk), PZ(XNk1), linestyle=':', linewidth=0.5)
        drawLine(ax, Xk, [XNk1[0],XNk1[1],Xk[2]], linestyle=':', linewidth=0.5)
        drawLine(ax, 0.5*(Xk + XNk1), XNk1, linestyle=':', linewidth=0.5)

        #deltas
        drawLine(ax, PZ(Xk), [Xk[0], XNk1[1], 0], linestyle='-', linewidth=0.5)
        drawLine(ax, PZ(XNk1), [Xk[0], XNk1[1], 0], linestyle='-', linewidth=0.5)
        drawLine(ax, XNk1, [XNk1[0], XNk1[1], Xk[2]], linestyle='-', linewidth=0.5)
        textAtPoint(ax, [0.7*Xk[0] + 0.3*XNk1[0], XNk1[1], 0.0], '$\Delta \mathring{X}_k + \Delta t \mathring{V}^X_k$', [0.0,s,0.0])
        textAtPoint(ax, [Xk[0], 0.8*Xk[1] + 0.2*XNk1[1], 0.0], '$\Delta \mathring{Y}_k + \Delta t \mathring{V}^Y_k$', [2.0*s, 0.0, -0.5*s])
        textAtPoint(ax, [XNk1[0], XNk1[1], 0.5*Xk[2] + 0.5*XNk1[2]], '$\Delta \mathring{Z}_k + \Delta t \mathring{V}^Z_k$', [0.0, 0.5*s, 0.0])


        #vectors
        drawArrow(ax, Xk, XNk)
        drawArrow(ax, XNk, XNk1)
        drawArrow(ax, Xk, 0.5*(Xk + XNk1))
        textAtPoint(ax, 0.5*(Xk + XNk1), '${\mathbf{V}}_k^*$', [0.0, -0.5*s, 0.0])

        #angles
        drawArcScale(ax, Xk, XNk1, [XNk1[0], XNk1[1], Xk[2]], scale = 0.10)
        textAtPoint(ax, Xk, '$\gamma^*_k$', [0.0, s, 0.0])

        drawArcScale(ax, [XNk1[0], XNk1[1], 0.0], [Xk[0], Xk[1], 0.0], [Xk[0], XNk1[1], 0.0], scale = 0.15)
        drawArcScale(ax, [XNk1[0], XNk1[1], 0.0], [Xk[0], Xk[1], 0.0], [Xk[0], XNk1[1], 0.0], scale = 0.18)
        textAtPoint(ax, PZ(XNk1), '$\\theta^*_k$', [1.5*s, -s, 0])

        #axes
        drawArrow(ax, [-0.01, 0.0, 0.0], [1.0, 0.0, 0.0])
        drawArrow(ax, [0.0, -0.01, 0.0], [0.0, 1.0, 0.0])
        drawArrow(ax, [0.0, 0.0, -0.01], [0.0, 0.0, 1.0])
        textAtPoint(ax, [1.0, 0.05, 0.0], '$x$')
        textAtPoint(ax, [0.1, 1.0, 0.0], '$y$')
        textAtPoint(ax, [0.05, 0.05, 1.0], '$z$')


        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.set_axis_off()
        fig.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(path)


    def showmodel(self, path, show=False): # model scheme with basic notation
        fig = plt.figure(figsize=(10, 6), dpi=200)
        ax = fig.gca(projection='3d')
        ax.view_init(31, 26)

        # set good viewpoint
        drawPoint(ax, [0.8, 0.8, 0.8], color = 'white', s = 0)


        s = 0.05
        Xk = np.array([0.7,0.2,0.2])
        XNk1 = np.array([0.2,0.4,0.8])
        v = 0.5*(Xk + XNk1)

        #points
        drawPoint(ax, Xk)
        textAtPoint(ax, Xk, '${\mathbf{X}}_{k}$', [0,-s,-s])

        #Xk projections
        drawLine(ax, Xk, PZ(Xk), linestyle=':', linewidth=0.5)
        drawLine(ax, PZ(Xk), Y(Xk), linestyle=':', linewidth=0.5)

        #Xk to XNk1 and projections
        drawLine(ax, PZ(Xk), PZ(v), linestyle=':', linewidth=0.5)
        drawLine(ax, Xk, [v[0],v[1],Xk[2]], linestyle=':', linewidth=0.5)
        drawLine(ax, v, PZ(v), linestyle=':', linewidth=0.5)

        #vectors
        drawArrow(ax, Xk, v)
        textAtPoint(ax, v, '${\mathbf{V}}_k^*$', [0.0, -0.5*s, 0.0])

        #angles
        drawArcScale(ax, Xk, XNk1, [XNk1[0], XNk1[1], Xk[2]], scale = 0.10)
        textAtPoint(ax, Xk, '$\gamma^*_k$', [-s, s, s])

        drawArcScale(ax, PZ(Xk), PZ(v), Y(Xk), scale = 0.30)
        drawArcScale(ax, PZ(Xk), PZ(v), Y(Xk), scale = 0.36)
        textAtPoint(ax, PZ(Xk), '$\\theta^*_k$', [-s, 1.5*s, 1.5*s])

        #axes
        drawArrow(ax, [-0.01, 0.0, 0.0], [.5, 0.0, 0.0])
        drawArrow(ax, [0.0, -0.005, 0.0], [0.0, .25, 0.0])
        drawArrow(ax, [0.0, 0.0, -0.01], [0.0, 0.0, .5])
        textAtPoint(ax, [.5, 0.025, 0.0], '$x$')
        textAtPoint(ax, [0.1, .25, 0.0], '$y$')
        textAtPoint(ax, [0.025, 0.025, .5], '$z$')


        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.set_axis_off()
        fig.tight_layout()

        if show:
            plt.show()
        else:
            plt.savefig(path)
        
 