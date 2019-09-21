from BasicModel.AUV import *
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import datetime

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


from matplotlib import rc
rc('font',**{'family':'serif'})
rc('text', usetex=True)
rc('text.latex',unicode=True)
rc('text.latex',preamble=r'\usepackage[T2A]{fontenc}')
rc('text.latex',preamble=r'\usepackage[utf8]{inputenc}')
rc('text.latex',preamble=r'\usepackage[russian]{babel}')

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

class TestEnvironment():
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
            self.auv.step()
            print(self.delta * i)
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
            ax.plot(self.auv.t_history, self.auv.X_history[:,k], color='black', linewidth=2.0, linestyle=':')
            for i,s in enumerate(self.auv.Sensors):
                ax.plot(self.auv.t_history, s.X_estimate_history[:,k], color=self.colors[i], label=str(i))
            ax.plot(self.auv.t_history, self.auv.X_estimate_history[:,k], color='black', linewidth=2.0)
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
                np.array2string(self.auv.X_estimate_history[self.Npoints - 1,:] - self.auv.X_history[self.Npoints - 1,:], 
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
            ax.plot(self.auv.t_history, self.auv.V_history[:,k], color='black', linewidth=2.0)
            for i,s in enumerate(self.auv.Sensors):
                if (i == sonars_xyz[k]):
                    ax.plot(self.auv.t_history[:], s.delta_X_estimate_history[:,k] / self.delta, color = 'red', label=str(i)) #color=self.colors[i]
            #ax.plot(self.auv.t_history, self.auv.X_estimate_history[:,k],
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
           
    def plot3Dtrajectory(self, sonars_xyz, path):
        #start = datetime.datetime.now()
        #self.run()
        #finish = datetime.datetime.now()
        fig = plt.figure(num=None, figsize=(15,5), dpi=200, facecolor='w', edgecolor='k')     
        gs = gridspec.GridSpec(1, 1)     
        ax = fig.add_subplot(gs[:], projection='3d')
        _ = ax.plot(self.auv.X_history[:,0], self.auv.X_history[:,1], self.auv.X_history[:,2], color = 'blue')
        _ = ax.plot(self.auv.Sensors[sonars_xyz[0]].X_estimate_history[:,0], self.auv.Sensors[sonars_xyz[1]].X_estimate_history[:,1], self.auv.Sensors[sonars_xyz[2]].X_estimate_history[:,2], color = 'black')

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
                #if isinstance(s, SensorControlled):
                #    bn, _, _, _, _ = s.beamnet(self.auv.X, self.auv.U(self.auv.t))
                #else:  
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

        
    def showscheme(self): # model scheme with basic notation
        fig = plt.figure(figsize=(10, 6), dpi=200)
        ax = fig.gca(projection='3d')
        ax.view_init(50, 30)

        bndX = [self.auv.X[0], self.auv.X[0]]
        bndY = [self.auv.X[1], self.auv.X[1]]
        
        
        for i, s in enumerate(self.auv.Sensors):
            bn, _, _, _, _ = s.beamnet(self.auv.X)
            bn2_X, bn2_Y, bn2_Z = bn[:,0], bn[:,1], bn[:,2]
            bndX = [np.min(np.hstack((bn2_X, bndX[0]))), np.max(np.hstack((bn2_X, bndX[1])))]
            bndY = [np.min(np.hstack((bn2_Y, bndY[0]))), np.max(np.hstack((bn2_Y, bndY[1])))]
            if i == 4: # take only one
                #_ = ax.scatter(bn2_X[:], bn2_Y[:], bn2_Z[:], color =
                                      #self.colors[i], s = 30)
                x, y, z = bn2_X[-8], bn2_Y[-8], bn2_Z[-8]
        print('')

        xa, ya, za = self.auv.X[0], self.auv.X[1], self.auv.X[2]


        _ = ax.scatter(x, y, z, color = 'black', s = 30)
        _ = ax.plot([xa, x], [ya, y], [za, z], color = 'black')
        _ = ax.plot([xa, x], [ya, y], [-35, -35], color = 'black', linewidth = 0.5)


        #_ = ax.plot([x, 0], [y, y], [z, z], color = 'black', linestyle=':')
        #_ = ax.plot([x, x], [y, 0], [z, z], color = 'black', linestyle=':')
        #_ = ax.plot([0, 0], [0, y], [z, z], color = 'black', linestyle=':')
        #_ = ax.plot([0, x], [0, 0], [z, z], color = 'black', linestyle=':')
        #_ = ax.plot([0, 0], [y, y], [-35, z], color = 'black', linestyle=':')
        #_ = ax.plot([x, x], [0, 0], [-35, z], color = 'black', linestyle=':')
        _ = ax.plot([x, 0], [y, y], [-35, -35], color = 'black', linestyle=':')
        _ = ax.plot([x, x], [y, 0], [-35, -35], color = 'black', linestyle=':')   
        _ = ax.plot([x, x], [y, y], [-35, z], color = 'black', linestyle=':')


        _ = ax.scatter(xa, ya, za, color = 'black', s = 100)
        _ = ax.plot([xa + 5.0, 0], [ya, ya], [-35, -35], color = 'black', linestyle=':')
        _ = ax.plot([xa, xa], [ya, 0], [-35, -35], color = 'black', linestyle=':')   
        _ = ax.plot([xa, xa], [ya, ya], [-35, za], color = 'black', linestyle=':')
    
        xb,yb,zb = xa + (x - xa) * 0.15, ya + (y - ya) * 0.15, za + (z - za) * 0.15 
        r = np.sqrt(np.power(xb - xa, 2) + np.power(yb - ya, 2) + np.power(zb - za,2))
        x_angle = np.concatenate([np.arange(xa,xb,(xb - xa) / 20.0), np.array([xb])])
        y_angle = np.concatenate([np.arange(ya,yb,(yb - ya) / 20.0), np.array([yb])])
        z_angle = za - np.sqrt(np.power(r, 2) - np.power(x_angle - xa, 2) - np.power(y_angle - ya, 2))
        _ = ax.plot(x_angle, y_angle, z_angle, color = 'black', linewidth = 0.5)

        ax.text(xa + (xb - xa) / 2, ya + (yb - ya) / 2, zb - 3.0 , '$\\phi_i$')
        
        xb,yb,zb = xa + (x - xa) * 0.15, ya + (y - ya) * 0.15, za + (z - za) * 0.15 
        r = np.sqrt(np.power(xb - xa, 2) + np.power(yb - ya, 2))
        y_angle = np.concatenate([np.arange(ya,yb,(yb - ya) / 20.0), np.array([yb])])
        x_angle = xa + np.sqrt(np.power(r, 2) - np.power(y_angle - ya, 2)) 
        _ = ax.plot(x_angle, y_angle, -35, color = 'black', linewidth = 0.5)
        xb,yb,zb = xa + (x - xa) * 0.18, ya + (y - ya) * 0.18, za + (z - za) * 0.18 
        r = np.sqrt(np.power(xb - xa, 2) + np.power(yb - ya, 2))
        y_angle = np.concatenate([np.arange(ya,yb,(yb - ya) / 20.0), np.array([yb])])
        x_angle = xa + np.sqrt(np.power(r, 2) - np.power(y_angle - ya, 2)) 
        _ = ax.plot(x_angle, y_angle, -35, color = 'black', linewidth = 0.5)

        ax.text(xa + 7.0, ya + 3.0, -35.7 , '$\\theta_k$')

        X = np.arange(bndX[0], bndX[1], (bndX[1] - bndX[0]) / 100.0)
        Y = np.arange(bndY[0], bndY[1], (bndY[1] - bndY[0]) / 100.0)
        X, Y = np.meshgrid(X, Y)
        Z = self.seabed.Z(X,Y)

        axisX = Arrow3D([-2, 35], [0, 0], [-35, -35], mutation_scale=10, arrowstyle="-|>", color="black")
        axisY = Arrow3D([0, 0], [-2, 35], [-35, -35], mutation_scale=10, arrowstyle="-|>", color="black")
        axisZ = Arrow3D([0, 0], [0, 0], [-36, -5], mutation_scale=10, arrowstyle="-|>", color="black")
        ax.add_artist(axisX)
        ax.add_artist(axisY)
        ax.add_artist(axisZ)

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.3)

        ax.text(xa, ya + 1.0, za + 1.0, '$\mathbf{X}(t_j)$')
        ax.text(x, y + 1.0, z + 1.0, '$(x,y,z)_{ik,j}^T$')
        ax.text(xa + (x - xa) / 2, ya + (y - ya) / 2 + 1.0, za + (z - za) / 2 + 1.0, '$L_{ik,j}$')

        ax.text(35, 1.0, -36, '$x$')
        ax.text(1.0, 35, -36, '$y$')
        ax.text(1.0, 1.0, -5, '$z$')


        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.set_axis_off()
        # Add a color bar which maps values to colors.
        #fig.colorbar(surf, shrink=0.5, aspect=5)
        #fig.tight_layout()
        #fig.subplots_adjust(left=-0.2, right=1.2, top=1.2, bottom =-0.2)
        plt.show()
