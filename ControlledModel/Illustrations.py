import matplotlib.pyplot as plt
from matplotlib import gridspec

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
#import datetime

from mpl_toolkits.mplot3d import Axes3D

#import sys
#sys.path.append('..\\Utils\\')

from Utils.DrawHelper import *

from matplotlib import rc
rc('font',**{'family':'serif'})
rc('text', usetex=True)
rc('text.latex',unicode=True)
rc('text.latex',preamble=r'\usepackage[T2A]{fontenc}')
rc('text.latex',preamble=r'\usepackage[utf8]{inputenc}')
rc('text.latex',preamble=r'\usepackage[russian]{babel}')

colors = ['red', 'blue', 'green']


def showstats(path, t_history, mean, std, show=False): # model scheme with basic notation
    fig = plt.figure(figsize=(10, 7), dpi=200, facecolor='w', edgecolor='k')
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.2, 1, 1, 1]) 
    gs.update(left=0.07, bottom=0.05, right=0.99, top=0.99, wspace=0.1, hspace=0.1)   

    labels_mean = ['$\mathbf{E}(\mathbf{X}_k - \hat{\mathbf{X}}_k^{CMNF})$', '$\mathbf{E}(\mathbf{X}_k - \hat{\mathbf{X}}_k^{PMKF})$', '$\mathbf{E}(\mathbf{X}_k - \hat{\mathbf{X}}_k^{EKF})$']
    labels_std = ['$\sigma(\mathbf{X}_k - \hat{\mathbf{X}}_k^{CMNF})$', '$\sigma(\mathbf{X}_k - \hat{\mathbf{X}}_k^{PMKF})$', '$\sigma(\mathbf{X}_k - \hat{\mathbf{X}}_k^{EKF})$']

    #labels_mean = ['$\mathbf{E}(\mathbf{X}_k - \hat{\mathbf{X}}_k^{CMNF})$','','']
    #labels_std = ['','','']

    ax = [None] * 3
    for j in range(1,4):
        i = j - 1
        ax[i] = plt.subplot(gs[j])
        for k in range(0, len(mean)):
            ax[i].plot(t_history, mean[k][:,i], color = colors[k], linewidth = 1.0, alpha = 0.7, linestyle=':')
            ax[i].plot(t_history, std[k][:,i], color = colors[k], linewidth = 2.0, alpha = 0.7)

        ax[i].set_xticks([])

    ax_l = plt.subplot(gs[0])
    for k in range(0, len(mean)):
        ax_l.plot([], [], color = colors[k], linewidth = 1.0, alpha = 0.7, linestyle=':', label = labels_mean[k])
        ax_l.plot([], [], color = colors[k], linewidth = 2.0, alpha = 0.7, label = labels_std[k])
    ax_l.legend(loc='lower center', ncol=len(mean), fancybox=True, bbox_to_anchor = (0.5,-0.5))
    ax_l.set_axis_off()

    ax[2].set_yticks([0, 1, 2])
    ax[2].set_yticklabels(['0', '1', '$Z_k - \hat{Z}_k$'])
    ax[2].set_ylim(-0.5,2.0)

    ax[2].set_xticks([0, 150, 300])
    ax[2].set_xticklabels(['0 min', 'time [mins]', '5 min'])

    #ax[1].set_yticks([-1, 0, 1, 2])
    #ax[1].set_yticklabels(['-1', '0', '1', '$Y_k - \hat{Y}_k$'])
    #ax[1].set_ylim(-1.5,2)
    ax[1].set_yticks([0, 1, 2])
    ax[1].set_yticklabels(['0', '1', '$Y_k - \hat{Y}_k$'])
    ax[1].set_ylim(-0.5,2)

    #ax[0].set_yticks([-2,  0, 2, 3])
    #ax[0].set_yticklabels(['-2', '0', '2', '$X_k - \hat{X}_k$'])
    #ax[0].set_ylim(-3.5,3)
    ax[0].set_yticks([0, 2, 3])
    ax[0].set_yticklabels(['0', '2', '$X_k - \hat{X}_k$'])
    ax[0].set_ylim(-0.5,3)

    #fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(path)

def showstats_pw(path, t_history, mean, std, show=False): # model scheme with basic notation
    fig = plt.figure(figsize=(10, 7), dpi=200, facecolor='w', edgecolor='k')
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.2, 1, 1, 1]) 
    gs.update(left=0.07, bottom=0.05, right=0.99, top=0.99, wspace=0.1, hspace=0.1)   

    labels_mean = ['$\mathbf{E}(\mathbf{X}_k - \hat{\mathbf{X}}_k^{CMNF})$', '$\mathbf{E}(\mathbf{X}_k - \hat{\mathbf{X}}_k^{PMKF})$', '$\mathbf{E}(\mathbf{X}_k - \hat{\mathbf{X}}_k^{EKF})$']
    labels_std = ['$\sigma(\mathbf{X}_k - \hat{\mathbf{X}}_k^{CMNF})$', '$\sigma(\mathbf{X}_k - \hat{\mathbf{X}}_k^{PMKF})$', '$\sigma(\mathbf{X}_k - \hat{\mathbf{X}}_k^{EKF})$']

    #labels_mean = ['$\mathbf{E}(\mathbf{X}_k - \hat{\mathbf{X}}_k^{CMNF})$','','']
    #labels_std = ['','','']

    t_history = np.hstack((t_history[0], np.repeat(t_history[1:],2)))
    for k in range(0, len(mean)):
        mean[k] = np.vstack((np.repeat(mean[k][0:-1], 2, axis = 0), mean[k][-1].reshape(1, mean[k].shape[1])))
        std[k] = np.vstack((np.repeat(std[k][0:-1], 2, axis = 0), std[k][-1].reshape(1, std[k].shape[1])))

    ax = [None] * 3
    for j in range(1,4):
        i = j - 1
        ax[i] = plt.subplot(gs[j])
        for k in range(0, len(mean)):
            ax[i].plot(t_history, mean[k][:,i], color = colors[k], linewidth = 1.0, alpha = 0.7, linestyle=':')
            ax[i].plot(t_history, std[k][:,i], color = colors[k], linewidth = 2.0, alpha = 0.7)

        ax[i].set_xticks([])

    ax_l = plt.subplot(gs[0])
    for k in range(0, len(mean)):
        ax_l.plot([], [], color = colors[k], linewidth = 1.0, alpha = 0.7, linestyle=':', label = labels_mean[k])
        ax_l.plot([], [], color = colors[k], linewidth = 2.0, alpha = 0.7, label = labels_std[k])
    ax_l.legend(loc='lower center', ncol=len(mean), fancybox=True, bbox_to_anchor = (0.5,-0.5))
    ax_l.set_axis_off()

    ax[2].set_yticks([0, 1, 2])
    ax[2].set_yticklabels(['0', '1', '$Z_k - \hat{Z}_k$'])
    ax[2].set_ylim(-0.5,2.0)

    ax[2].set_xticks([0, 150, 300])
    ax[2].set_xticklabels(['0 min', 'time [mins]', '5 min'])

    ax[1].set_yticks([-1, 0, 1, 2])
    ax[1].set_yticklabels(['-1', '0', '1', '$Y_k - \hat{Y}_k$'])
    ax[1].set_ylim(-1.5,2)
    #ax[1].set_yticks([0, 1, 2])
    #ax[1].set_yticklabels(['0', '1', '$Y_k - \hat{Y}_k$'])
    #ax[1].set_ylim(-0.5,2)

    ax[0].set_yticks([-2,  0, 2, 3])
    ax[0].set_yticklabels(['-2', '0', '2', '$X_k - \hat{X}_k$'])
    ax[0].set_ylim(-3.5,3)
    #ax[0].set_yticks([0, 2, 3])
    #ax[0].set_yticklabels(['0', '2', '$X_k - \hat{X}_k$'])
    #ax[0].set_ylim(-0.5,3)

    #fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(path)


def showstats_part(path, t_history, mean, std, show=False): # model scheme with basic notation
    fig = plt.figure(figsize=(7, 7), dpi=200, facecolor='w', edgecolor='k')
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.3, 1, 1, 1]) 
    gs.update(left=0.10, bottom=0.05, right=0.99, top=0.99, wspace=0.1, hspace=0.1)   

    labels_mean = ['$\mathbf{E}(\mathbf{X}_k - \hat{\mathbf{X}}_k^{CMNF})$', '$\mathbf{E}(\mathbf{X}_k - \hat{\mathbf{X}}_k^{PMKF})$', '$\mathbf{E}(\mathbf{X}_k - \hat{\mathbf{X}}_k^{EKF})$']
    labels_std = ['$\sigma(\mathbf{X}_k - \hat{\mathbf{X}}_k^{CMNF})$', '$\sigma(\mathbf{X}_k - \hat{\mathbf{X}}_k^{PMKF})$', '$\sigma(\mathbf{X}_k - \hat{\mathbf{X}}_k^{EKF})$']

    #labels_mean = ['$\mathbf{E}(\mathbf{X}_k - \hat{\mathbf{X}}_k^{CMNF})$','','']
    #labels_std = ['','','']

    ax = [None] * 3
    for j in range(1,4):
        i = j - 1
        ax[i] = plt.subplot(gs[j])
        for k in range(0, len(mean)):
            ax[i].plot(t_history, mean[k][:,i], color = colors[k], linewidth = 1.0, alpha = 0.7, linestyle=':')
            ax[i].plot(t_history, std[k][:,i], color = colors[k], linewidth = 2.0, alpha = 0.7)

        ax[i].set_xticks([])

    ax_l = plt.subplot(gs[0])
    for k in range(0, len(mean)):
        ax_l.plot([], [], color = colors[k], linewidth = 1.0, alpha = 0.7, linestyle=':', label = labels_mean[k])
        ax_l.plot([], [], color = colors[k], linewidth = 2.0, alpha = 0.7, label = labels_std[k])
    ax_l.legend(loc='lower center', ncol=len(mean), fancybox=True, bbox_to_anchor = (0.5,-0.3))
    ax_l.set_axis_off()

    ax[2].set_yticks([0, 2, 4.5])
    ax[2].set_yticklabels(['0', '2', '$Z_k - \hat{Z}_k$'])
    ax[2].set_ylim(-1.5,4.5)

    ax[2].set_xticks([0, 5, 10])
    ax[2].set_xticklabels(['0 sec', 'time [sec]', '10 sec'])

    ax[1].set_yticks([0, 2, 4.5])
    ax[1].set_yticklabels(['0', '2', '$Y_k - \hat{Y}_k$'])
    ax[1].set_ylim(-1.5,4.5)

    ax[0].set_yticks([ 0, 2, 4.5])
    ax[0].set_yticklabels(['0', '2', '$X_k - \hat{X}_k$'])
    ax[0].set_ylim(-1.5,4.5)

    #fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(path)

def showstats_part_pw(path, t_history, mean, std, show=False): # model scheme with basic notation
    fig = plt.figure(figsize=(7, 5), dpi=200, facecolor='w', edgecolor='k')
    gs = gridspec.GridSpec(2, 3, height_ratios=[0.12, 1], width_ratios=[1,1,1]) 
    gs.update(left=0.05, bottom=0.05, right=0.98, top=0.99, wspace=0.15, hspace=0.16)   

    labels_mean = ['$\mathbf{E}(\mathbf{X}_k - \hat{\mathbf{X}}_k^{CMNF})$', '$\mathbf{E}(\mathbf{X}_k - \hat{\mathbf{X}}_k^{PMKF})$', '$\mathbf{E}(\mathbf{X}_k - \hat{\mathbf{X}}_k^{EKF})$']
    labels_std = ['$\sigma(\mathbf{X}_k - \hat{\mathbf{X}}_k^{CMNF})$', '$\sigma(\mathbf{X}_k - \hat{\mathbf{X}}_k^{PMKF})$', '$\sigma(\mathbf{X}_k - \hat{\mathbf{X}}_k^{EKF})$']

    #labels_mean = ['$\mathbf{E}(\mathbf{X}_k - \hat{\mathbf{X}}_k^{CMNF})$','','']
    #labels_std = ['','','']

    t_history = np.hstack((t_history[0], np.repeat(t_history[1:],2)))
    for k in range(0, len(mean)):
        mean[k] = np.vstack((np.repeat(mean[k][0:-1], 2, axis = 0), mean[k][-1].reshape(1, mean[k].shape[1])))
        std[k] = np.vstack((np.repeat(std[k][0:-1], 2, axis = 0), std[k][-1].reshape(1, std[k].shape[1])))

    ax = [None] * 3
    for j in range(1,4):
        i = j - 1
        ax[i] = plt.subplot(gs[1,i])
        for k in range(0, len(mean)):
            ax[i].plot(t_history, mean[k][:,i], color = colors[k], linewidth = 1.0, alpha = 0.7, linestyle=':')
            ax[i].plot(t_history, std[k][:,i], color = colors[k], linewidth = 2.0, alpha = 0.7)

        ax[i].set_xticks([])

    ax_l = plt.subplot(gs[0, :])
    for k in range(0, len(mean)):
        ax_l.plot([], [], color = colors[k], linewidth = 1.0, alpha = 0.7, linestyle=':', label = labels_mean[k])
        ax_l.plot([], [], color = colors[k], linewidth = 2.0, alpha = 0.7, label = labels_std[k])
    ax_l.legend(loc='lower center', ncol=len(mean), fancybox=True, bbox_to_anchor = (0.5,-0.3))
    ax_l.set_axis_off()

    ax[2].set_yticks([0, 5, 10])
    ax[2].set_yticklabels(['0', '5', '10'])
    ax[2].set_ylim(-0.5,10)
    ax[2].set_title('$Z_k - \hat{Z}_k$')
    ax[2].set_xticks([0, 5, 10])
    ax[2].set_xticklabels(['0 sec', 'time [sec]', '10 sec'])

    ax[1].set_yticks([0, 10, 20])
    ax[1].set_yticklabels(['0', '10', '20'])
    ax[1].set_ylim(-1.0,20)
    ax[1].set_title('$Y_k - \hat{Y}_k$')
    ax[1].set_xticks([0, 5, 10])
    ax[1].set_xticklabels(['0 sec', 'time [sec]', '10 sec'])

    ax[0].set_yticks([ 0, 30, 60])
    ax[0].set_yticklabels(['0', '30', '60'])
    ax[0].set_ylim(-3.0,60)
    ax[0].set_title('$X_k - \hat{X}_k$')
    ax[0].set_xticks([0, 5, 10])
    ax[0].set_xticklabels(['0 sec', 'time [sec]', '10 sec'])
    #fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(path)


def showsample(path, t_history, nominal, error, show=False): # model scheme with basic notation
    fig = plt.figure(figsize=(10, 7), dpi=200, facecolor='w', edgecolor='k')
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.2, 1, 1, 1]) 
    gs.update(left=0.07, bottom=0.05, right=0.99, top=0.99, wspace=0.1, hspace=0.1)   

    labels = ['$\mathbf{X}_k^{CMNF}$', '$\mathbf{X}_k^{PMKF}$', '$\mathbf{X}_k^{EKF}$']

    ax = [None] * 3
    for j in range(1,4):
        i = j - 1   
        ax[i] = plt.subplot(gs[j])
        ax[i].plot(t_history, nominal[:,i], color = 'black', linewidth = 4.0, alpha = 0.5)
        for k in range(0, len(colors)):
            ax[i].plot(t_history, nominal[:,i]+error[k][:,i], color = colors[k], linewidth = 2.0, alpha = 0.7)
        ax[i].set_xticks([])

    ax_l = plt.subplot(gs[0])
    for k in range(0, len(colors)):
        ax_l.plot([], [], color = colors[k], linewidth = 2.0, alpha = 0.7, label = labels[k])
    ax_l.legend(loc='lower center', ncol=3, fancybox=True, bbox_to_anchor = (0.5,-0.4))
    ax_l.set_axis_off()

    ax[2].set_yticks([-30, -20, -10, 0, 10])
    ax[2].set_yticklabels(['', '-20m', '-10m', '0m', '$\mathring{Z}_{k},\,Z_{k}$'])
    ax[2].set_ylim(-30,10)

    ax[2].set_xticks([0, 150, 300])
    ax[2].set_xticklabels(['0 min', 'time [mins]', '5 min'])

    ax[1].set_yticks([-40, -20, 0, 20, 40])
    ax[1].set_yticklabels(['', '-20m', '0m', '20m', '$\mathring{Y}_{k},\,Y_{k}$'])
    ax[1].set_ylim(-40,40)

    ax[0].set_yticks([-100, 0, 200, 400, 500])
    ax[0].set_yticklabels(['', '0m', '200m', '400m', '$\mathring{X}_{k},\,X_{k}$'])
    ax[0].set_ylim(-100,500)
    
    #ax[1].set_ylabel('Nominal path $\mathring{\mathbf{X}}_{k} [m]$')


    #fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(path)


def showsample_pw(path, t_history, nominal, error, show=False): # model scheme with basic notation
    fig = plt.figure(figsize=(10, 7), dpi=200, facecolor='w', edgecolor='k')
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.2, 1, 1, 1]) 
    gs.update(left=0.07, bottom=0.05, right=0.99, top=0.99, wspace=0.1, hspace=0.1)   

    labels = ['$\mathbf{X}_k^{CMNF}$', '$\mathbf{X}_k^{PMKF}$', '$\mathbf{X}_k^{EKF}$']

    t_history_pw = np.hstack((t_history[0], np.repeat(t_history[1:],2)))
    nominal_pw = np.vstack((np.repeat(nominal[0:-1], 2, axis = 0), nominal[-1].reshape(1,nominal.shape[1])))
    error_pw = [None] * len(error)
    for k in range(0, len(colors)):
        error_pw[k] = np.vstack((np.repeat(error[k][0:-1], 2, axis = 0), error[k][-1].reshape(1, error[k].shape[1])))
    ax = [None] * 3
    for j in range(1,4):
        i = j - 1   
        ax[i] = plt.subplot(gs[j])
        ax[i].plot(t_history, nominal[:,i], color = 'black', linewidth = 4.0, alpha = 0.5)
        for k in range(0, len(colors)):
            ax[i].plot(t_history_pw, nominal_pw[:,i]+error_pw[k][:,i], color = colors[k], linewidth = 2.0, alpha = 0.7)
        ax[i].set_xticks([])

    ax_l = plt.subplot(gs[0])
    for k in range(0, len(colors)):
        ax_l.plot([], [], color = colors[k], linewidth = 2.0, alpha = 0.7, label = labels[k])
    ax_l.legend(loc='lower center', ncol=3, fancybox=True, bbox_to_anchor = (0.5,-0.4))
    ax_l.set_axis_off()

    ax[2].set_yticks([-30, -20, -10, 0, 10])
    ax[2].set_yticklabels(['', '-20m', '-10m', '0m', '$\mathring{Z}_{k},\,Z_{k}$'])
    ax[2].set_ylim(-30,10)

    ax[2].set_xticks([0, 150, 300])
    ax[2].set_xticklabels(['0 min', 'time [mins]', '5 min'])

    ax[1].set_yticks([-40, -20, 0, 20, 40])
    ax[1].set_yticklabels(['', '-20m', '0m', '20m', '$\mathring{Y}_{k},\,Y_{k}$'])
    ax[1].set_ylim(-40,40)

    ax[0].set_yticks([-100, 0, 200, 400, 500])
    ax[0].set_yticklabels(['', '0m', '200m', '400m', '$\mathring{X}_{k},\,X_{k}$'])
    ax[0].set_ylim(-100,500)
    
    #ax[1].set_ylabel('Nominal path $\mathring{\mathbf{X}}_{k} [m]$')


    #fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(path)


def showsample3d(path, nominal, error, XB, show=False): # model scheme with basic notation
    XB = np.array(XB)
    fig = plt.figure(figsize=(10, 6), dpi=200)
    ax = fig.gca(projection='3d')
    ax.view_init(65, 57)
    ax.set_aspect('equal')

    # set good viewpoint
    #drawPoint(ax, [1.1, 1.1, 1.1], color = 'white', s = 0)

    bndX = [XB[:,0].min() - 150, XB[:,0].max() + 150]
    bndY = [XB[:,1].min() - 50, XB[:,1].max() + 50]
    bndZ = [-30, 5.0]
    dX = bndX[1] - bndX[0]
    dY = bndY[1] - bndY[0]
    dZ = bndZ[1] - bndZ[0]

    _ = ax.plot(nominal[:,0], nominal[:,1], nominal[:,2], color = 'black', linewidth = 4.0, alpha = 0.5)
    for k in range(0, len(colors)):
        _ = ax.plot(nominal[:error[k].shape[0],0]+error[k][:,0], nominal[:error[k].shape[0],1]+error[k][:,1], nominal[:error[k].shape[0],2]+error[k][:,2], color = colors[k], linewidth = 2.0, alpha = 0.7)

    #axes
    drawArrow(ax, [bndX[0]-dX/10.0, bndY[0]-dY/20.0, bndZ[0]-dZ/20.0], [bndX[1]+dX/20.0, bndY[0]-dY/20.0, bndZ[0]-dZ/20.0])
    drawArrow(ax, [bndX[0]-dX/20.0, bndY[0]-dY/15.0, bndZ[0]-dZ/20.0], [bndX[0]-dX/20.0, bndY[1]+dY/10.0, bndZ[0]-dZ/20.0])
    drawArrow(ax, [bndX[0]-dX/20.0, bndY[0]-dY/20.0, bndZ[0]-dZ/10.0], [bndX[0]-dX/20.0, bndY[0]-dY/20.0, bndZ[1]+dZ/10.0])
    textAtPoint(ax, [bndX[1]+dX/20.0, bndY[0] + dY/20.0 , bndZ[0]], '$x$')
    textAtPoint(ax, [bndX[0]+dX/100.0, bndY[1]+ dY/10.0, bndZ[0]], '$y$')
    textAtPoint(ax, [bndX[0]+dX/100.0, bndY[0], bndZ[1]+dZ/10.0], '$z$')

    #seabed
    X = np.arange(bndX[0], bndX[1], dX/100)
    Y = np.arange(bndY[0], bndY[1], dX/100)
    X, Y = np.meshgrid(X, Y)
    Zsb = np.random.normal(-30.0,0.2, X.shape)
    surf = ax.plot_surface(X, Y, Zsb, cmap=cm.copper, linewidth=0, antialiased=False, alpha=0.1)
        
    #surface
    Zsf = np.random.normal(0.0,0.1, X.shape)
    surf = ax.plot_surface(X, Y, Zsf, cmap=cm.Blues, linewidth=0, antialiased=False, alpha=0.1)



    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_axis_off()
    set_axes_aspect(ax, [1.0, 1.5, 15.0])

    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(path)

def shownominal2d(path, t_history, nominal, show=False): # model scheme with basic notation
    fig = plt.figure(figsize=(10, 6), dpi=200, facecolor='w', edgecolor='k')
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])     
    gs.update(left=0.07, bottom=0.05, right=0.99, top=0.98, wspace=0.1, hspace=0.1)   

    ax = [None] * 3
    for i in range(0,3):
        ax[i] = plt.subplot(gs[i])
        ax[i].plot(t_history, nominal[:,i], color = 'black', linewidth = 4.0, alpha = 0.5)
        ax[i].set_xticks([])

    ax[2].set_yticks([-13, -12, -10, -8, -7])
    ax[2].set_yticklabels(['', '-12m', '-10m', '-8m', '$\mathring{Z}_{k}$'])

    ax[2].set_xticks([0, 150, 300])
    ax[2].set_xticklabels(['0 min', 'time [mins]', '5 min'])

    ax[1].set_yticks([-20, -10, 0, 10, 20])
    ax[1].set_yticklabels(['', '-10m', '0m', '10m', '$\mathring{Y}_{k}$'])

    ax[0].set_yticks([-100, 0, 200, 400, 500])
    ax[0].set_yticklabels(['', '0m', '200m', '400m', '$\mathring{X}_{k}$'])
    
    #ax[1].set_ylabel('Nominal path $\mathring{\mathbf{X}}_{k} [m]$')



    #fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(path)



def shownominal3d(path, nominal, XB, show=False): # model scheme with basic notation
    XB = np.array(XB)
    fig = plt.figure(figsize=(10, 6), dpi=200)
    ax = fig.gca(projection='3d')
    ax.view_init(65, 57)
    ax.set_aspect('equal')

    # set good viewpoint
    #drawPoint(ax, [1.1, 1.1, 1.1], color = 'white', s = 0)

    bndX = [XB[:,0].min() - 150, XB[:,0].max() + 150]
    bndY = [XB[:,1].min() - 50, XB[:,1].max() + 50]
    bndZ = [-30, 5.0]
    dX = bndX[1] - bndX[0]
    dY = bndY[1] - bndY[0]
    dZ = bndZ[1] - bndZ[0]


    _ = ax.plot(nominal[:,0], nominal[:,1], nominal[:,2], color = 'black', linewidth = 4.0, alpha = 0.5)

    for i in range(0, XB.shape[0]):
        drawPoint(ax, XB[i], color = 'blue')

    #axes
    drawArrow(ax, [bndX[0]-dX/10.0, bndY[0]-dY/20.0, bndZ[0]-dZ/20.0], [bndX[1]+dX/20.0, bndY[0]-dY/20.0, bndZ[0]-dZ/20.0])
    drawArrow(ax, [bndX[0]-dX/20.0, bndY[0]-dY/15.0, bndZ[0]-dZ/20.0], [bndX[0]-dX/20.0, bndY[1]+dY/10.0, bndZ[0]-dZ/20.0])
    drawArrow(ax, [bndX[0]-dX/20.0, bndY[0]-dY/20.0, bndZ[0]-dZ/10.0], [bndX[0]-dX/20.0, bndY[0]-dY/20.0, bndZ[1]+dZ/10.0])
    textAtPoint(ax, [bndX[1]+dX/20.0, bndY[0] + dY/20.0 , bndZ[0]], '$x$')
    textAtPoint(ax, [bndX[0]+dX/100.0, bndY[1]+ dY/10.0, bndZ[0]], '$y$')
    textAtPoint(ax, [bndX[0]+dX/100.0, bndY[0], bndZ[1]+dZ/10.0], '$z$')

    #seabed
    X = np.arange(bndX[0], bndX[1], dX/100)
    Y = np.arange(bndY[0], bndY[1], dX/100)
    X, Y = np.meshgrid(X, Y)
    Zsb = np.random.normal(-30.0,0.2, X.shape)
    surf = ax.plot_surface(X, Y, Zsb, cmap=cm.copper, linewidth=0, antialiased=False, alpha=0.1)
        
    #surface
    Zsf = np.random.normal(0.0,0.1, X.shape)
    surf = ax.plot_surface(X, Y, Zsf, cmap=cm.Blues, linewidth=0, antialiased=False, alpha=0.1)



    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_axis_off()
    set_axes_aspect(ax, [1.0, 1.5, 15.0])

    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(path)


def showbearing(path, show=False): # model scheme with basic notation
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
    sZ = XB[2] + np.random.normal(0.0, 0.005, sX.shape)
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

        
def showsonarmodel(path, show=False): # model scheme with basic notation
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
    sZ = x[2] + np.random.normal(0.0, 0.005, sX.shape)
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

def showoptimalcontrol(path, show=False): # model scheme with basic notation
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


def showmodel(path, show=False): # model scheme with basic notation
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
    textAtPoint(ax, v, '${\mathbf{V}}_k$', [0.0, -0.5*s, 0.0])

    #angles
    drawArcScale(ax, Xk, XNk1, [XNk1[0], XNk1[1], Xk[2]], scale = 0.10)
    textAtPoint(ax, Xk, '$\gamma_k$', [-s, s, s])

    drawArcScale(ax, PZ(Xk), PZ(v), Y(Xk), scale = 0.30)
    drawArcScale(ax, PZ(Xk), PZ(v), Y(Xk), scale = 0.36)
    textAtPoint(ax, PZ(Xk), '$\\theta_k$', [-s, 1.5*s, 1.5*s])

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
        
 