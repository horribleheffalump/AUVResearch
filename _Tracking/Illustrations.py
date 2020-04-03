import matplotlib.pyplot as plt
from matplotlib import gridspec

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import pandas as pd
#import datetime

from mpl_toolkits.mplot3d import Axes3D

from Utils.DrawHelper import *

from matplotlib import rc

from _Tracking.TrackingModel import *

rc('font',**{'family':'serif'})
rc('text', usetex=True)
rc('text.latex',unicode=True)
rc('text.latex',preamble=r'\usepackage[T2A]{fontenc}')
rc('text.latex',preamble=r'\usepackage[utf8]{inputenc}')
rc('text.latex',preamble=r'\usepackage[russian]{babel}')

from DynamicModel.io import load_path
import os.path

def showplane(path, show=False): # model scheme with basic notation
    cols = ['X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'x', 'y', 'z', 'v', 'phi', 'a', 'alpha', 'beta', 'RX', 'RY', 'RZ']
    dir = 'D:/pycharm.git/NavigationResearch/_Tracking/data/'
    file_name = os.path.join(dir, 'trajectory_00000.txt')
    data = load_path(file_name, cols)

    net_points = np.arange(0.0, 1.0, 0.1)
    fig = plt.figure(figsize=(12, 3), dpi=200)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,1])
    gs.update(left=0.0, bottom=0.00, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
    #ax = fig.gca(projection='3d')

    #ax1
    ax = plt.subplot(gs[0], projection='3d')
    ax.view_init(31, 26)
    # set good viewpoint
    drawPoint(ax, [1.1, 1.1, 1.1], color = 'white', s = 0)
    ax.dist = 8

    for x in net_points:
        drawLine(ax, np.array([x, 0, 0]), np.array([x, 1.0, 0]), linestyle=':', linewidth=0.2)
        drawLine(ax, np.array([0, x, 0]), np.array([1.0, x, 0]), linestyle=':', linewidth=0.2)

    x = (np.abs(data.x) / np.max(np.abs(data.x))).values
    y = (np.abs(data.y) / np.max(np.abs(data.y))).values
    z = np.zeros_like(data.x.values)
    ax.plot(x, y, z, color='red')
    p = np.array([tr([c[0], c[1], c[2]]) for c in zip(x, y, z)])

    #axes
    drawArrow(ax, [-0.01, 0.0, 0.0], [1.0, 0.0, 0.0])
    drawArrow(ax, [0.0, -0.01, 0.0], [0.0, 1.0, 0.0])
    drawArrow(ax, [0.0, 0.0, -0.01], [0.0, 0.0, 1.0])
    textAtPoint(ax, [1.0, -0.05, 0.1], '$x$')
    textAtPoint(ax, [0.05, 1.0, 0.1], '$y$')
    textAtPoint(ax, [0.05, 0.05, 1.0], '$z$')

   #set_axes_aspect(ax, [1.0, 1.0, 1.0])
   # ax.zaxis.set_major_locator(LinearLocator(10))
   # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_axis_off()

    #ax2
    ax = plt.subplot(gs[1], projection='3d')
    ax.view_init(31, 26)
    # set good viewpoint
    drawPoint(ax, [1.1, 1.1, 1.1], color = 'white', s = 0)
    ax.dist = 8

    for x_ in net_points:
        drawLine(ax, tr(np.array([x_, 0, 0])), tr(np.array([x_, 1.0, 0])), linestyle=':', linewidth=0.2)
        drawLine(ax, tr(np.array([0, x_, 0])), tr(np.array([1.0, x_, 0])), linestyle=':', linewidth=0.2)

    ax.plot(p[:,0], p[:,1], p[:,2], color='red')

    drawArcScale(ax, [0,0.001,0], tr([0,0,1]), [0,0,1], scale = 0.20)
    drawArcScale(ax, [0,0.001,0], [0,0,1], tr([0,0,1]), scale = 0.18)
    drawArcScale(ax, [0,0.001,0], [1,0,0], tr([1,0,0]), scale = 0.18)

    textAtPoint(ax, [0.3, -0.05, 0.02], '$\\alpha$')
    textAtPoint(ax, [0.05, 0.05, 0.25], '$\\beta$')

    #drawArc(ax, [0.0, 0.01, 0.0], tr([0.0,0.0,1.0]), [0.0,0.0,1.0])


    #axes
    drawArrow(ax, [-0.01, 0.0, 0.0], [1.0, 0.0, 0.0])
    drawArrow(ax, [0.0, -0.01, 0.0], [0.0, 1.0, 0.0])
    drawArrow(ax, [0.0, 0.0, -0.01], [0.0, 0.0, 1.0])
    textAtPoint(ax, [1.0, -0.05, 0.1], '$X$')
    textAtPoint(ax, [0.05, 1.0, 0.1], '$Y$')
    textAtPoint(ax, [0.05, 0.05, 1.0], '$Z$')

    #newaxes
    drawArrow(ax, [-0.01, 0.0, 0.0], tr([1.0, 0.0, 0.0]))
    drawArrow(ax, [0.0, -0.01, 0.0], tr([0.0, 1.0, 0.0]))
    drawArrow(ax, [0.0, 0.0, -0.01], tr([0.0, 0.0, 1.0]))
    textAtPoint(ax, tr([1.0, 0.0, 0.1]), '$x$')
    textAtPoint(ax, tr([0.0, 1.0, 0.1]), '$y$')
    textAtPoint(ax, tr([0.0, 0.0, 1.0]), '$z$')

    #set_axes_aspect(ax, [1.0, 1.0, 1.0])
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_axis_off()

    #fig.tight_layout()

    #ax3
    ax = plt.subplot(gs[2], projection='3d')
    ax.view_init(31, 26)
    # set good viewpoint
    drawPoint(ax, [2.1, 2.1, 2.1], color = 'white', s = 0)
    ax.dist = 7

    R0 = np.array([0.5,1, -1.1])
    textAtPoint(ax, 0.5*R0+[0.2,-0.2,0.1], '$R_0$')


    for x_ in net_points:
        drawLine(ax, tr(np.array([x_, 0, 0]), R0), tr(np.array([x_, 1.0, 0]), R0), linestyle=':', linewidth=0.2)
        drawLine(ax, tr(np.array([0, x_, 0]), R0), tr(np.array([1.0, x_, 0]), R0), linestyle=':', linewidth=0.2)

    p = np.array([tr([c[0], c[1], c[2]], R0) for c in zip(x, y, z)])
    ax.plot(p[:,0], p[:,1], p[:,2], color='red')

    #drawArc(ax, [0.0, 0.01, 0.0], tr([0.0,0.0,1.0]), [0.0,0.0,1.0])


    #axes
    drawArrow(ax, [-0.03, 0.0, 0.0], [1.3, 0.0, 0.0])
    drawArrow(ax, [0.0, -0.03, 0.0], [0.0, 1.3, 0.0])
    drawArrow(ax, [0.0, 0.0, -0.03], [0.0, 0.0, 1.3])
    textAtPoint(ax, [1.3, -0.05, 0.1], '$X$')
    textAtPoint(ax, [0.05, 1.3, 0.1], '$Y$')
    textAtPoint(ax, [0.05, 0.05, 1.3], '$Z$')

    #newaxes
    drawArrow(ax, [-0.01, 0.0, 0.0]+R0, tr([1.0, 0.0, 0.0], R0))
    drawArrow(ax, [0.0, -0.01, 0.0]+R0, tr([0.0, 1.0, 0.0], R0))
    drawArrow(ax, [0.0, 0.0, -0.01]+R0, tr([0.0, 0.0, 1.0], R0))
    textAtPoint(ax, tr([1.0, -0.1, 0.1], R0), '$x$')
    textAtPoint(ax, tr([0.0, 1.0, 0.1], R0), '$y$')
    textAtPoint(ax, tr([0.0, 0.0, 1.0], R0), '$z$')

    drawLine(ax, [0,0,0], R0, linestyle=':')
    #set_axes_aspect(ax, [1.0, 1.0, 1.0])
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_axis_off()

    sX = np.arange(-0.1, 1.3, 1.0 / 100.0)
    sY = np.arange(-0.1, 1.3, 1.0 / 100.0)
    sX, sY = np.meshgrid(sX, sY)
    sZ = np.random.normal(0.0, 0.005, sX.shape)
    # Plot the surface.
    surf = ax.plot_surface(sX, sY, sZ, cmap=cm.Blues, linewidth=0, antialiased=False, alpha=0.1)
    ax.text(-0.0, 0.3, 0.0, 'sea surface', (0,1,0.1), fontsize=6)

    if show:
        plt.show()
    else:
        plt.savefig(path)


def showpaths(path2d, path3d, show=False): # model random paths in 2D and 3D
    M = 100
    x, _ = generate_sample_paths(M, int(T/delta))

    fig = plt.figure(figsize=(5, 4), dpi=200)
    ax = fig.gca()
    for i in range(0, M):
        ax.plot(x[i,:,6], x[i,:,7], color="red", alpha = 0.3)
    ax.set_xticks([-500, 0, 500, 800])
    ax.set_xticklabels(['-500m', '0m', '500m', '$x_t$'])
    ax.set_yticks([0, -500, -1000, -1300])
    ax.set_yticklabels(['0m', '-500m', '-1000m', '$y_t$'])
    if show:
        plt.show()
    else:
        plt.savefig(path2d)

    fig = plt.figure(figsize=(5, 4), dpi=200)
    ax = fig.gca(projection='3d')
    ax.grid(False)
    for i in range(0,M):
        ax.plot(x[i,:,0], x[i,:,1], x[i,:,2], color="red", alpha = 0.3)
    #ax.scatter(Xb[:,0],Xb[:,1],Xb[:,2])
    ax.set_xticks([-3000, -2000, 0, 2000])
    ax.set_xticklabels(['$X_t$', '-2km', '0km', '2km', ])
    ax.set_yticks([18000, 20000, 22000, 23000])
    ax.set_yticklabels(['18km', '20km', '22km', '$Y_t$'])
    ax.set_zticks([-1200, -1000, -800, -700])
    ax.set_zticklabels(['-1200m', '-1000m', '-800m', '$Z_t$'])

    if show:
        plt.show()
    else:
        plt.savefig(path3d)


def showstaticstd(path, show=False): # model scheme with basic notation
    cols = ['X', 'Y', 'Z']
    dir = 'D:/pycharm.git/NavigationResearch/_Tracking/data/'
    file_name = os.path.join(dir, 'static_data_nolabels.txt')
    data = load_path(file_name, cols)

    range = [20, 10, 5]

    X_Linear = [data['X'][0], data['X'][5], data['X'][10]]
    X_LS = [data['X'][1], data['X'][6], data['X'][11]]
    X_ML = [data['X'][2], data['X'][7], data['X'][12]]
    X_CM_LS = [data['X'][3], data['X'][8], data['X'][13]]
    X_CM_ML = [data['X'][4], data['X'][9], data['X'][14]]

    Y_Linear = [data['Y'][0], data['Y'][5], data['Y'][10]]
    Y_LS = [data['Y'][1], data['Y'][6], data['Y'][11]]
    Y_ML = [data['Y'][2], data['Y'][7], data['Y'][12]]
    Y_CM_LS = [data['Y'][3], data['Y'][8], data['Y'][13]]
    Y_CM_ML = [data['Y'][4], data['Y'][9], data['Y'][14]]

    Z_Linear = [data['Z'][0], data['Z'][5], data['Z'][10]]
    Z_LS = [data['Z'][1], data['Z'][6], data['Z'][11]]
    Z_ML = [data['Z'][2], data['Z'][7], data['Z'][12]]
    Z_CM_LS = [data['Z'][3], data['Z'][8], data['Z'][13]]
    Z_CM_ML = [data['Z'][4], data['Z'][9], data['Z'][14]]

    fig = plt.figure(figsize=(8, 4), dpi=200, facecolor='w', edgecolor='k')
    gs = gridspec.GridSpec(2, 3, height_ratios=[0.03, 1], width_ratios=[1,1,1])
    gs.update(left=0.05, bottom=0.07, right=0.98, top=0.99, wspace=0.15, hspace=0.16)
    #ax = fig.gca(projection='3d')

    #ax1
    ax = plt.subplot(gs[1,0])

    ax.plot(range, X_Linear, color='green', linewidth=0.5)
    ax.plot(range, X_LS, color='red', linestyle=':', linewidth=0.5)
    ax.plot(range, X_ML, color='blue', linestyle=':', linewidth=0.5)
    ax.plot(range, X_CM_LS, color='red', linewidth=1.0)
    ax.plot(range, X_CM_ML, color='blue', linewidth=0.5)

    ax.set_yticks([100, 140, 180, 185])
    ax.set_yticklabels(['100', '140', '180', '$\sigma_{X}$'])
    ax.set_ylim(95,185)
    #ax.set_title('')
    ax.set_xticks([5, 12.5, 20])
    ax.set_xticklabels(['5km', 'Range $R$', '20km'])

    #ax2
    ax = plt.subplot(gs[1,1])

    ax.plot(range, Y_Linear, color='green', linewidth=0.5)
    ax.plot(range, Y_LS, color='red', linestyle=':', linewidth=0.5)
    ax.plot(range, Y_ML, color='blue', linestyle=':', linewidth=0.5)
    ax.plot(range, Y_CM_LS, color='red', linewidth=1.0)
    ax.plot(range, Y_CM_ML, color='blue', linewidth=0.5)

    ax.set_yticks([100, 250, 400, 420])
    ax.set_yticklabels(['100', '250', '400', '$\sigma_{Y}$'])
    ax.set_ylim(90,420)
    ax.set_title('')
    ax.set_xticks([5, 12.5, 20])
    ax.set_xticklabels(['5km', 'Range $R$', '20km'])

    #ax3
    ax = plt.subplot(gs[1,2])

    ax.plot(range, Z_Linear, color='green', linewidth=0.5)
    ax.plot(range, Z_LS, color='red', linestyle=':', linewidth=0.5)
    ax.plot(range, Z_ML, color='blue', linestyle=':', linewidth=0.5)
    ax.plot(range, Z_CM_LS, color='red', linewidth=1.0)
    ax.plot(range, Z_CM_ML, color='blue', linewidth=0.5)

    ax.set_yticks([40, 80, 120, 125])
    ax.set_yticklabels(['40', '80', '120', '$\sigma_{Z}$'])
    ax.set_ylim(35,125)
    #ax.set_title('')
    ax.set_xticks([5, 12.5, 20])
    ax.set_xticklabels(['5km', 'Range $R$', '20km'])


    ax = plt.subplot(gs[0,1])

    ax.plot([], [], color='green', linewidth=0.5, label='$\sigma^{Linear}$')
    ax.plot([], [], color='red', linestyle=':', linewidth=0.5, label='$\sigma^{LS}$')
    ax.plot([], [], color='blue', linestyle=':', linewidth=0.5, label='$\sigma^{ML}$')
    ax.plot([], [], color='red', linewidth=0.5, label='$\sigma^{CM\,LS}$')
    ax.plot([], [], color='blue', linewidth=0.5, label='$\sigma^{CM\,ML}$')
    ax.set_axis_off()
    ax.legend(loc='lower center', ncol=5, fancybox=True, bbox_to_anchor = (0.5,-2.5))

    if show:
        plt.show()
    else:
        plt.savefig(path)

def showsdynamicstd_xyz(path, show=False): # model scheme with basic notation
    cols = ['X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'x', 'y', 'z', 'v', 'phi', 'a', 'alpha', 'beta', 'RX', 'RY', 'RZ']
    dir = 'D:/pycharm.git/NavigationResearch/_Tracking/data/dynamic_data/estimates'
    data = loaddata(dir, cols)
    print(data.columns)

    t = np.arange(0, T+delta)

    fig = plt.figure(figsize=(8, 4), dpi=200, facecolor='w', edgecolor='k')
    gs = gridspec.GridSpec(2, 3, height_ratios=[0.03, 1], width_ratios=[1,1,1])
    gs.update(left=0.05, bottom=0.07, right=0.98, top=0.99, wspace=0.15, hspace=0.16)

    #ax1
    ax = plt.subplot(gs[1,0])
    ax.plot(t, data.X_cmnf_ls_nodoppler_theor, color='red', linewidth=1.0)
    ax.plot(t, data.X_cmnf_ls_theor, color='blue', linewidth=1.0)

    ax.set_yticks([0, 50, 100, 110])
    ax.set_yticklabels(['0', '50', '100', '$\sigma_{X}$'])
    ax.set_ylim(0,110)
    #ax.set_title('')
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(['0sec', 'Time $t$', '100sec'])

    #ax1
    ax = plt.subplot(gs[1,1])
    ax.plot(t, data.Y_cmnf_ls_nodoppler_theor, color='red', linewidth=1.0)
    ax.plot(t, data.Y_cmnf_ls_theor, color='blue', linewidth=1.0)

    ax.set_yticks([0, 100, 200, 220])
    ax.set_yticklabels(['0', '100', '200', '$\sigma_{Y}$'])
    ax.set_ylim(0,220)
    #ax.set_title('')
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(['0sec', 'Time $t$', '100sec'])

    #ax1
    ax = plt.subplot(gs[1,2])
    ax.plot(t, data.Z_cmnf_ls_nodoppler_theor, color='red', linewidth=1.0)
    ax.plot(t, data.Z_cmnf_ls_theor, color='blue', linewidth=1.0)

    ax.set_yticks([20, 40, 60, 65])
    ax.set_yticklabels(['20', '40', '60', '$\sigma_{Z}$'])
    ax.set_ylim(15,65)
    #ax.set_title('')
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(['0sec', 'Time $t$', '100sec'])

    ax = plt.subplot(gs[0,1])

    ax.plot([], [], color='blue', linewidth=0.5, label='$\sigma_{Doppler}$')
    ax.plot([], [], color='red', linewidth=0.5, label='$\sigma_{NoDoppler}$')
    ax.set_axis_off()
    ax.legend(loc='lower center', ncol=5, fancybox=True, bbox_to_anchor = (0.5,-2.5))

    if show:
        plt.show()
    else:
        plt.savefig(path)


def showsdynamicstd_Vxyz(path, show=False): # model scheme with basic notation
    cols = ['X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'x', 'y', 'z', 'v', 'phi', 'a', 'alpha', 'beta', 'RX', 'RY', 'RZ']
    dir = 'D:/pycharm.git/NavigationResearch/_Tracking/data/dynamic_data'
    data = loaddata(dir, cols)
    print(data.columns)

    t = np.arange(0, T+delta)

    fig = plt.figure(figsize=(8, 4), dpi=200, facecolor='w', edgecolor='k')
    gs = gridspec.GridSpec(2, 3, height_ratios=[0.03, 1], width_ratios=[1,1,1])
    gs.update(left=0.05, bottom=0.07, right=0.98, top=0.99, wspace=0.15, hspace=0.16)

    #ax1
    ax = plt.subplot(gs[1,0])
    ax.plot(t, data.VX_cmnf_ls_nodoppler_theor, color='red', linewidth=1.0)
    ax.plot(t, data.VX_cmnf_ls_theor, color='blue', linewidth=1.0)

    ax.set_yticks([0, 3, 3.3])
    ax.set_yticklabels(['0', '3', '$\sigma_{V^X}$'])
    ax.set_ylim(0,3.3)
    #ax.set_title('')
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(['0sec', 'Time $t$', '100sec'])

    #ax1
    ax = plt.subplot(gs[1,1])
    ax.plot(t, data.VY_cmnf_ls_nodoppler_theor, color='red', linewidth=1.0)
    ax.plot(t, data.VY_cmnf_ls_theor, color='blue', linewidth=1.0)

    ax.set_yticks([0, 3, 3.3])
    ax.set_yticklabels(['0', '3', '$\sigma_{V^Y}$'])
    ax.set_ylim(0,3.3)
    #ax.set_title('')
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(['0sec', 'Time $t$', '100sec'])

    #ax1
    ax = plt.subplot(gs[1,2])
    ax.plot(t, data.VZ_cmnf_ls_nodoppler_theor, color='red', linewidth=1.0)
    ax.plot(t, data.VZ_cmnf_ls_theor, color='blue', linewidth=1.0)

    ax.set_yticks([0, 1, 1.1])
    ax.set_yticklabels(['0', '1', '$\sigma_{V^Z}$'])
    ax.set_ylim(0,1.1)
    #ax.set_title('')
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(['0sec', 'Time $t$', '100sec'])

    ax = plt.subplot(gs[0,1])

    ax.plot([], [], color='blue', linewidth=0.5, label='$\sigma_{Doppler}$')
    ax.plot([], [], color='red', linewidth=0.5, label='$\sigma_{NoDoppler}$')
    ax.set_axis_off()
    ax.legend(loc='lower center', ncol=5, fancybox=True, bbox_to_anchor = (0.5,-2.5))

    if show:
        plt.show()
    else:
        plt.savefig(path)

def showsdynamicstd_va(path, show=False): # model scheme with basic notation
    cols = ['X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'x', 'y', 'z', 'v', 'phi', 'a', 'alpha', 'beta', 'RX', 'RY', 'RZ']
    dir = 'D:/pycharm.git/NavigationResearch/_Tracking/data/dynamic_data'
    data = loaddata(dir, cols)
    print(data.columns)

    t = np.arange(0, T+delta)

    fig = plt.figure(figsize=(6, 4), dpi=200, facecolor='w', edgecolor='k')
    gs = gridspec.GridSpec(2, 2, height_ratios=[0.03, 1], width_ratios=[1,1])
    gs.update(left=0.05, bottom=0.07, right=0.98, top=0.99, wspace=0.15, hspace=0.16)

    #ax1
    ax = plt.subplot(gs[1,0])
    ax.plot(t, data.v_cmnf_ls_nodoppler_theor, color='red', linewidth=1.0)
    ax.plot(t, data.v_cmnf_ls_theor, color='blue', linewidth=1.0)
    ax.plot(t, data.v, color='green', linewidth=1.0)

    ax.set_yticks([0, 2, 2.1])
    ax.set_yticklabels(['0', '2', '$\sigma_{v}$'])
    ax.set_ylim(0,2.1)
    #ax.set_title('')
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(['0sec', 'Time $t$', '100sec'])

    #ax1
    ax = plt.subplot(gs[1,1])
    ax.plot(t, data.a_cmnf_ls_nodoppler_theor, color='red', linewidth=1.0)
    ax.plot(t, data.a_cmnf_ls_theor, color='blue', linewidth=1.0)
    ax.plot(t, data.a, color='green', linewidth=1.0)

    ax.set_yticks([0, 0.12, 0.13])
    ax.set_yticklabels(['0', '0.12', '$\sigma_{a}$'])
    ax.set_ylim(0,0.13)
    #ax.set_title('')
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(['0sec', 'Time $t$', '100sec'])

    ax = plt.subplot(gs[0,1])

    ax.plot([], [], color='blue', linewidth=0.5, label='$\sigma_{Doppler}$')
    ax.plot([], [], color='red', linewidth=0.5, label='$\sigma_{NoDoppler}$')
    ax.plot([], [], color='green', linewidth=0.5, label='$\sigma_{prior}$')
    ax.set_axis_off()
    ax.legend(loc='lower left', ncol=5, fancybox=True, bbox_to_anchor = (-0.7,-2.5))

    if show:
        plt.show()
    else:
        plt.savefig(path)

def showsdynamicstd_R(path, show=False): # model scheme with basic notation
    cols = ['X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'x', 'y', 'z', 'v', 'phi', 'a', 'alpha', 'beta', 'RX', 'RY', 'RZ']
    dir = 'D:/pycharm.git/NavigationResearch/_Tracking/data/dynamic_data'
    data = loaddata(dir, cols)
    print(data.columns)

    t = np.arange(0, T+delta)

    fig = plt.figure(figsize=(8, 4), dpi=200, facecolor='w', edgecolor='k')
    gs = gridspec.GridSpec(2, 3, height_ratios=[0.03, 1], width_ratios=[1,1,1])
    gs.update(left=0.05, bottom=0.07, right=0.98, top=0.99, wspace=0.15, hspace=0.16)

    #ax1
    ax = plt.subplot(gs[1,0])
    ax.plot(t, data.RX_cmnf_ls_nodoppler_theor, color='red', linewidth=1.0)
    ax.plot(t, data.RX_cmnf_ls_theor, color='blue', linewidth=1.0)

    ax.set_yticks([0, 50, 100, 110])
    ax.set_yticklabels(['0', '50', '100', '$\sigma_{R_0^X}$'])
    ax.set_ylim(0,110)
    #ax.set_title('')
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(['0sec', 'Time $t$', '100sec'])

    #ax1
    ax = plt.subplot(gs[1,1])
    ax.plot(t, data.RY_cmnf_ls_nodoppler_theor, color='red', linewidth=1.0)
    ax.plot(t, data.RY_cmnf_ls_theor, color='blue', linewidth=1.0)

    ax.set_yticks([0, 100, 200, 220])
    ax.set_yticklabels(['0', '100', '200', '$\sigma_{R_0^Y}$'])
    ax.set_ylim(0,220)
    #ax.set_title('')
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(['0sec', 'Time $t$', '100sec'])

    #ax1
    ax = plt.subplot(gs[1,2])
    ax.plot(t, data.RZ_cmnf_ls_nodoppler_theor, color='red', linewidth=1.0)
    ax.plot(t, data.RZ_cmnf_ls_theor, color='blue', linewidth=1.0)

    ax.set_yticks([20, 40, 60, 65])
    ax.set_yticklabels(['20', '40', '60', '$\sigma_{R_0^Z}$'])
    ax.set_ylim(20,65)
    #ax.set_title('')
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(['0sec', 'Time $t$', '100sec'])

    ax = plt.subplot(gs[0,1])

    ax.plot([], [], color='blue', linewidth=0.5, label='$\sigma_{Doppler}$')
    ax.plot([], [], color='red', linewidth=0.5, label='$\sigma_{NoDoppler}$')
    ax.set_axis_off()
    ax.legend(loc='lower center', ncol=5, fancybox=True, bbox_to_anchor = (0.5,-2.5))

    if show:
        plt.show()
    else:
        plt.savefig(path)

def showsdynamicstd_angles(path, show=False): # model scheme with basic notation
    cols = ['X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'x', 'y', 'z', 'v', 'phi', 'a', 'alpha', 'beta', 'RX', 'RY', 'RZ']
    dir = 'D:/pycharm.git/NavigationResearch/_Tracking/data/dynamic_data'
    data = loaddata(dir, cols)
    print(data.columns)

    t = np.arange(0, T+delta)

    fig = plt.figure(figsize=(8, 4), dpi=200, facecolor='w', edgecolor='k')
    gs = gridspec.GridSpec(2, 3, height_ratios=[0.03, 1], width_ratios=[1,1,1])
    gs.update(left=0.05, bottom=0.07, right=0.98, top=0.99, wspace=0.16, hspace=0.16)

    #ax1
    ax = plt.subplot(gs[1,0])
    ax.plot(t, data.phi_cmnf_ls_nodoppler_theor, color='red', linewidth=1.0)
    ax.plot(t, data.phi_cmnf_ls_theor, color='blue', linewidth=1.0)
    ax.plot(t, data.phi, color='green', linewidth=1.0)

    ax.set_yticks([0, 1.2, 1.3])
    ax.set_yticklabels(['0', '1.2', '$\sigma_{\\varphi}$'])
    ax.set_ylim(0,1.3)
    #ax.set_title('')
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(['0sec', 'Time $t$', '100sec'])

    #ax1
    ax = plt.subplot(gs[1,1])
    ax.plot(t, data.alpha_cmnf_ls_nodoppler_theor, color='red', linewidth=1.0)
    ax.plot(t, data.alpha_cmnf_ls_theor, color='blue', linewidth=1.0)
    ax.plot(t, data.alpha, color='green', linewidth=1.0)

    ax.set_yticks([0.05, 0.1, 0.11])
    ax.set_yticklabels(['0.05', '0.1', '$\sigma_{\\alpha}$'])
    ax.set_ylim(0.05,0.11)
    #ax.set_title('')
    ax.set_xticks([0.05, 50, 100])
    ax.set_xticklabels(['0sec', 'Time $t$', '100sec'])

    #ax1
    ax = plt.subplot(gs[1,2])
    ax.plot(t, data.beta_cmnf_ls_nodoppler_theor, color='red', linewidth=1.0)
    ax.plot(t, data.beta_cmnf_ls_theor, color='blue', linewidth=1.0)
    ax.plot(t, data.beta, color='green', linewidth=1.0)

    ax.set_yticks([0.05, 0.1, 0.11])
    ax.set_yticklabels(['0.05', '0.1', '$\sigma_{\\beta}$'])
    ax.set_ylim(0.05, 0.11)
    # ax.set_title('')
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(['0sec', 'Time $t$', '100sec'])

    ax = plt.subplot(gs[0,1])

    ax.plot([], [], color='blue', linewidth=0.5, label='$\sigma_{Doppler}$')
    ax.plot([], [], color='red', linewidth=0.5, label='$\sigma_{NoDoppler}$')
    ax.plot([], [], color='green', linewidth=0.5, label='$\sigma_{prior}$')

    ax.set_axis_off()
    ax.legend(loc='lower center', ncol=5, fancybox=True, bbox_to_anchor = (0.5,-2.5))

    if show:
        plt.show()
    else:
        plt.savefig(path)


filters = []
filters_theor = ['cmnf_ls', 'cmnf_ls_nodoppler']
estimates = ['ls']




def loaddata(dir, cols):

    file_template = os.path.join(dir, 'estimates/<filter>/estimate_error_<filter>_std.txt')
    theor_file_template = os.path.join(dir, 'estimates/<filter>/KHat.npy')

    dfs = []
    for f in filters:
        file_name = file_template.replace('<filter>', f)
        data = load_path(file_name, [f'{c}_{f}' for c in cols])
        dfs.append(data)

    for f in filters_theor:
        file_name = theor_file_template.replace('<filter>', f)
        KHat = np.load(file_name)
        theor_data = pd.DataFrame(np.array([np.sqrt(KHat[:,i,i]) for i in range(0, len(cols))]).T, columns=[f'{c}_{f}_theor' for c in cols])
        dfs.append(theor_data)

    for e in estimates:
        file_name = file_template.replace('<filter>', e)
        data = load_path(file_name, [f'{c}_{e}' for c in cols[:3]])
        dfs.append(data)

    file_name = os.path.join(dir, 'trajectories/trajectory_std.txt')
    data = load_path(file_name, cols)
    dfs.append(data)

    alldata = pd.concat(dfs, axis=1, sort=False)
    return alldata

def all_cols(col):
    f_cols_theor = [f'{col}_{f}_theor' for f in filters_theor]
    f_cols = [f'{col}_{f}' for f in filters]
    e_cols = [f'{col}_{e}' for e in estimates]
    return f_cols + f_cols_theor + e_cols + [col]

def f_cols(col):
    f_cols = [f'{col}_{f}' for f in filters]
    f_cols_theor = [f'{col}_{f}_theor' for f in filters_theor]
    return f_cols + f_cols_theor + [col]

def all_cols_by_estimate(est, cols):
    _cols = [f'{col}_{est}' for col in cols]
    return _cols + cols


def tr(X, R0=[0,0,0]):
    alpha = np.pi/12
    beta = np.pi/12
    A = np.array([[np.cos(alpha), -np.sin(alpha) * np.cos(beta), np.sin(alpha) * np.sin(beta)],
                  [np.sin(alpha), np.cos(alpha) * np.cos(beta), -np.cos(alpha)*np.sin(beta)],
                  [0, np.sin(beta), np.cos(beta)]])
    return A @ X + R0