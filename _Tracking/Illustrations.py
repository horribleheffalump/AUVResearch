import matplotlib.pyplot as plt
from matplotlib import gridspec

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
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




def tr(X, R0=[0,0,0]):
    alpha = np.pi/12
    beta = np.pi/12
    A = np.array([[np.cos(alpha), -np.sin(alpha) * np.cos(beta), np.sin(alpha) * np.sin(beta)],
                  [np.sin(alpha), np.cos(alpha) * np.cos(beta), -np.cos(alpha)*np.sin(beta)],
                  [0, np.sin(beta), np.cos(beta)]])
    return A @ X + R0