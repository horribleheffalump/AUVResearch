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
        
 