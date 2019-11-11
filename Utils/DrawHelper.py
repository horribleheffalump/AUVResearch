import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def PX(A):
    return np.array([0.0, A[1], A[2]])
def PY(A):
    return np.array([A[0], 0.0, A[2]])
def PZ(A):
    return np.array([A[0], A[1], 0.0])

def X(A):
    return np.array([A[0], 0.0, 0.0])
def Y(A):
    return np.array([0.0, A[1], 0.0])
def Z(A):
    return np.array([0.0, 0.0, A[2]])


def drawPoint(ax, P, color = 'black', s=30, alpha = 1.0):
    P = np.array(P)
    return ax.scatter(P[0], P[1], P[2], color = color, s = s, alpha = alpha)

def drawLine(ax, A, B, color = 'black', linestyle='-', linewidth=1.0, alpha = 1.0):
    return ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], color = color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)

def drawArrow(ax, A,B, color = 'black', linewidth= 1.0):
    deltaX = Arrow3D([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], mutation_scale=10, arrowstyle="-|>", color=color, linewidth = linewidth)
    return ax.add_artist(deltaX)

def textAtPoint(ax, A, string, shift = [0.0,0.0,0.0]):
    return ax.text(A[0]+shift[0], A[1]+shift[1], A[2]+shift[2], string)

def cart2sphere(X: np.array):
    r = np.sqrt(X[0] * X[0] + X[1] * X[1] + X[2] * X[2])
    theta = np.arctan2(X[1], X[0]);
    gamma = np.arctan2(X[2], X[0] / np.cos(theta));
    return r, gamma, theta


def drawArcAngles(ax, C, P, theta: float, gamma: float, N=100):
    #ax - axes, C - center, P - point on the circle where to start the arc, theta, gamma - spheric angles (define length and direction of the arc), N - number of points on the arc
    tol = 1e-10     
    C = np.array(C)
    P = np.array(P)
    if np.abs(theta) > 0:
        th = np.arange(0,theta+tol*np.sign(theta), theta / N)
    else:
        th = np.zeros(N)
        
    if np.abs(gamma) > 0:
        gm = np.arange(0,gamma+tol*np.sign(gamma), gamma / N)
    else:
        gm = np.zeros(N)

        
    r, gm0, th0 = cart2sphere(P-C) 

    x = np.array(list(map(lambda i: C[0] + r * np.cos(gm[i]+gm0) * np.cos(th[i]+th0), np.arange(0, N, 1))))
    y = np.array(list(map(lambda i: C[1] + r * np.cos(gm[i]+gm0) * np.sin(th[i]+th0), np.arange(0, N, 1))))
    z = np.array(list(map(lambda i: C[2] + r * np.sin(gm[i]+gm0), np.arange(0, N, 1))))
    return ax.plot(x, y, z, color = 'black', linewidth = 0.5)


def drawArc(ax, C, P, D, N=20):
    #ax - axes, C - center, P - point on the circle where to start the arc, D - third point defines the direction where the arc ends, N - number of points on the arc
    C = np.array(C)
    P = np.array(P)
    D = np.array(D)
    _, gm0, th0 = cart2sphere(P-C) 
    _, gm1, th1 = cart2sphere(D-C) 
    return drawArcAngles(ax, C, P, th1-th0, gm1-gm0, N)


def drawArcScale(ax, C, P, D, scale=1.0, N=100):
    #ax - axes, C - center, P - point on the circle where to start the arc, D - third point defines the direction where the arc ends, N - number of points on the arc
    C = np.array(C)
    P = np.array(P)
    D = np.array(D)
    _, gm0, th0 = cart2sphere(P-C) 
    _, gm1, th1 = cart2sphere(D-C) 

    return drawArcAngles(ax, C, C + (P - C) * scale, th1-th0, gm1-gm0, N)



def set_axes_aspect(ax, aspect = [1.0,1.0,1.0]):
    #Adjust the aspect of the axes. Default [1,1,1] makes the equal scale
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius / aspect[0], x_middle + plot_radius / aspect[0]])
    ax.set_ylim3d([y_middle - plot_radius / aspect[1], y_middle + plot_radius / aspect[1]])
    ax.set_zlim3d([z_middle - plot_radius / aspect[2], z_middle + plot_radius / aspect[2]])

