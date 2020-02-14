import numpy as np
from numba import jit

class SimpleNonlinearModel:
    """
    General nonlinear stochastic discrete time model.
    .. math::
        x(t) = Phi(x(t-1)) + W(t), W ~ N(0,diag(self.std_w))
        y(t) = Psi(x(t)) + Nu(t), Nu ~ N(0,diag(self.std_nu))
    W, Nu - Gaussian white noise with independent components
    :param phi: state dynamics transformation
    :param psi: state to observations transformation
    :param std_w: std of the Gaussian noise in state equation
    :param std_nu: std of the Gaussian noise in observations
    :param x0: initial condition generator x0=x0()
    """

    def __init__(self, phi, psi, std_w, std_nu, x0, delta):
        self.phi = phi  # Nonlinear function of the state dynamics. Phi = Phi(t-1, x)
        self.psi = psi  # Nonlinear function of the observations. Phi = Psi(t, x)
        self.std_w = std_w  # Standard deviation of the Gaussian noise in state equation
        self.m_w = np.zeros_like(std_w)
        self.std_nu = std_nu  # Standard deviation of the Gaussian noise in the observations
        self.m_nu = np.zeros_like(std_nu)
        self.x0 = x0  # Initial condition
        self.delta = delta  # discretization step size for SDE approximate
        # numerical solution with Euler–Maruyama method

    def generate_sample_path(self, n):
        """
        Generates a sample path
        :param n: path length
        :returns x, y: state and observations sample path
        """
        x = np.zeros((n + 1, self.m_w.shape[0]))
        y = np.zeros((n + 1, self.m_nu.shape[0]))
        # w = np.random.normal(self.m_w, self.std_w, (n + 1, self.m_w.shape[0]))
        # nu = np.random.normal(self.m_nu, self.std_nu, (n + 1, self.m_nu.shape[0]))
        x[0, :] = self.x0()
        y[0, :] = self.psi(x[0, :]) + np.random.normal(self.m_nu, self.std_nu)
        for i in range(0, n):
            x[i + 1, :] = self.phi(x[i, :]) + np.random.normal(self.m_w, self.std_w)
            y[i + 1, :] = self.psi(x[i + 1, :]) + np.random.normal(self.m_nu, self.std_nu)
        return x, y

    def generate_sample_paths(self, m, n):
        """
        Generates a sample path
        :param n: path length
        :returns x, y: state and observations sample path
        """
        x = np.zeros((m, n + 1, self.m_w.shape[0]))
        y = np.zeros((m, n + 1, self.m_nu.shape[0]))
        # w = np.random.normal(self.m_w, self.std_w, (n + 1, self.m_w.shape[0]))
        # nu = np.random.normal(self.m_nu, self.std_nu, (n + 1, self.m_nu.shape[0]))
        for j in range(0, m):
            x[j, 0, :] = self.x0()
            y[j, 0, :] = self.psi(x[j, 0, :]) + np.random.normal(self.m_nu, self.std_nu)
            for i in range(0, n):
                x[j, i + 1, :] = self.phi(x[j, i, :]) + np.random.normal(self.m_w, self.std_w)
                y[j, i + 1, :] = self.psi(x[j, i + 1, :]) + np.random.normal(self.m_nu, self.std_nu)
        return x, y

