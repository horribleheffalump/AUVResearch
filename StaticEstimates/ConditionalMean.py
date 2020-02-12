import numpy as np


class ConditionalMean:
    def __init__(self):
        self.P = []
        self.m_x = []
        self.m_y = []

    def fit(self, x, y):
        n = x.shape[0]
        self.m_x = x.mean(axis=0)[np.newaxis, :]
        self.m_y = y.mean(axis=0)[np.newaxis, :]
        cx = x - self.m_x
        cy = y - self.m_y
        cov_yx = (cy.T @ cx) / (n - 1.)
        cov_xx = np.cov(x, rowvar=False)
        self.P = cov_yx @ np.linalg.pinv(cov_xx)

    def predict(self, x):
        return (self.P @ (x - self.m_x).T).T + self.m_y
