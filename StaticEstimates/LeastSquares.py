from functools import partial

import numpy as np
from scipy.optimize import least_squares


class LeastSquares:
    def __init__(self, residuals, init):
        self.residuals = residuals
        self.init = init

    def fit(self, x, y):
        return

    def predict(self, x):
        return np.apply_along_axis(lambda y_: least_squares(partial(self.residuals, y_=y_), self.init).x, 1, x)
