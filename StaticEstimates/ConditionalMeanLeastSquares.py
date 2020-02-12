from functools import partial

import numpy as np
from scipy.optimize import least_squares

from StaticEstimates.ConditionalMean import ConditionalMean


class ConditionalMeanLeastSquares:
    def __init__(self, residuals, init):
        self.cm = ConditionalMean()
        self.residuals = residuals
        self.init = init

    def fit(self, x, y):
        additional_features = np.apply_along_axis(lambda y_: least_squares(partial(self.residuals, y_=y_), self.init).x, 1, x)
        x = np.hstack((x, additional_features))
        self.cm.fit(x, y)

    def predict(self, x):
        additional_features = np.apply_along_axis(lambda y_: least_squares(partial(self.residuals, y_=y_), self.init).x, 1, x)
        x = np.hstack((x, additional_features))
        return self.cm.predict(x)
