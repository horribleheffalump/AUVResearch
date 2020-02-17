import numpy as np
from sklearn.model_selection import train_test_split


class PriorLearn:
    def __init__(self, pipeline, train_size=1.0, already_fit=False):
        self.pipeline = pipeline
        self.already_fit = already_fit
        self.train_size = train_size

    def fit(self, x, y):
        # if the train set is very big and we need only a part of it,
        # we usr the train_test_split to get rid of extra data
        #
        # if train_size == 1, use all the data to fit the pipeline
        if self.train_size == 1.0:
            x_train = x
            y_train = y
        else:
            _, x_train,  _, y_train = train_test_split(x, y, test_size=self.train_size)
        if not self.already_fit:
            self.pipeline.fit(x_train, y_train)
        return

    def predict(self, x):
        return self.pipeline.predict(x)
