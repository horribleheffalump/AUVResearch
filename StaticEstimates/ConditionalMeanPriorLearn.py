import numpy as np
from sklearn.model_selection import train_test_split

from StaticEstimates.ConditionalMean import ConditionalMean


class ConditionalMeanPriorLearn:
    def __init__(self, pipeline, additional_train_size=0.0, already_fit=False):
        self.pipeline = pipeline
        self.already_fit = already_fit
        self.additional_train_size = additional_train_size
        self.cm = ConditionalMean()

    def fit(self, x, y):
        # train_test_split is used here to split the train test in two parts:
        # one is to train the model given by the pipeline
        # the other is to calculate the Conditional Mean estimate along with the additional features
        # provided by the pipeline
        #
        # if additional_train_size is set to zero, then the pipeline is trained
        # and the additional features are extracted from the same set
        if self.additional_train_size == 0.0:
            x_train = x
            x_add_train = x
            y_train = y
            y_add_train = y
        else:
            x_train, x_add_train, y_train, y_add_train = train_test_split(x, y, test_size=self.additional_train_size)
        if not self.already_fit:
            self.pipeline.fit(x_add_train, y_add_train)
        additional_features = self.pipeline.predict(x_train)
        x = np.hstack((x_train, additional_features))
        self.cm.fit(x, y_train)
        return

    def predict(self, x):
        additional_features = self.pipeline.predict(x)
        x = np.hstack((x, additional_features))
        return self.cm.predict(x)
