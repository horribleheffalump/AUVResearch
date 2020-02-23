from StaticEstimates.ConditionalMean import ConditionalMean
from StaticEstimates.PriorLearn import PriorLearn
from _Tracking.TrackingModel import *

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.pipeline import Pipeline

from joblib import dump, load
# from sklearn.model_selection import train_test_split

Mtrain = 10000  # number of sample paths for CMNF parameters estimation (train set)
ml_train_part = 2000/(Mtrain)

pipe_file_name = "Z:/Наука - Data/2019 - Sensors - Tracking/data/pipe.joblib"
do_save = True
do_load = False


def extract_features_and_variables(States, Observations):
    X = Observations[:, :, :2 * Xb.shape[0]].reshape(Observations.shape[0] * Observations.shape[1], -1)
    Y = States[:, :, :3].reshape(States.shape[0] * States.shape[1], -1)
    return X, Y


states, observations = generate_sample_paths(Mtrain, 0)
x_train, y_train = extract_features_and_variables(states, observations)


if do_load:
    # load previous
    pipe_lasso = load(pipe_file_name)
else:
    pipe_lasso = Pipeline(steps=[
        ('polynomial', PolynomialFeatures(degree=2, interaction_only=False)),
        ('lasso', MultiTaskLassoCV(eps=0.001, n_alphas=20, normalize=True, cv=5, n_jobs=-1, max_iter=10000))
        ])

estimator_ml = PriorLearn(pipe_lasso, train_size=ml_train_part, already_fit=do_load)

estimator_ml.fit(x_train, y_train)

if do_save:
    dump(pipe_lasso, pipe_file_name)

