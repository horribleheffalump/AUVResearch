from StaticEstimates.ConditionalMean import ConditionalMean
from StaticEstimates.ConditionalMeanLeastSquares import ConditionalMeanLeastSquares
from StaticEstimates.ConditionalMeanPriorLearn import ConditionalMeanPriorLearn
from StaticEstimates.LeastSquares import LeastSquares
from StaticEstimates.PriorLearn import PriorLearn
from _Tracking.TrackingModel import *

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.pipeline import Pipeline

from joblib import dump, load
# from sklearn.model_selection import train_test_split

Mtrain = 10000000  # number of sample paths for CMNF parameters estimation (train set)
Mtest = 10000000  # number of sample paths for CMNF parameters estimation (train set)
ml_train_part = 5000/Mtrain

pipe_file_name = "Z:/Наука - Data/2019 - Sensors - Tracking/data/pipe.joblib"
do_save = False
do_load = False


def extract_features_and_variables(States, Observations):
    X = Observations[:, :, :2 * Xb.shape[0]].reshape(Observations.shape[0] * Observations.shape[1], -1)
    Y = States[:, :, :3].reshape(States.shape[0] * States.shape[1], -1)
    return X, Y

#m_shift = np.array([0.0, 5000.0, -1000.0])

#Xb = np.array([[-10000.0, 0.0, -25.0], [-5000.0, 2000.0, -10000.0]])

states, observations = generate_sample_paths(Mtrain, 0)  # int(T / delta)
x_train, y_train = extract_features_and_variables(states, observations)

states, observations = generate_sample_paths(Mtrain, 0)  # int(T / delta)
x_test, y_test = extract_features_and_variables(states, observations)

if do_load:
    # load previous
    pipe_lasso = load(pipe_file_name)
else:
    pipe_lasso = Pipeline(steps=[
        ('polynomial', PolynomialFeatures(degree=2, interaction_only=False)),
        ('lasso', MultiTaskLassoCV(eps=0.001, n_alphas=20, normalize=True, cv=5, n_jobs=-1, max_iter=10000))
        ])

estimator_ml = PriorLearn(pipe_lasso, train_size=ml_train_part, already_fit=do_load)
estimator_linear = ConditionalMean()
estimator_cmml = ConditionalMeanPriorLearn(pipe_lasso, additional_train_size=ml_train_part, already_fit=True) # already_fit=True to fit pipe_lasso only once
estimator_ls = LeastSquares(lambda x_, y_: cart2sphere(x_[0:3])[:2 * Xb.shape[0]] - y_[:2 * Xb.shape[0]], X0Hat[0:3])
estimator_cmls = ConditionalMeanLeastSquares(lambda x_, y_: cart2sphere(x_[0:3])[:2 * Xb.shape[0]] - y_[:2 * Xb.shape[0]], X0Hat[0:3])




estimators = {
    "linear": estimator_linear,
    "LS": estimator_ls,
    "ML": estimator_ml,
    "CMNF with LS": estimator_cmls,
    "CMNF with ML": estimator_cmml,
}

for name, estimator in estimators.items():
    estimator.fit(x_train, y_train)
    predict = estimator.predict(x_test)
    std = np.std(predict - y_test, axis=0)
    print(f'{name}: {std}')


if do_save:
    dump(pipe_lasso, pipe_file_name)
