import warnings
import numpy as np
from datathin.helpers import *

multivariate_distributions = ["mvnormal", "mvgaussian", "multinomial"]

def datathin(data, family, K=2, epsilon=None, arg=None) -> np.ndarray:

    # Validate inputs
    if not isinstance(data, np.ndarray):
        if isinstance(data, list):
            data = np.array(data)
        else:
            raise ValueError("`data` must be either a list or numpy array.")
    data_shape = data.shape

    if family in ["poisson", "exponential", "chi-squared", "scaled-uniform"]:
        if arg is not None:
            raise warnings.warn(
                f"Extra parameter provided was not used in {family} thinning.",
                UserWarning
            )
    else:
        if arg is None:
            raise ValueError("Extra parameter is missing.")

        if not isinstance(arg, np.ndarray):
            if isinstance(arg, (int, float)):
                if family == "multinomial":
                    arg = np.repeat(arg, data_shape[0])
                else:
                    arg = arg * np.ones(data_shape)
            elif isinstance(arg, list):
                arg = np.array(arg)
            else:
                raise ValueError(
                    "`arg` must be either a numerical value or an array of numerical values."
                )
        arg_shape = arg.shape

        if family in ["mvnormal", "mvgaussian"]:
            if len(arg_shape) == 2:
                # For MVN with n x p data matrix, arg can either be 1) a p x p covariance matrix
                if arg_shape != (data_shape[1], data_shape[1]):
                    raise ValueError(
                        "Incorrect dimensions for multivariate normal covariance matrices."
                    )
            elif len(arg_shape) == 3:
                # Or 2) arg can be an n x p x p array of covariance matrices
                if arg_shape != (data_shape[0], data_shape[1], data_shape[1]):
                    raise ValueError(
                        "Incorrect dimensions for multivariate normal covariance matrices."
                    )
        elif family == "multinomial":
            if arg_shape[0] != data_shape[0]:
                # TODO: check this condition
                raise ValueError(
                    "Incorrect dimensions for multinomial trials parameter."
                )
        else:
            if arg_shape[0] > 1:
                if arg_shape != data_shape:
                    raise ValueError(
                        "If `arg` is not a scalar, its dimensions must match those of `data`."
                    )

    if family in [
        "poisson",
        "negative binomial",
        "normal",
        "gaussian",
        "mvnormal",
        "mvgaussian",
        "binomial",
        "multinomial",
        "exponential",
        "gamma",
    ]:
        if epsilon is None:
            epsilon = np.repeat(1 / K, K)
        else:
            if np.sum(epsilon) != 1:
                raise ValueError("`epsilon` does not sum to 1.")
            if len(epsilon) != K:
                raise warnings.warn(
                    "`K` parameter will be ignored in favor of the length of `epsilon`."
                )
        n_folds = len(epsilon)
    else:
        n_folds = K

    # TODO: refine this, currently the idea is to flatten data array for univariate distributions
    if family in [
        "poisson",
        "exponential",
        "gamma",
        "negative binomial",
        "binomial",
    ]:
        data = data.flatten()

    if family == "poisson":
        X = poisthin(data, epsilon)
    elif family == "negative binomial":
        X = nbthin(data, epsilon, arg)
    elif family in ["normal", "gaussian"]:
        X = normthin(data, epsilon, arg)
    elif family in ["normal-variance", "gaussian-variance"]:
        X = normvarthin(data, arg, K)
    elif family in ["mvnormal", "mvgaussian"]:
        X = mvnormthin(data, epsilon, arg)
    elif family == "binomial":
        X = binomthin(data, epsilon, arg)
    elif family == "multinomial":
        X = multinomthin(data, epsilon, arg)
    elif family == "exponential":
        X = gammathin(data, epsilon, np.ones(data_shape))
    elif family == "gamma":
        X = gammathin(data, epsilon, arg)
    elif family == "chi-squared":
        X = chisqthin(data, K)
    elif family == "gamma-weibull":
        X = gammaweibullthin(data, K, arg)
    elif family == "weibull":
        X = weibullthin(data, arg, K)
    elif family == "pareto":
        X = paretothin(data, arg, K)
    elif family == "shifted-exponential":
        X = shiftexpthin(data, arg, K)
    elif family == "scaled-uniform":
        X = scaledbetathin(data, np.ones(data_shape), K)
    elif family == "scaled-beta":
        X = scaledbetathin(data, arg, K)
    else:
        raise ValueError(f"Family `{family}` not recognized.")
    
    X = X.reshape((*data_shape, n_folds))

    return X

if __name__ == "__main__":
    data = np.random.poisson(lam=7, size=(100000,1))
    X = datathin(data, family="poisson", epsilon=[0.3, 0.7])
    n_folds = 2
    print((*data.shape, n_folds))
