import warnings
import numpy as np
from datathin.helpers import *


def datathin(data, family, K=2, epsilon=None, arg=None) -> np.ndarray:

    # Validate inputs
    if isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, np.ndarray):
        pass
    else:
        raise ValueError("`data` must be either a list or numpy array.")
    data_shape = data.shape

    if family in ["poisson", "exponential", "chi-squared", "scaled-uniform"]:
        if arg is not None:
            raise warnings.warn(
                f"Extra parameter provided was not used in {family} thinning.",
                UserWarning,
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
                    "`arg` must be either a numerical value or array of numerical values."
                )
        arg_shape = arg.shape

        if family in ["mvnormal", "mvgaussian"]:
            if len(arg_shape) == 2:
                # TODO: check this condition
                if np.any(arg_shape != [data_shape[1], data_shape[1]]):
                    raise ValueError(
                        "Incorrect dimensions for multivariate normal covariance matrices."
                    )
            elif len(arg_shape) == 3:
                if np.any(arg_shape != [data_shape[0], data_shape[1], data_shape[1]]):
                    raise ValueError(
                        "Incorrect dimensions for multivariate normal covariance matrices."
                    )
        elif family == "multinomial":
            if len(arg) not in [1, data_shape[0]]:
                raise ValueError(
                    "Incorrect dimensions for multinomial trials parameter."
                )
        else:
            if len(arg) > 1:
                if np.any(arg_shape != data_shape):
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

    return X


if __name__ == "__main__":

    data = np.random.poisson(lam=7, size=(100000, 1))
    print(data.shape)

    X = datathin(data, family="poisson", K=2)
    print(X.shape)
