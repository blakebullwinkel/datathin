import datathin.helpers as helpers
import warnings
import numpy as np

from typing import Union, List
from pydantic import BaseModel, validator

class DataThinner(BaseModel):
    data: Union[List[Union[int, float]], np.ndarray]
    family: str
    K: int = 2
    epsilon: Union[List[float], np.ndarray, None] = None
    arg: Union[int, float, List[Union[int, float]], np.ndarray, None] = None

def datathin(data, family, K=2, epsilon=None, arg=None) -> np.ndarray:
    data = np.array(data)
    data_shape = data.shape
    # TODO: preprocess arg in a better way than this
    if isinstance(arg, (int, float)):
        arg_arr = np.array([arg])
    else:
        arg_arr = np.array(arg)
    arg_shape = arg_arr.shape

    if family in ["poisson", "exponential", "chi-squared", "scaled-uniform"]:
        if arg is not None:
            raise warnings.warn(
                f"Extra parameter provided was not used in {family} thinning.", 
                UserWarning
            )
    else:
        if arg is None:
            raise ValueError("Extra parameter is missing.")
        elif not isinstance(arg, (int, float)): # TODO: figure out how to check this properly
            raise ValueError("Non-numeric parameter provided.")
        else:
            if family in ["mvnormal", "mvgaussian"]:
                if len(arg_shape) == 2:
                    # TODO: check this condition
                    if np.any(arg_shape != [data_shape[1], data_shape[1]]):
                        raise ValueError("Incorrect dimensions for multivariate normal covariance matrices.")
                elif len(arg_shape) == 3:
                    if np.any(arg_shape != [data_shape[1], data_shape[1], data_shape[1]]):
                        raise ValueError("Incorrect dimensions for multivariate normal covariance matrices.")
            elif family == "multinomial":
                if len(arg_arr) not in [1, data_shape[0]]:
                    raise ValueError("Incorrect dimensions for multinomial trials parameter.")
            else:
                if len(arg_arr) > 1:
                    if np.any(arg_shape != data_shape):
                        raise ValueError("If `arg` is not a scalar, its dimensions must match those of `data`.")
                    
    if family in ["poisson", "negative binomial", "normal", "gaussian", "mvnormal",
                  "mvgaussian", "binomial", "multinomial", "exponential", "gamma"]:
        if epsilon is None:
            epsilon = np.repeat(1/K, K)
        else:
            if np.sum(epsilon) != 1:
                raise ValueError("`epsilon` does not sum to 1.")
            if len(epsilon) != K:
                raise warnings.warn("`K` parameter will be ignored in favor of the length of `epsilon`.")
            
    if len(arg_arr) == 1:
        if family == "multinomial":
            arg = np.repeat(arg, data_shape[0])
        else:
            arg = arg*np.ones(data_shape)

    if family == "poisson":
        X = helpers.poisthin(data, epsilon)
    elif family == "negative binomial":
        X = helpers.nbthin(data, epsilon, arg)
    elif family in ["normal", "gaussian"]:
        X = helpers.normthin(data, epsilon, arg)
    elif family in ["normal-variance", "gaussian-variance"]:
        X = helpers.normvarthin(data, arg, K)
    elif family in ["mvnormal", "mvgaussian"]:
        X = helpers.mvnormthin(data, epsilon, arg)
    elif family == "binomial":
        X = helpers.binomthin(data, epsilon, arg)
    elif family == "multinomial":
        X = helpers.multinomthin(data, epsilon, arg)
    elif family == "exponential":
        X = helpers.gammathin(data, epsilon, np.ones(data_shape))
    elif family == "gamma":
        X = helpers.gammathin(data, epsilon, arg)
    elif family == "chi-squared":
        X = helpers.chisqthin(data, K)
    elif family == "gamma-weibull":
        X = helpers.gammaweibullthin(data, K, arg)
    elif family == "weibull":
        X = helpers.weibullthin(data, arg, K)
    elif family == "pareto":
        X = helpers.paretothin(data, arg, K)
    elif family == "shifted-exponential":
        X = helpers.shiftexpthin(data, arg, K)
    elif family == "scaled-uniform":
        X = helpers.scaledbetathin(data, np.ones(data_shape), K)
    elif family == "scaled-beta":
        X = helpers.scaledbetathin(data, arg, K)
    else:
        raise ValueError(f"Family `{family}` not recognized.")
    
    return X

if __name__ == "__main__":

    pass