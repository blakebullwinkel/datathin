import pytest
import numpy as np
from datathin.helpers import *

np.random.seed(0)

def mean_helper(X, means, delta):
    for i in range(X.shape[-1]):
        assert means[i] - delta <= np.mean(X[:,:,i]) <= means[i] + delta

def sum_helper(X, data):
    assert np.all(np.sum(X, axis=-1) == data)

def correlation_helper(X, data, delta):
    for i in range(X.shape[-1]):
        X_test = X[:,:,i]
        X_train = data - X_test
        assert np.abs(np.corrcoef(X_train.flatten(), X_test.flatten())[0][1]) <= delta

class TestPoisson:

    data = np.random.poisson(lam=7, size=(100000,1))
    X = poisthin(data, epsilon=[0.3, 0.7])

    def test_means(self):
        mean_helper(self.X, means=[2.1, 4.9], delta=0.1)

    def test_sum(self):
        sum_helper(self.X, self.data)

    def test_correlation(self):
        correlation_helper(self.X, self.data, delta=0.1)

