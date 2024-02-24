import pytest
import numpy as np
from datathin.thinning import datathin

np.random.seed(0)

def train_test_helper(X, data, i):
    X_test = X[:,:,i]
    X_train = data - X_test   
    return X_train, X_test

def mean_helper(X, data, means, delta=1e-1):
    n_folds = X.shape[-1]
    if n_folds > 2:
        for i in range(n_folds):
            X_train, X_test = train_test_helper(X, data, i)
            assert means[0] - delta <= np.mean(X_train) <= means[0] + delta
            assert means[1] - delta <= np.mean(X_test) <= means[1] + delta
    else:
        for i in range(n_folds):
            assert means[i] - delta <= np.mean(X[:,:,i]) <= means[i] + delta

def sum_helper(X, data, delta=1e-6):
    assert np.all((np.sum(X, axis=-1) - data) <= delta)

def correlation_helper(X, data, delta=1e-1):
    n_folds = X.shape[-1]
    for i in range(n_folds):
        X_train, X_test = train_test_helper(X, data, i)
        assert np.abs(np.corrcoef(X_train.flatten(), X_test.flatten())[0][1]) <= delta

class TestPoisson:

    data = np.random.poisson(lam=7, size=(100000,1))
    X = datathin(data, family="poisson", epsilon=[0.3, 0.7])

    def test_means(self):
        mean_helper(self.X, self.data, means=[2.1, 4.9])

    def test_sum(self):
        sum_helper(self.X, self.data)

    def test_correlation(self):
        correlation_helper(self.X, self.data)


class TestExponential:

    data = np.random.exponential(scale=5, size=(100000,1))
    X = datathin(data, family="exponential", K=5)

    def test_means(self):
        mean_helper(self.X, self.data, means=[4.0, 1.0])

    def test_sum(self):
        sum_helper(self.X, self.data)

    def test_correlation(self):
        correlation_helper(self.X, self.data)


class TestGamma:

    # figure out why size=(10000,2) doesn't work + test size=(10000,)
    data = np.random.gamma(shape=12, scale=1/2, size=(10000,1))
    X = datathin(data, family="gamma", arg=12)

    def test_means(self):
        mean_helper(self.X, self.data, means=[3.0, 3.0])

    def test_sum(self):
        sum_helper(self.X, self.data)

    def test_correlation(self):
        correlation_helper(self.X, self.data)


class TestGaussian1:

    data = np.random.normal(loc=5, scale=np.sqrt(2), size=(10000,10))
    X = datathin(data, family="normal", arg=2)

    def test_means(self):
        mean_helper(self.X, self.data, means=[2.5, 2.5])

    def test_sum(self):
        sum_helper(self.X, self.data)

    def test_correlation(self):
        correlation_helper(self.X, self.data)


class TestGaussian2:

    data = np.random.normal(loc=5, scale=(np.sqrt(0.1), np.sqrt(2), np.sqrt(20)), size=(100000,3))
    correct_args = np.ones((100000,3))*np.array([0.1, 2, 20])
    X = datathin(data, family="normal", arg=correct_args)

    def test_means(self):
        mean_helper(self.X, self.data, means=[2.5, 2.5])

    def test_sum(self):
        sum_helper(self.X, self.data)

    def test_correlation(self):
        correlation_helper(self.X, self.data)


class TestNegativeBinomial:

    n = 7; mu = 6
    p = n / (n + mu)
    data = np.random.negative_binomial(n=n, p=p, size=(100000,1))
    X = datathin(data, family="negative binomial", epsilon=[0.2, 0.8], arg=n)

    def test_means(self):
        mean_helper(self.X, self.data, means=[1.2, 4.8])

    def test_sum(self):
        sum_helper(self.X, self.data)

    def test_correlation(self):
        correlation_helper(self.X, self.data)


class TestBinomial:

    data = np.random.binomial(n=16, p=0.25, size=(100000,1))
    X = datathin(data, family="binomial", arg=16)

    def test_means(self):
        mean_helper(self.X, self.data, means=[2.0, 2.0])

    def test_sum(self):
        sum_helper(self.X, self.data)

    def test_correlation(self):
        correlation_helper(self.X, self.data)
