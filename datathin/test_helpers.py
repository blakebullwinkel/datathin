import pytest
import numpy as np
from scipy.special import gamma
from datathin.thinning import datathin


# Helper functions

def get_ith_fold(X, i):
    return X[:,:,i]

def get_train_test(X, data, i):
    X_test = get_ith_fold(X, i)
    X_train = data - X_test
    return X_train, X_test

def assert_mean_threshold(X_train, X_test, means, delta):
    assert means[0] - delta <= np.mean(X_train) <= means[0] + delta
    assert means[1] - delta <= np.mean(X_test) <= means[1] + delta 

def assert_multivariate_mean_threshold(X_train, X_test, mean_vec, delta):
    train_mean_vec = np.mean(X_train, axis=0)
    test_mean_vec = np.mean(X_test, axis=0)
    assert np.all(np.abs(train_mean_vec - mean_vec) <= delta)
    assert np.all(np.abs(test_mean_vec - mean_vec) <= delta)

def mean_helper(X, data, means, delta=1e-1, shift_scale=False):
    n_folds = X.shape[-1]
    if n_folds == 2 or shift_scale:
        X_train, X_test = get_ith_fold(X, 0), get_ith_fold(X, 1)
        assert_mean_threshold(X_train, X_test, means, delta)
    else:
        for i in range(n_folds):
            X_train, X_test = get_train_test(X, data, i)
            assert_mean_threshold(X_train, X_test, means, delta)

def multivariate_mean_helper(X, mean_vec, delta=1e-1):
    """
    Assumes K=2.
    """
    X_train, X_test = get_ith_fold(X, 0), get_ith_fold(X, 1)
    assert_multivariate_mean_threshold(X_train, X_test, mean_vec, delta)

def variance_helper(X, variance, delta=1e-1):
    X_train, X_test = get_ith_fold(X, 0), get_ith_fold(X, 1)
    assert variance - delta <= np.var(X_train) <= variance + delta
    assert variance - delta <= np.var(X_test) <= variance + delta

def sum_helper(X, data, delta=1e-6):
    assert np.all(np.abs(np.sum(X, axis=-1) - data) <= delta)

def assert_correlation_threshold(X_train, X_test, delta):
    assert np.abs(np.corrcoef(X_train.flatten(), X_test.flatten())[0][1]) <= delta

def correlation_helper(X, data, delta=1e-1, shift_scale=False):
    n_folds = X.shape[-1]
    if n_folds == 2 or shift_scale:
        X_train, X_test = get_ith_fold(X, 0), get_ith_fold(X, 1)
        assert_correlation_threshold(X_train, X_test, delta)
    else:
        for i in range(n_folds):
            X_train, X_test = get_train_test(X, data, i)
            assert_correlation_threshold(X_train, X_test, delta)

def multivariate_correlation_helper(X, data, delta=1e-1):
    X_train, X_test = get_ith_fold(X, 0), get_ith_fold(X, 1)
    n_cols = data.shape[1]
    for i in range(n_cols):
        assert_correlation_threshold(X_train[:,i], X_test[:,i], delta)

# TODO: add shape assertion

# Thinning function tests
        
np.random.seed(0)

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


class TestScaledBeta:
    """
    For scaled beta we don't expect the folds to sum to data.
    """

    theta = 10; alpha = 7; beta = 1
    data = theta*np.random.beta(alpha, beta, size=(100000, 1))
    X = datathin(data, family="scaled-beta", arg=alpha)

    def test_means(self):
        mu = 10*(7/2)/(7/2 + 1)
        mean_helper(self.X, self.data, means=[mu, mu], shift_scale=True)

    def test_correlation(self):
        correlation_helper(self.X, self.data, shift_scale=True)


class TestShiftedExponential:

    lam = 2
    data = 6 + np.random.exponential(1/lam, size=(100000, 1))
    X = datathin(data, family="shifted-exponential", arg=lam)

    def test_means(self):
        mean_helper(self.X, self.data, means=[7.0, 7.0], shift_scale=True)

    def test_correlation(self):
        correlation_helper(self.X, self.data, shift_scale=True)


def scaled_weibull(shape, scale, size):
    X = scale*np.random.weibull(shape, size)
    return X

class TestWeibull:

    shape = 5; scale = 2
    data = scaled_weibull(shape, scale, size=(100000, 1))
    X = datathin(data, family="weibull", arg=shape)

    def test_means(self):
        mu = (1/2)*(2**5)
        mean_helper(self.X, self.data, means=[mu, mu], shift_scale=True)

    def test_correlation(self):
        correlation_helper(self.X, self.data, shift_scale=True)


class TestGammaWeibull:

    shape = 4; rate = 4
    data = np.random.gamma(shape, 1/rate, size=(100000, 1))
    X = datathin(data, family="gamma-weibull", K=4, arg=3)

    def test_means(self):
        mu = 4**(-1/3)*gamma(1+(1/3))
        mean_helper(self.X, self.data, means=[mu, mu], shift_scale=True)

    def test_correlation(self):
        correlation_helper(self.X, self.data, shift_scale=True)


class TestGaussianVariance:
    """
    Tests for normal distribution with known mean, unknown variance.
    """

    mu = 5; var = 2
    # TODO: size should be (10000,10) here; still have to add size tests
    data = np.random.normal(mu, scale=np.sqrt(var), size=(100000,1))
    X = datathin(data, family="normal-variance", arg=mu)

    def test_means(self):
        mean_helper(self.X, self.data, means=[1.0, 1.0])

    def test_correlation(self):
        correlation_helper(self.X, self.data)


class TestChiSquared:

    data = 3*np.random.chisquare(df=5, size=(100000,1))
    X = datathin(data, family="chi-squared", K=5)

    def test_variances(self):
        variance = np.mean(self.data) / 5
        variance_helper(self.X, variance=variance)

    def test_correlation(self):
        correlation_helper(self.X, self.data, shift_scale=True)


class TestPareto:

    a = 6; m = 2
    data = (np.random.pareto(a=a, size=(100000,1)) + 1) * m
    X = datathin(data, family="pareto", arg=m)

    def test_means(self):
        mu = (1/2)/6
        mean_helper(self.X, self.data, means=[mu, mu])

    def test_correlation(self):
        correlation_helper(self.X, self.data)


class TestMultivariateGaussian:

    mu = np.array([1, 2, 3, 4])
    Sig = np.array([[0.6**np.abs(i-j) for i in range(4)] for j in range(4)])
    data = np.random.multivariate_normal(mean=mu, cov=Sig, size=100000)
    X = datathin(data, family="mvnormal", arg=Sig)

    def test_means(self):
        multivariate_mean_helper(self.X, mean_vec=0.5*self.mu)

    def test_correlation(self):
        multivariate_correlation_helper(self.X, self.data)


class TestMultinomial:

    n = 20; pvals = np.array([0.1, 0.2, 0.3, 0.4])
    data = np.random.multinomial(n=n, pvals=pvals, size=100000)
    X = datathin(data, family="multinomial", arg=n)

    def test_means(self):
        multivariate_mean_helper(self.X, mean_vec=(self.n/2)*self.pvals)

    def test_correlation(self):
        multivariate_correlation_helper(self.X, self.data)
