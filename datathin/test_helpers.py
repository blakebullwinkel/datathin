import pytest
import numpy as np
from scipy.special import gamma
from datathin.thinning import datathin


# Test helper functions

def get_ith_fold(X, i):
    """
    Returns the ith fold of X.
    """
    return X[:,:,i]

def get_train_test(X, data, i):
    """
    Returns X_train and X_test, where X_test is the ith fold of 
    X and X_train is computing using the property that X_train 
    and X_test sum to data.
    """
    X_test = get_ith_fold(X, i)
    X_train = data - X_test
    return X_train, X_test

def assert_mean_threshold(X_train, X_test, means, delta):
    """
    Tests whether the means of X_train and X_test are equal to 
    their expected means, within some delta.
    """
    assert means[0] - delta <= np.mean(X_train) <= means[0] + delta
    assert means[1] - delta <= np.mean(X_test) <= means[1] + delta 

def assert_multivariate_mean_threshold(X_train, X_test, mean_vec, delta):
    """
    Tests whether the means of multivariate data X_train and 
    X_test are equal to their expected mean vectors, within some 
    delta.
    """
    train_mean_vec = np.mean(X_train, axis=0)
    test_mean_vec = np.mean(X_test, axis=0)
    assert np.all(np.abs(train_mean_vec - mean_vec) <= delta)
    assert np.all(np.abs(test_mean_vec - mean_vec) <= delta)

def mean_helper(X, data, means, delta=1e-1, use_ith_fold=False):
    """
    Mean test helper function. If X has only two folds or we
    specify use_ith_fold=True, X_train and X_test will be indexed
    directly from the first two folds of X. Otherwise, they are
    computed using the summation property. 
    """
    n_folds = X.shape[-1]
    if n_folds == 2 or use_ith_fold:
        X_train, X_test = get_ith_fold(X, 0), get_ith_fold(X, 1)
        assert_mean_threshold(X_train, X_test, means, delta)
    else:
        for i in range(n_folds):
            X_train, X_test = get_train_test(X, data, i)
            assert_mean_threshold(X_train, X_test, means, delta)

def multivariate_mean_helper(X, mean_vec, delta=1e-1):
    """
    Mean test helper function for multivariate data. Assumes K=2 
    folds in X.
    """
    X_train, X_test = get_ith_fold(X, 0), get_ith_fold(X, 1)
    assert_multivariate_mean_threshold(X_train, X_test, mean_vec, delta)

def variance_helper(X, variance, delta=1e-1):
    """
    Tests whether the variance of X_train and X_test are equal to 
    their expected variance. Assumes K=2 folds in X.
    """
    X_train, X_test = get_ith_fold(X, 0), get_ith_fold(X, 1)
    assert variance - delta <= np.var(X_train) <= variance + delta
    assert variance - delta <= np.var(X_test) <= variance + delta

def sum_helper(X, data, delta=1e-6):
    """
    Tests whether the folds of X sum to data, within some delta.
    """
    assert np.all(np.abs(np.sum(X, axis=-1) - data) <= delta)

def assert_correlation_threshold(X_train, X_test, delta):
    """
    Tests whether the correlation between X_train and X_test is
    zero, within some delta.
    """
    assert np.abs(np.corrcoef(X_train.flatten(), X_test.flatten())[0][1]) <= delta

def correlation_helper(X, data, delta=1e-1, use_ith_fold=False):
    """
    Correlation test helper function. If X has only two folds or we
    specify use_ith_fold=True, X_train and X_test will be indexed
    directly from the first two folds of X. Otherwise, they are
    computed using the summation property. 
    """
    n_folds = X.shape[-1]
    if n_folds == 2 or use_ith_fold:
        X_train, X_test = get_ith_fold(X, 0), get_ith_fold(X, 1)
        assert_correlation_threshold(X_train, X_test, delta)
    else:
        for i in range(n_folds):
            X_train, X_test = get_train_test(X, data, i)
            assert_correlation_threshold(X_train, X_test, delta)

def multivariate_correlation_helper(X, data, delta=1e-1):
    """
    Correlation test helper function for multivariate data. Assumes 
    K=2 folds in X.
    """
    X_train, X_test = get_ith_fold(X, 0), get_ith_fold(X, 1)
    n_cols = data.shape[1]
    for i in range(n_cols):
        assert_correlation_threshold(X_train[:,i], X_test[:,i], delta)

def shape_helper(X, data, n_folds=2):
    """
    Tests whether X has the correct shape.
    """
    assert X.shape == (*data.shape, n_folds)


# Thinning function tests
        
class TestPoisson:
    """
    Test class for Poisson data.
    """

    data = np.random.poisson(lam=7, size=(100000,1))
    X = datathin(data, family="poisson", epsilon=[0.3, 0.7])

    def test_means(self):
        mean_helper(self.X, self.data, means=[2.1, 4.9])

    def test_sum(self):
        sum_helper(self.X, self.data)

    def test_correlation(self):
        correlation_helper(self.X, self.data)

    def test_shape(self):
        shape_helper(self.X, self.data)

class TestExponential:
    """
    Test class for exponential data.
    """

    data = np.random.exponential(scale=5, size=(100000,1))
    X = datathin(data, family="exponential", K=5)

    def test_means(self):
        mean_helper(self.X, self.data, means=[4.0, 1.0])

    def test_sum(self):
        sum_helper(self.X, self.data)

    def test_correlation(self):
        correlation_helper(self.X, self.data)

    def test_shape(self):
        shape_helper(self.X, self.data, n_folds=5)

class TestGamma:
    """
    Test class for gamma data.
    """

    # figure out why size=(10000,2) doesn't work + test size=(10000,)
    data = np.random.gamma(shape=12, scale=1/2, size=(10000,1))
    X = datathin(data, family="gamma", arg=12)

    def test_means(self):
        mean_helper(self.X, self.data, means=[3.0, 3.0])

    def test_sum(self):
        sum_helper(self.X, self.data)

    def test_correlation(self):
        correlation_helper(self.X, self.data)

    def test_shape(self):
        shape_helper(self.X, self.data)

class TestGaussianMean1:
    """
    Test class for data drawn from a single Gaussian with known 
    variance and unknown mean.
    """

    data = np.random.normal(loc=5, scale=np.sqrt(2), size=(10000,10))
    X = datathin(data, family="normal", arg=2)

    def test_means(self):
        mean_helper(self.X, self.data, means=[2.5, 2.5])

    def test_sum(self):
        sum_helper(self.X, self.data)

    def test_correlation(self):
        correlation_helper(self.X, self.data)

    def test_shape(self):
        shape_helper(self.X, self.data)

class TestGaussianMean2:
    """
    Test class for data drawn from multiple Gaussians with known
    variances and unknown mean.
    """

    data = np.random.normal(loc=5, scale=(np.sqrt(0.1), np.sqrt(2), np.sqrt(20)), size=(100000,3))
    correct_args = np.ones((100000,3))*np.array([0.1, 2, 20])
    X = datathin(data, family="normal", arg=correct_args)

    def test_means(self):
        mean_helper(self.X, self.data, means=[2.5, 2.5])

    def test_sum(self):
        sum_helper(self.X, self.data)

    def test_correlation(self):
        correlation_helper(self.X, self.data)

    def test_shape(self):
        shape_helper(self.X, self.data)

class TestGaussianVariance:
    """
    Test class for data drawn from Gaussian with known mean and 
    unknown variance.
    """

    mu = 5; var = 2
    # TODO: size should be (10000,10) here; still have to add size tests
    data = np.random.normal(mu, scale=np.sqrt(var), size=(100000,1))
    X = datathin(data, family="normal-variance", arg=mu)

    def test_means(self):
        mean_helper(self.X, self.data, means=[1.0, 1.0])

    def test_correlation(self):
        correlation_helper(self.X, self.data)

    def test_shape(self):
        shape_helper(self.X, self.data)

class TestNegativeBinomial:
    """
    Test class for negative binomial data.
    """

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

    def test_shape(self):
        shape_helper(self.X, self.data)

class TestBinomial:
    """
    Test class for binomial data.
    """

    data = np.random.binomial(n=16, p=0.25, size=(100000,1))
    X = datathin(data, family="binomial", arg=16)

    def test_means(self):
        mean_helper(self.X, self.data, means=[2.0, 2.0])

    def test_sum(self):
        sum_helper(self.X, self.data)

    def test_correlation(self):
        correlation_helper(self.X, self.data)

    def test_shape(self):
        shape_helper(self.X, self.data)

class TestScaledBeta:
    """
    Test class for scaled beta data.
    """

    theta = 10; alpha = 7; beta = 1
    data = theta*np.random.beta(alpha, beta, size=(100000, 1))
    X = datathin(data, family="scaled-beta", arg=alpha)

    def test_means(self):
        mu = 10*(7/2)/(7/2 + 1)
        mean_helper(self.X, self.data, means=[mu, mu], use_ith_fold=True)

    def test_correlation(self):
        correlation_helper(self.X, self.data, use_ith_fold=True)

    def test_shape(self):
        shape_helper(self.X, self.data)

class TestShiftedExponential:
    """
    Test class for shifted exponential data.
    """

    lam = 2
    data = 6 + np.random.exponential(1/lam, size=(100000, 1))
    X = datathin(data, family="shifted-exponential", arg=lam)

    def test_means(self):
        mean_helper(self.X, self.data, means=[7.0, 7.0], use_ith_fold=True)

    def test_correlation(self):
        correlation_helper(self.X, self.data, use_ith_fold=True)

    def test_shape(self):
        shape_helper(self.X, self.data)

class TestWeibull:
    """
    Test class for Weibull data.
    """

    shape = 5; scale = 2
    data = scale*np.random.weibull(shape, size=(100000, 1))
    X = datathin(data, family="weibull", arg=shape)

    def test_means(self):
        mu = (1/2)*(2**5)
        mean_helper(self.X, self.data, means=[mu, mu], use_ith_fold=True)

    def test_correlation(self):
        correlation_helper(self.X, self.data, use_ith_fold=True)

    def test_shape(self):
        shape_helper(self.X, self.data)

class TestGammaWeibull:
    """
    Test class for Gamma-Weibull data.
    """

    shape = 4; rate = 4
    data = np.random.gamma(shape, 1/rate, size=(100000, 1))
    X = datathin(data, family="gamma-weibull", K=4, arg=3)

    def test_means(self):
        mu = 4**(-1/3)*gamma(1+(1/3))
        mean_helper(self.X, self.data, means=[mu, mu], use_ith_fold=True)

    def test_correlation(self):
        correlation_helper(self.X, self.data, use_ith_fold=True)

    def test_shape(self):
        shape_helper(self.X, self.data, n_folds=4)

class TestChiSquared:
    """
    Test class for Chi-squared data.
    """

    data = 3*np.random.chisquare(df=5, size=(100000,1))
    X = datathin(data, family="chi-squared", K=5)

    def test_variances(self):
        variance = np.mean(self.data) / 5
        variance_helper(self.X, variance=variance)

    def test_correlation(self):
        correlation_helper(self.X, self.data, use_ith_fold=True)

    def test_shape(self):
        shape_helper(self.X, self.data, n_folds=5)

class TestPareto:
    """
    Test class for Pareto data.
    """

    a = 6; m = 2
    data = (np.random.pareto(a=a, size=(100000,1)) + 1) * m
    X = datathin(data, family="pareto", arg=m)

    def test_means(self):
        mu = (1/2)/6
        mean_helper(self.X, self.data, means=[mu, mu])

    def test_correlation(self):
        correlation_helper(self.X, self.data)

    def test_shape(self):
        shape_helper(self.X, self.data)

class TestMultivariateGaussian:
    """
    Test class for multivariate Gaussian data with known 
    covariance matrix.
    """

    mu = np.array([1, 2, 3, 4])
    Sig = np.array([[0.6**np.abs(i-j) for i in range(4)] for j in range(4)])
    data = np.random.multivariate_normal(mean=mu, cov=Sig, size=100000)
    X = datathin(data, family="mvnormal", arg=Sig)

    def test_means(self):
        multivariate_mean_helper(self.X, mean_vec=0.5*self.mu)

    def test_correlation(self):
        multivariate_correlation_helper(self.X, self.data)

    def test_shape(self):
        shape_helper(self.X, self.data)


class TestMultinomial:
    """
    Test class for multinomial data.
    """

    n = 20; pvals = np.array([0.1, 0.2, 0.3, 0.4])
    data = np.random.multinomial(n=n, pvals=pvals, size=100000)
    X = datathin(data, family="multinomial", arg=n)

    def test_means(self):
        multivariate_mean_helper(self.X, mean_vec=(self.n/2)*self.pvals)

    def test_correlation(self):
        multivariate_correlation_helper(self.X, self.data)

    def test_shape(self):
        shape_helper(self.X, self.data)
