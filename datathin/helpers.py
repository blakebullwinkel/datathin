import numpy as np

def _is_non_negative(X):
    if np.min(X) < 0:
        return False
    return True

def _is_integer_valued(X):
    if np.any(X - np.floor(X) != 0):
        return False
    return True

def _is_greater_than(X, val):
    if np.min(X) <= val:
        return False
    return True

def poisthin(data, epsilon):
    if not _is_non_negative(data):
        raise ValueError("Poisson data must be non-negative.")
    if not _is_integer_valued(data):
        raise ValueError("Poisson data must be integer valued.")

    X = np.array([np.random.multinomial(x, epsilon, 1) for x in data])

    return X

def gammathin(data, epsilon, shape):
    if not _is_greater_than(data, val=0):
        raise ValueError("Gamma data must be positive.")
    if not _is_greater_than(shape, val=0):
        raise ValueError("Shape parameter must be positive.")

    X = np.array([x*np.random.dirichlet(y*epsilon, 1) for x, y in zip(data, shape)])

    return X

def normthin(data, epsilon, sigma):
    # TODO: figure out how to make this function faster!
    # TODO: I think maybe we should just flatten data and then reshape? Because surely the other functions need to be able to handle higher dimensional inputs too, right?
    if not _is_greater_than(sigma, val=0):
        raise ValueError("Variance parameter must be positive.")

    epsilon1d = np.array(epsilon) # TODO: I think we will have some line like this in the datathin() function
    epsilon2d = epsilon1d.reshape(-1,1)

    X = np.array([
        np.random.multivariate_normal(x*epsilon1d, y*(np.diag(epsilon1d) - epsilon2d@np.transpose(epsilon2d)), 1)
        for row_data, row_sigma in zip(data, sigma)
        for x, y in zip(row_data, row_sigma)
    ]).reshape(data.shape[0], data.shape[1], -1)

    return X

def _multivariate_hypergeometric_helper(colors, nsample, size):
    rng = np.random.default_rng()
    sample = rng.multivariate_hypergeometric(colors.astype(int), nsample, size)
    return sample

def binomthin(data, epsilon, pop):
    if not _is_non_negative(data):
        raise ValueError("Binomial data must be non-negative.")
    if not _is_integer_valued(data):
        raise ValueError("Binomial data must be integer valued.")
    test = np.outer(epsilon, pop)
    if not _is_integer_valued(test):
        raise ValueError("Epsilon implies non-integer thinned population parameters.")

    X = np.array([_multivariate_hypergeometric_helper(y*epsilon, x, 1) for x, y in zip(data, pop)])

    return X

def _dirichlet_multinomial_helper(n, alpha, size):
    dir_sample = np.random.dirichlet(alpha)
    dir_sample /= dir_sample.sum()
    multi_sample = np.random.multinomial(n, dir_sample, size)
    return multi_sample

def nbthin(data, epsilon, b):
    if not _is_non_negative(data):
        raise ValueError("Negative binomial data must be non-negative.")
    if not _is_integer_valued(data):
        raise ValueError("Negative binomial data must be integer valued.")
    if not _is_greater_than(b, val=0):
        raise ValueError("Overdispersion parameter must be positive.")

    X = np.array([_dirichlet_multinomial_helper(x, y*epsilon, 1) for x, y in zip(data, b)])

    return X

def normvarthin(data, mu, K):
    X = gammathin((data - mu)**2, np.repeat(1/K, K), 0.5*np.ones(data.shape))
    return X

def mvnormthin(data, epsilon, sigma):
    if len(sigma.shape) == 2:
        temp = np.zeros((data.shape[0], *sigma.shape))
        for i in range(data.shape[0]):
            temp[i, :, :] = sigma
        sigma = temp
        # TODO: check this np.tile(Sig, (data.shape[0], 1, 1))

    if sigma.shape[1] != sigma.shape[2]:
        raise ValueError("Sigma matrices must be square.")

    nfold = len(epsilon)
    X = np.zeros((*data.shape, nfold))
    resdat = data
    sigma2 = sigma

    for i in range(nfold-1):
        epsfold = epsilon[i] / np.sum(epsilon[i:nfold])

        for j in range(data.shape[0]):
            X[j,:,i] = np.random.multivariate_normal(resdat[j,:]*epsfold, epsfold*(1-epsfold)*sigma2[j,:,:], 1)
            resdat[j,:] -= X[j,:,i]

        sigma2 = sigma2*(1-epsfold)

    X[:,:,nfold-1] = resdat

    return X

def multinomthin(data, epsilon, pop):
    if not _is_non_negative(data):
        raise ValueError("Multinomial data must be non-negative.")
    if not _is_integer_valued(data):
        raise ValueError("Multinomial data must be integer valued.")
    test = np.outer(epsilon, pop)
    if not _is_integer_valued(test):
        raise ValueError("Epsilon implies non-integer thinned population parameters.")

    nfold = len(epsilon)
    X = np.zeros((*data.shape, nfold), dtype=np.int64)
    resdat = data.copy()
    pop2 = pop.copy()

    for j in range(data.shape[0]):
        pop2 = pop[j][0]*epsilon

        for i in range(nfold-1):
            # TODO: fix this [int()] silliness!
            X[j,:,i] = _multivariate_hypergeometric_helper(data[j,:], [int(pop2[i])], 1)
            resdat[j,:] -= X[j,:,i]

    X[:,:,nfold-1] = resdat

    return X

def scaledbetathin(data, alpha, K):
    if not _is_greater_than(data, val=0):
        raise ValueError("Scaled beta data must be positive.")
    if not _is_greater_than(alpha, val=0):
        raise ValueError("First beta parameter must be positive.")

    X = np.array([x*np.random.beta(y/K, 1, K) for x, y in zip(data, alpha)]).reshape(*data.shape, K)
    C = np.random.choice(np.arange(1, K+1), size=data.shape, p=np.repeat(1/K, K))-1

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            X[i, j, C[i, j]] = data[i, j]

    return X

def shiftexpthin(data, lam, K):
    if not _is_greater_than(lam, val=0):
        raise ValueError("Rate parameter must be positive.")

    X = np.array([x + np.random.exponential(K/y, K) for x, y in zip(data, lam)]).reshape(*data.shape, K)
    C = np.random.choice(np.arange(1, K+1), size=data.shape, p=np.repeat(1/K, K))-1

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            X[i, j, C[i, j]] = data[i, j]

    return X

def weibullthin(data, nu, K):
    if not _is_greater_than(data, val=0):
        raise ValueError("Weibull data must be positive.")
    if not _is_greater_than(nu, val=0):
        raise ValueError("Scale parameter must be positive.")

    X = gammathin(data**nu, np.repeat(1/K, K), np.ones(data.shape))

    return X

def gammaweibullthin(data, K, nu):
    if not _is_greater_than(data, val=0):
        raise ValueError("Weibull data must be positive.")
        
    X = gammathin(data, np.repeat(1/K, K), K*np.ones(data.shape))

    for k in range(K):
        X[:,:,k] = X[:,:,k]**(1/nu)
        
    return X

def _chisq_helper(x, K):
    Z = np.random.multivariate_normal(mean=np.zeros(K), cov=np.eye(K)).reshape(1, K)
    return (np.sqrt(x)*Z) / np.sqrt(np.sum(Z**2))

def chisqthin(data, K):
    if not _is_greater_than(data, val=0):
        raise ValueError("Chi-squared data must be positive.")

    X = np.array([_chisq_helper(x, K) for x in data])

    return X

def paretothin(data, nu, K):
    if not _is_greater_than(data, val=1):
        raise ValueError("Pareto data must be greater than 1.")
    if not _is_greater_than(nu, val=0):
        raise ValueError("Scale parameter must be positive.")

    X = gammathin(np.log(data / nu), np.repeat(1/K, K), np.ones(data.shape))

    return X
