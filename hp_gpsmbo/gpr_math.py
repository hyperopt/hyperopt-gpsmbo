"""
Formulae for Gaussian Process Regression

"""

import numpy as np
import theano.tensor as TT
from theano.sandbox.linalg import cholesky, matrix_inverse, det, psd
from .op_Kcond import normal_logEI_diff_sigma_elemwise


def dots(*args):
    rval = args[0]
    for a in args[1:]:
        rval = TT.dot(rval, a)
    return rval


def s_nll(K, y, var_y, prior_var):
    """
    Marginal negative log likelihood of model

    K - gram matrix (matrix-like)
    y - the training targets (vector-like)
    var_y - the variance of uncertainty about y (vector-like)

    :note: See RW.pdf page 37, Eq. 2.30.

    """

    n = y.shape[0]
    rK = psd(prior_var * K + var_y * TT.eye(n))

    fit = .5 * dots(y, matrix_inverse(rK), y)
    complexity = 0.5 * TT.log(det(rK))
    normalization = n / 2.0 * TT.log(2 * np.pi)
    nll = fit + complexity + normalization
    return nll


def s_mean(K, y, var_y, prior_var, K_new):
    rK = psd(prior_var * K + var_y * TT.eye(y.shape[0]))
    alpha = TT.dot(matrix_inverse(rK), y)
    y_x = TT.dot(alpha, prior_var * K_new)
    return y_x


def s_variance(K, y, var_y, prior_var, K_new, var_min):
    rK = psd(prior_var * K + var_y * TT.eye(y.shape[0]))
    L = cholesky(rK)
    v = dots(matrix_inverse(L), prior_var * K_new)
    var_x = TT.maximum(prior_var - (v ** 2).sum(axis=0), var_min)
    return var_x


def s_normal_pdf(x, mean, var):
    energy = 0.5 * ((x - mean) ** 2) / var
    return TT.exp(-energy) / TT.sqrt(2 * np.pi * var)


def s_normal_logpdf(x, mean, var):
    energy = 0.5 * ((x - mean) ** 2) / var
    return -energy - 0.5 * TT.log(2 * np.pi * var)


def s_normal_cdf(x, mean, var):
    z = (x - mean) / TT.sqrt(var)
    return .5 * TT.erfc(-z / np.sqrt(2))


def s_normal_logcdf(x, mean, var):
    z = (x - mean) / TT.sqrt(var)
    return TT.log(.5) + TT.log(TT.erfc(-z / np.sqrt(2)))


def s_normal_EI(thresh, mean, var):
    """analytic expected improvement over (above) threshold

        int_{thresh}^{\infty} (y - thresh) P(y; mean, var) dy

    """
    s_thresh = TT.as_tensor_variable(thresh)
    sigma = TT.sqrt(var)
    z = (mean - s_thresh) / sigma
    # -- the following formula is cuter, but
    #    Theano doesn't produce as stable a gradient I think?
    #return sigma * (z * s_normal_cdf(z, 0, 1) + s_normal_pdf(z, 0, 1))
    a = (mean - s_thresh) * s_normal_cdf(z, 0, 1)
    b = sigma * s_normal_pdf(z, 0, 1)
    return a + b


def s_normal_logEI(thresh, mean, var, quad_approx=False):
    """analytic log-expected improvement over (above) threshold

        log(int_{thresh}^{\infty} (y - thresh) P(y; mean, var) dy)

    quad_approx uses a 2nd-order polynomial approximation to the true function
    when the threshold is way above the mean (34 standard deviations), where
    there's almost no density to integrate.
    """
    return normal_logEI_diff_sigma_elemwise(thresh - mean, TT.sqrt(var))


def s_normal_EBI(lbound, ubound, mean, var):
    """ int_l^u (y - l) P(y; mean, var)
    """
    s_l = TT.as_tensor_variable(lbound)
    s_u = TT.as_tensor_variable(ubound)

    EI_l = s_normal_EI(s_l, mean, var)
    EI_u = s_normal_EI(s_u, mean, var)

    #sigma = TT.maximum(TT.sqrt(var), 1e-15)
    sigma = TT.sqrt(var)
    term = (s_l - s_u) * s_normal_cdf((mean - s_u) / sigma, 0, 1)

    return EI_l - EI_u + term


def s_normal_logEBI(lbound, ubound, mean, var):
    return TT.log(s_normal_EBI(lbound, ubound, mean, var))


# -- eof flake8
