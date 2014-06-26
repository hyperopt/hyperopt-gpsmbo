
import theano
import theano.tensor as TT

from .gpr_math import s_nll, s_mean, s_variance

#TODO: Match name to scikits.learn
def euclidean_sq_distances(x, z):
    """Matrix of distances for each row in x to each row in z
    """

    # -- TODO: better numerical accuracy
    d = ((x ** 2).sum(axis=1).dimshuffle(0, 'x')
            + (z ** 2).sum(axis=1)
            - 2 * TT.dot(x, z.T))
    return TT.maximum(d, 0)


class Kernel(object):

    def s_nll_params(self, x, y, var_y, prior_var, params=None, ret_K=False):
        # return a cost, and parameter vector suitable for fitting
        # the GP, and bounds on that parameter vector

        # -- turn these to constants
        x = TT.as_tensor_variable(x)
        y = TT.as_tensor_variable(y)
        if params is None:
            params = theano.tensor.dvector()
        else:
            params = theano.tensor.as_tensor_variable(params)
            assert params.ndim == 1
        K, params0, bounds = self.opt_K(x, params)
        nll = s_nll(K, y, var_y=var_y, prior_var=prior_var)
        if ret_K:
            return nll, params, params0, bounds, K
        return nll, params, params0, bounds

    def s_mean_var(self, x, y, var_y, prior_var, best_params, var_min,
                  x_new=None,
                  return_K_new=False):
        # s_mean, s_x for computing mean from s_x

        # -- turn these to constants
        x = TT.as_tensor_variable(x)
        y = TT.as_tensor_variable(y)
        if x_new is None:
            x_new = TT.matrix()
        else:
            assert x_new.ndim == 2
        params = TT.as_tensor_variable(best_params)
        K, K_new = self.predict_K(x, x_new, params)
        K.name = 'K'
        K_new.name = 'K_new'
        mean = s_mean(K, y, var_y, prior_var, K_new)
        var = s_variance(K, y, var_y, prior_var, K_new, var_min)
        mean.name = 'mean_new'
        var.name = 'var_new'
        rval = [mean, var, x_new]
        if return_K_new:
            rval.append(K_new)
        return rval

    def predict_K(self, *args, **kwargs):
        logK, logK_new = self.predict_logK(*args, **kwargs)
        return TT.exp(logK), TT.exp(logK_new)

    def opt_K(self, *args, **kwargs):
        logK, params, bounds = self.opt_logK(*args, **kwargs)
        return TT.exp(logK), params, bounds

