import time
import numpy as np
import scipy.optimize
import theano
import theano.tensor as TT
import theano.sandbox.rng_mrg
from .gpr_math import s_normal_logEI
from .hmc import HMC_sampler


def raises(exc, fn, args):
    try:
        fn(*args)
        return False
    except exc:
        return True
    return False


class GPR_Base(object):
    def __init__(self, kernel,
                 maxiter=None,
                 prior_var=None,
                 prior_mean=None,
                 warn_floatX=True,
                 ):
        self.kernel = kernel
        self.maxiter = maxiter
        self.prior_var = prior_var
        self.prior_mean = prior_mean
        self.s_var_min = TT.as_tensor_variable(1e-8, name='s_var_min')
        self.s_emp_mean = theano.shared(0.0, name='s_emp_mean')
        self.s_emp_var = theano.shared(1.0, name='s_emp_var')
        self.s_X = theano.shared(np.zeros((2, 2)), name='s_X')
        self.s_y = theano.shared(np.zeros((2,)), name='s_y')
        self.s_var_y_raw = theano.shared(np.zeros(2,), name='s_var_y_raw')
        self.s_params = theano.tensor.dvector('params')
        self._logEI_cache = {}
        if theano.config.floatX != 'float64':
            raise TypeError('GPR requires floatX==float64')

        self.s_var_y = TT.maximum(self.s_var_y_raw, self.s_var_min)

    def set_emp_mean(self, y):
        if self.prior_mean is None:
            self.s_emp_mean.set_value(np.mean(y))
        else:
            self.s_emp_mean.set_value(self.prior_mean)

    def set_emp_var(self, y, var_y):
        self.s_var_y_raw.set_value(np.zeros(len(y)) + var_y)
        if self.prior_var is None:
            self.s_emp_var.set_value(max(np.var(y),
                                         np.min(var_y),
                                         1e-6))
        else:
            self.s_emp_var.set_value(self.prior_var)

    def set_Xy(self, X, y):
        X_ = np.atleast_2d(X)
        self.s_X.set_value(X_)
        self.s_y.set_value(np.atleast_1d(y) - self.s_emp_mean.get_value())
        return self.s_X, self.s_y

    def fit(self, X, y, var_y=0.0):
        self.set_emp_mean(y)
        self.set_emp_var(y, var_y)
        s_X, s_y = self.set_Xy(X, y)

        _, params, params0, _ = self.kernel.s_nll_params(
            X, y,
            var_y=var_y,
            prior_var=self.s_emp_var)

        self._params_list = [params0.copy()]
        self._params_weights = [1.0]


    def predict(self, x, eval_MSE=False):
        if eval_MSE:
            return self.mean_variance(x)
        else:
            return self.mean(x)

    def mean(self, x):
        """
        Compute mean at points in x_new
        """
        try:
            self._mean
        except AttributeError:
            s_mean_x, s_var_x, s_x = self.kernel.s_mean_var(
                self.s_X,
                self.s_y,
                self.s_var_y,
                self.s_emp_var,
                self.s_params,
                self.s_var_min)
            self._mean = theano.function(
                [s_x, self.s_params],
                s_mean_x + self.s_emp_mean,
                allow_input_downcast=True,)
        means = [self._mean(x, p) for p in self._params_list]
        weights = self._params_weights
        return np.dot(weights, means)

    def mean_variance(self, x):
        """
        Compute mean and variance at points in x_new
        """
        try:
            self._mean_variance
        except AttributeError:
            s_mean_x, s_var_x, s_x = self.kernel.s_mean_var(
                self.s_X,
                self.s_y,
                self.s_var_y,
                self.s_emp_var,
                self.s_params,
                self.s_var_min)
            self._mean_variance = theano.function(
                [s_x, self.s_params],
                [s_mean_x + self.s_emp_mean, s_var_x],
                allow_input_downcast=True,)
        means, variances = zip(*[
            self._mean_variance(x, p) for p in self._params_list])
        weights = self._params_weights
        mean = np.dot(weights, means)
        variance = np.dot(weights, variances)
        return mean, variance

    def logEI_fn(self, direction, quad_approx):
        direction = float(direction)
        quad_approx = bool(quad_approx)
        try:
            self._logEI_cache[(direction, quad_approx)]
        except KeyError:
            s_thresh = TT.dscalar('thresh')
            s_mean_x, s_var_x, s_x = self.kernel.s_mean_var(
                self.s_X,
                self.s_y,
                self.s_var_y,
                self.s_emp_var,
                self.s_params,
                self.s_var_min)
            s_logEI = s_normal_logEI(
                direction * s_thresh,
                direction * (s_mean_x + self.s_emp_mean),
                s_var_x,
                quad_approx=quad_approx)
            self._logEI_cache[(direction, quad_approx)] = theano.function(
                [s_x, s_thresh, self.s_params],
                s_logEI,
                allow_input_downcast=True)
        return self._logEI_cache[(direction, quad_approx)]

    def logEI(self, x, thresh, direction=1, quad_approx=False):
        logEI_fn = self.logEI_fn(direction, quad_approx)
        logEIs = [logEI_fn(x, thresh, p) for p in self._params_list]
        weights = self._params_weights
        rval = np.dot(weights, logEIs)
        return np.atleast_1d(rval)


class GPR_ML2(GPR_Base):
    """
    Fit by maximum marginal likelihood of kernel hyperparameters

    """

    def __init__(self, *args, **kwargs):
        GPR_Base.__init__(self, *args, **kwargs)

        nll, params, params0, bounds, K = self.kernel.s_nll_params(
            self.s_X, self.s_y,
            params=self.s_params,
            var_y=self.s_var_y,
            prior_var=self.s_emp_var, ret_K=True)

        cost = nll - self.kernel.s_logprior(params)
        assert nll.ndim == 0, nll.type

        self._K = theano.function([params], K)
        self._fit_f_df = theano.function([params],
                                         [cost, TT.grad(cost, params)])
        self._params0 = params0
        self._bounds = bounds

    def _fit_params0(self):
        new_x0 = self._params0
        nll_pp = []
        for ii in range(12):
            try:
                f, df = self._fit_f_df(new_x0)
                # -- don't start where the function is too steep
                if np.sqrt(np.dot(df, df)) > 10000:
                    f = np.inf
            except np.linalg.LinAlgError:
                f = np.inf
            # -- ii is in list to break ties, which
            #    happens if there are multiple infs
            nll_pp.append((f, ii, np.array(new_x0)))
            new_x0 = self.kernel.reduce_lenscale(new_x0)

        x0 = sorted(nll_pp)[0][2]
        if np.isinf(sorted(nll_pp)[0][0]):
            raise Exception('fit impossible')
        return x0

    def _fit_ml2(self):
        x0 = self._fit_params0()

        # -- for some reason, the result object returned by minimize
        #    seems occasionally to include a parameter vector (pp)
        #    for which f_df returned np.inf, when there were other non-inf
        #    evaluations (!?)
        #    Therefore, this best_f and best_pp mechanism is used.
        best_f_pp = [np.inf, None]

        def f_df(pp):
            if not np.all(np.isfinite(pp)):
                return np.inf, pp
            try:
                ff, df = self._fit_f_df(pp)
                if ff < best_f_pp[0]:
                    best_f_pp[:] = [ff, pp.copy()]
                return ff, df
            except np.linalg.LinAlgError:
                return np.inf, pp
            except ValueError, exc:
                if 'NaN' in str(exc):
                    return np.inf, pp
                else:
                    raise
        try:
            scipy.optimize.minimize(
                fun=f_df, #self._fit_f_df,
                x0=x0,
                jac=True, # -- means f returns cost and jacobian
                method='SLSQP',
                #method='L-BFGS-B',
                options={} if self.maxiter is None else (
                    {'maxiter': self.maxiter,}),
                bounds=self._bounds,
                )
        except ValueError, e:
            if 'NaN' in str(e):
                print 'WARNING: GPR.fit caught error', e
                print 'WARNING: hopeless fit fail, falling back on params0'
                self._params_list = [self._params0]
            else:
                raise
        return best_f_pp

    def fit_ml2(self, X, y, var_y=0, debug=False, ion=False):
        """
        Fit GPR kernel parameters by minimizing magininal nll.

        Returns: None

        Side effect: chooses optimal kernel parameters.
        """
        self.set_emp_mean(y)
        self.set_emp_var(y, var_y)
        s_X, s_y = self.set_Xy(X, y)
        best_f, best_params = self._fit_ml2()
        self._params_list = [best_params]
        self._params_weights = [1.0]
        return self

    def fit(self, X, y, var_y=0, debug=False, ion=False):
        return self.fit_ml2(X, y, var_y, debug, ion)


class GPR_HMC(GPR_ML2):
    """
    Fit by collecting kernel hyperparameter samples (by HMC).

    """
    def __init__(self, kernel,
                 maxiter=None,
                 prior_var=None,
                 prior_mean=None,
                 hmc_burn_in=0, # -- keep ML first point
                 hmc_draws=200,
                 hmc_keep_step=25):
        GPR_ML2.__init__(self, kernel,
                         maxiter=maxiter,
                         prior_var=prior_var,
                         prior_mean=prior_mean)
        self.positions = theano.shared(np.zeros((1, self.kernel.n_params)),
                                      name='positions')

        nll, s_params, params0, bounds = self.kernel.s_nll_params(
            self.s_X, self.s_y, var_y=self.s_var_y,
            params=self.s_params,
            prior_var=self.s_emp_var,)
        cost = nll - self.kernel.s_logprior(s_params)
        self.nll_cost_fn = theano.function([s_params], [nll, cost])
        self._params0 = params0

        def energy_fn(params_matrix):
            # PRECONDITOIN: params_matrix has SINGLE ROW
            nll, params, params0, bounds = self.kernel.s_nll_params(
                self.s_X, self.s_y, var_y=self.s_var_y,
                prior_var=self.s_emp_var,
                params=params_matrix[0])
            logprior = self.kernel.s_logprior(params_matrix[0])
            energy = nll - logprior
            #energy = theano.printing.Print('energy')(energy)
            return energy.dimshuffle('x')

        print 'creating HMC sampler'
        self.sampler = HMC_sampler.new_from_shared_positions(
            self.positions, energy_fn,
            s_rng=theano.sandbox.rng_mrg.MRG_RandomStreams(1234),
            stepsize_dec=0.95,
            stepsize_inc=1.02,
            stepsize_min=1.0e-8,
            stepsize_max=2.5e-1,
            )
        self._stepsize0 = .001
        self.hmc_burn_in = hmc_burn_in
        self.hmc_draws = hmc_draws
        self.hmc_keep_step = hmc_keep_step

    def fit_hmc(self, X, y, var_y=1e-16, debug=False, ion=False,
               init_params_method='cycle'):

        self.set_emp_mean(y)
        self.set_emp_var(y, var_y)
        self.set_Xy(X, y)

        if init_params_method == 'cycle':
            init_params_method = ['ml2', 'prior'][len(y) % 2]
        if init_params_method == 'ml2':
            _, ml_params = self._fit_ml2()
        elif init_params_method == 'prior':
            ml_params = self._fit_params0()
        else:
            raise NotImplementedError(init_params_method)

        self.sampler.positions.set_value(np.asarray([ml_params]))
        self.sampler.stepsize.set_value(self._stepsize0)

        def get_state(sampler):
            return {
                'positions': sampler.positions.get_value(),
                'stepsize': sampler.stepsize.get_value(),
                'avg_acceptance_rate': sampler.avg_acceptance_rate.get_value(),
            }
        def set_state(sampler, state):
            for k, v in state.items():
                getattr(sampler, k).set_value(v)

        def draw():
            state = get_state(self.sampler)
            while state['stepsize'] > 1e-12:
                try:
                    set_state(self.sampler, state)
                    pos = self.sampler.draw()
                    return pos
                except (ValueError, np.linalg.LinAlgError):
                    print 'shrinking stepsize %f to stabilize sampler' % (
                        self.sampler.stepsize.get_value(),
                    )
                    state['positions'][0] = self.kernel.reduce_lenscale(
                        state['positions'][0])
                    state['stepsize'] /= 2.0
            raise ValueError('hopeless: Nan or inf in K')

        samples = []
        nlls = []
        costs = []
        t0 = time.time()
        for ii in range(self.hmc_burn_in):
            pos = draw()
        for ii in range(self.hmc_draws):
            pos = draw()
            samples.append(pos.ravel().copy())
            if 0:
                nll_ii, cost_ii = self.nll_cost_fn(pos.flatten())
                print 'current position', pos.flatten(),
                print 'accept rate', self.sampler.avg_acceptance_rate.get_value(),
                print 'nll', nll_ii, 'cost', cost_ii
                nlls.append(nll_ii)
                costs.append(cost_ii)
        print 'HMC took', (time.time() - t0)
        samples = np.asarray(samples)
        keep = samples[::self.hmc_keep_step]
        if keep.size == 0:
            raise NotImplementedError()

        if debug:
            import matplotlib.pyplot as plt
            if ion:
                plt.figure(2)
            if self.kernel.n_params == 1:
                plt.subplot(211)
                plt.cla()
                plt.hist(np.asarray(samples).flatten())
                plt.title('nlls observed during sampling')
                plt.subplot(212)
                plt.cla()
                plt.scatter(samples, nlls, label='nll', c='b')
                plt.scatter(samples, costs, label='cost', c='g')
                plt.title('nlls vs. alpha')
                plt.legend()
            if self.kernel.n_params == 2:
                plt.cla()
                plt.scatter(samples[:, 0], samples[:, 1])
                plt.scatter(keep[:, 0], keep[:, 1], s=60)
            if ion:
                plt.draw()
            else:
                plt.show()

        self._params_list = keep
        self._params_weights = np.ones(len(keep)) / len(keep)


    def fit(self, X, y, var_y=0, debug=False, ion=False):
        return self.fit_hmc(X, y, var_y, debug, ion)
