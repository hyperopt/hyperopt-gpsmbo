        if 1:
            keyfunc = lambda nc: nc[1]['node'].name
            hps_by_type = dict()
            idxs_by_type = dict()
            kerns = []
            for distname, labels_hps in groupby(sorted(self.config.items(),
                                                       key=keyfunc),
                                                keyfunc):
                label_list, hp_list = zip(*list(labels_hps))
                hps_by_type[distname] = hp_list
                idxs_by_type[distname] = map(self.hps.index, label_list)
                foo = hps_by_type[distname]
                print distname, len(foo), idxs_by_type[distname]
                kerns.append(ph['kernel'])

            param_helper = ParamHelper(self.config)

            x_bounds = [(None, None)] * len(self.hps)
            ndim_offset = 0
            for hpname in self.hps:
                ph = self.param_helpers[hpname] = param_helper(hpname)

            import sys
            sys.exit()
        else:




class ConvexMixtureKernel(object):
    """

    Attributes:
    
        kernels -
        element_ranges - each kernel looks at these elements (default ALL)
        feature_names - 
        raw_coefs - 
        coefs - 

    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        coefs = self.coefs_f()
        ks = [str(k) for k in self.kernels]
        return 'ConvexMixtureKernel{%s}'%(','.join(['%s*%s'%(str(c),s) for c,s in zip(coefs, ks)]))

    def summary(self):
        import StringIO
        ss = StringIO.StringIO()
        coefs = self.coefs_f()
        print >> ss,  "ConvexMixtureKernel:"
        for c, k,fname in zip(coefs,self.kernels, self.feature_names):
            print >> ss,  "  %f * %s '%s'" %(c, str(k), fname)
        return ss.getvalue()

    @classmethod
    def alloc(cls, kernels, coefs=None, element_ranges=None, feature_names=None):
        if coefs is None:
            raw_coefs = theano.shared(np.zeros(len(kernels)))
            print "HAAACK"
            raw_coefs.get_value(borrow=True)[0] += 1 
        else:
            raise NotImplementedError()
        coefs=TT.nnet.softmax(raw_coefs.dimshuffle('x',0))[0]
        coefs_f = theano.function([], coefs)
        return cls(
                kernels=kernels,
                coefs=coefs,
                coefs_f = coefs_f, #DEBUG
                raw_coefs = raw_coefs,
                element_ranges=element_ranges,
                feature_names = feature_names,
                )

    def params(self):
        rval = [self.raw_coefs]
        for k in self.kernels:
            rval.extend(k.params())
        return rval
    def param_bounds(self):
        rval = [(self.raw_coefs_min, self.raw_coefs_max)]
        for k in self.kernels:
            rval.extend(k.param_bounds())
        return rval

    def K(self, x, y):
        # get the kernel matrix from each sub-kernel
        if self.element_ranges is None:
            Ks = [kernel.K(x,y) for kernel in  self.kernels]
        else:
            assert len(self.element_ranges) == len(self.kernels)
            Ks = [kernel.K(x[:,er[0]:er[1]],y[:,er[0]:er[1]])
                    for (kernel,er) in zip(self.kernels, self.element_ranges)]
        # stack them up
        Kstack = TT.stack(*Ks)
        # multiply by coefs
        # and sum down to one kernel
        K = TT.sum(self.coefs.dimshuffle(0,'x','x') * Kstack,
                axis=0)
        return K



class Exp(SqExp):
    """
    K(x,y) = exp(- ||x-y|| / l)

    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if self.log_lenscale.ndim!=0:
            raise TypeError('log_lenscale must be scalar', self.log_lenscale)

    def __str__(self):
        l = np.exp(self.log_lenscale.value)
        return "ExponentialKernel{l=%s}"%str(l)

    @classmethod
    def alloc(cls, l=1, l_min=1e-4, l_max=1000):
        log_l = np.log(l)
        log_lenscale = theano.shared(log_l)
        if l_min is None:
            log_lenscale_min = None
        else:
            log_lenscale_min = np.log(2*(l_min**2))
        if l_max is None:
            log_lenscale_max = None
        else:
            log_lenscale_max = np.log(2*(l_max**2))
        return cls(log_lenscale=log_lenscale,
                log_lenscale_min=log_lenscale_min,
                log_lenscale_max=log_lenscale_max)

    def params(self):
        return [self.log_lenscale]

    def param_bounds(self):
        return [(self.log_lenscale_min, self.log_lenscale_max)]

    def K(self, x, y):
        l = TT.exp(self.log_lenscale)
        d = ((x**2).sum(axis=1).dimshuffle(0,'x')
                + (y**2).sum(axis=1)
                - 2 * TT.dot(x, y.T))
        K = TT.exp(-TT.sqrt(d)/l)
        return K


class CategoryKernel(object):
    """
    K(x,y) is 1 if x==y else exp(-1/l)

    The idea is that it's like a SquaredExponentialKernel
    where every point is a distance of 1 from every other one, 
    except itself.

    Attributes:
        
        l - 

    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if self.l.ndim!=0:
            raise TypeError('log_denom must be scalar', self.l)
    def lenscale(self, thing=None):
        if thing is None:
            thing = self.l
        return value(thing)
    def __str__(self):
        l = self.lenscale()
        (a,b), = self.param_bounds()
        return "CategoryKernel{l=%s,bounds=(%s,%s)}"%(
                str(l), str(a), str(b))

    @classmethod
    def alloc(cls, l=1.0, l_min=1e-5, l_max=100.):
        l = theano.shared(l)
        return cls(l=l,
                l_min=l_min,
                l_max=l_max,
                )

    def params(self):
        return [self.l]
    def param_bounds(self):
        return [(self.l_min, self.l_max)]

    def K(self, x, y):
        xx = x.reshape((x.shape[0],))
        yy = y.reshape((y.shape[0],))
        xx = xx.dimshuffle(0,'x') # drop cols because there should only be 1
        yy = yy.dimshuffle(0)     # drop cols because there should only be 1
        K = TT.exp(-TT.neq(xx,yy)/self.l)
        return K



class GPR_HMC_for_SGD_EI_OPT(object):
    def __init__(self):
        # ...

        self.s_EI_pts = theano.shared(np.zeros((2, 2)))
        self.s_EI_vals = theano.shared(np.zeros(2))
        self.s_EI_step = theano.tensor.dscalar('EI_step')
        self.s_EI_thresh = theano.shared(0.0)

        s_mean_x, s_var_x, s_x = self.kernel.s_mean_var(
            self.s_X,
            self.s_y,
            self.s_var_y,
            self.s_emp_var,
            self.positions[0],
            self.s_var_min,
            x_new=self.s_EI_pts)
        s_logEI = s_normal_logEI(
            - self.s_EI_thresh,
            - (s_mean_x + self.s_emp_mean),
            s_var_x,
            quad_approx=True)
        print 'compiling update_EI_pts fn'
        self.update_EI_pts = theano.function(
            [self.s_EI_step],
            [],
            updates=[
                (self.s_EI_pts, TT.clip(
                    self.s_EI_pts + self.s_EI_step * TT.grad(s_logEI.sum(),
                                                             self.s_EI_pts),
                    np.asarray(bounds)[:, 0],
                    np.asarray(bounds)[:, 1])),
                (self.s_EI_vals, 0.95 * self.s_EI_vals + .05 * s_logEI),
            ],
            allow_input_downcast=True)

    def fit_and_optimize_EI(self, X, y, var_y, debug, ion,
                           EI_pts):
        print 'setting up'
        self.s_emp_mean.set_value(np.mean(y))
        self.s_emp_var.set_value(max(np.var(y), np.min(var_y)))
        self.s_X.set_value(X)
        self.s_y.set_value(y - self.s_emp_mean.get_value())
        self.s_var_y.set_value(var_y + np.zeros(len(y)))
        self.s_EI_pts.set_value(EI_pts)
        self.s_EI_vals.set_value(np.zeros(len(EI_pts)))
        self.s_EI_thresh.set_value(np.min(y))

        samples = []
        nlls = []
        costs = []
        t0 = time.time()
        hmc_duration = 10.0 # seconds
        print 'running the sampler'
        while time.time() < (t0 + hmc_duration):
            try:
                tt = time.time() - t0
                pos = self.sampler.draw()
                self.update_EI_pts(.003 * min(1, 1. / (.1 + tt)))
                samples.append(pos.flatten())
                if debug:
                    nll_ii, cost_ii = self.nll_fn(pos.flatten())
                    #print s_EI_vals.get_value()
                    print 'best_EI', self.s_EI_vals.get_value().min()
                    print 'current position', pos.flatten(),
                    print 'accept rate', self.sampler.avg_acceptance_rate.get_value(),
                    print 'nll', nll_ii, 'cost', cost_ii
                    nlls.append(nll_ii)
                    costs.append(cost_ii)
            except ValueError, e:
                # -- XXX should not happen
                print 'ERROR: HMC crashed after %i draws' % len(samples)
                raise
                break

            except np.linalg.LinAlgError, e:
                print 'ERROR: HMC singular matrix after %i draws' % len(samples)
                break
        samples = np.asarray(samples)
        print 'hmc drew', len(samples)
        step = max(1, len(samples) // 10)
        keep = samples[::step]
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
            if ion:
                plt.draw()
            else:
                plt.show()

        class Res(object):
            pass

        rval = Res()
        best_idx = np.argmax(self.s_EI_vals.get_value())
        rval.x = self.s_EI_pts.get_value()[best_idx]
        rval.fun = self.s_EI_vals.get_value()[best_idx]

        self._params_list = keep
        self._params_weights = np.ones(len(keep)) / len(keep)
        return rval

class LengthscaleBounds(object):
    def __init__(self, config):
        self.config = config

    def LU0(self, name):
        node = self.config[name]['node']
        return getattr(self, node.name)(node)

    def randint(self, node):
        return 0.001, 2.0, 1.0

    def categorical(self, node):
        return 0.001, 2.0, 1.0

    def uniform(self, node):
        low = float(node.arg['low'].obj)
        high = float(node.arg['high'].obj)
        thetaL = (high - low) / 20.0
        thetaU = (high - low) * 2.
        return thetaL, thetaU, (high - low) / 2

    def quniform(self, node):
        # -- quantization is irrelevant
        return self.uniform(node)

    def loguniform(self, node):
        # -- log-scaling has been handled by feature code
        return self.uniform(node)

    def qloguniform(self, node):
        # -- log-scaling has been handled by feature code
        #    quantization is irrelevant
        return self.uniform(node)

    def normal(self, node):
        sigma = float(node.arg['sigma'].obj)
        thetaL = sigma / 20.0
        thetaU = sigma * 2.
        return thetaL, thetaU, sigma

    def qnormal(self, node):
        # -- quantization is irrelevant
        return self.normal(node)

    def lognormal(self, node):
        # -- log-scaling has been handled by feature code
        return self.normal(node)

    def qlognormal(self, node):
        # -- log-scaling has been handled by feature code
        #    quantization is irrelevant
        return self.normal(node)


import numpy as np

import theano
import theano.tensor as TT
from hyperopt import rand

from . import gpr_math
from .hpsuggest import SuggestBest, DomainGP


class DomainGP_LUCB(DomainGP):
    _optimism = 1.0
    _sigmoid_bias = -0.0

    def init_cost_fns(self):
        try:
            self._cost_fn
        except AttributeError:
            s_optimism = TT.dscalar('optimism')
            s_ubound = TT.dscalar('ubound')
            s_lbound = TT.dscalar('lbound')

            # s_mean_x means "symbolic mean of x"
            s_mean_x, s_var_x, s_x, K_new = self.gpr.kernel.s_mean_var(
                self.gpr.s_X,
                self.gpr.s_y,
                self.gpr.s_var_y,
                self.gpr.s_emp_var,
                self.gpr.s_params,
                self.gpr.s_var_min,
                return_K_new=True)

            corrected_mean = s_mean_x + self.gpr.s_emp_mean
            # -- good vars are for maximizing,
            #    in keeping with EI being about improving *over* thresh
            good_max = -s_lbound
            good_best_seen = -s_ubound
            good_mean = -corrected_mean
            good_var = s_var_x

            #scalar = 1.0 + s_optimism

            #z = (corrected_mean - s_lbound) / TT.sqrt(s_var_x)
            #acq = gpr_math.s_normal_EBI(
            #    0,
            #    -(s_lbound - corrected_mean),
            #    0,
            #    s_var_x)
            #s_cost = -(tradeoff * acq) - (1 - tradeoff) * corrected_mean * TT.erf(-z)


            if 1: # -- use LUCB
                #good_var = good_var * s_optimism ** 2
                lost_mass = gpr_math.s_normal_cdf(-good_max,
                                                  -good_mean,
                                                  good_var)
                gap = good_max - good_best_seen
                drop = s_optimism * gap
                #coef = 1. / s_optimism
                EBI_ceil = TT.minimum(
                    good_mean,
                    good_max - drop)
                    #coef * good_min + (1 - coef) * good_max)
                #max_ceil = good_max - (s_optimism - 1) * gap
                acq = (
                    EBI_ceil
                    + (
                        gpr_math.s_normal_EBI(
                            EBI_ceil, good_max,
                            EBI_ceil, good_var) / (1 - lost_mass)))
                        #+ (good_max - good_mean) * ))

            elif 1: # -- use bounded EI type thing
                ebi_term = gpr_math.s_normal_EBI(
                    good_min, good_max,
                    good_mean, good_var)
                mass_above_good_max = gpr_math.s_normal_cdf(
                    -good_max, -good_mean, good_var)
                acq = ebi_term + (good_max - good_min) * mass_above_good_max

            s_cost = -acq
            try:
                s_gx = TT.grad(s_cost.sum(), s_x)
                self._cost_deriv = theano.function(
                    [s_x, self.gpr.s_params, s_optimism, s_ubound, s_lbound],
                    [s_cost, s_gx],
                    on_unused_input='warn')
            except theano.gradient.DisconnectedInputError:
                self._cost_deriv = None
            self._cost_fn = theano.function(
                [s_x, self.gpr.s_params, s_optimism, s_ubound, s_lbound],
                s_cost,
                on_unused_input='warn')
            self._K_new = theano.function(
                [s_x, self.gpr.s_params], K_new)

    def crit(self, X):
        self.init_cost_fns()
        if len(self.gpr._params_list) > 1:
            raise NotImplementedError()
        pp, = self.gpr._params_list
        return self._cost_fn(X, pp, self._optimism, self._ubound, self._lbound)

    def crit_deriv(self, X):
        if self._cost_deriv is None:
            raise NotImplementedError()
        self.init_cost_fns()
        if len(self.gpr._params_list) > 1:
            raise NotImplementedError()
        pp, = self.gpr._params_list
        return self._cost_deriv(X, pp, self._optimism, self._ubound, self._lbound)

    def optimize_over_X(self, n_buckshots, n_finetunes, rng):
        while True:
            rval_raw = DomainGP.optimize_over_X(self,
                                                n_buckshots,
                                                n_finetunes,
                                                rng,
                                                ret_raw=True)
            Ks = self._K_new(np.atleast_2d(rval_raw), self.gpr._params_list[0])
            # XXX: todo, if other non-redundant local optima were discoverd by
            # the fine-tuning process then it might better to take them,
            # before distorting the utility landscape with this "optimism"
            # multiplier. I wonder if one is more "right" to do than the other
            if (Ks.max() > (1 - 1e-5)):
                if self._optimism < 1e8:
                    self._optimism *= 2
                    print 'LUCB raising optimism to', self._optimism
                else:
                    print "LUCB error finding new point!"
            else:
                break
        best_pt = self.best_pt_from_featurevec(rval_raw)
        return best_pt


_suggest_domain_cache = {}
def suggest(new_ids, domain, trials, seed,
            warmup_cutoff=1,
            n_buckshots=10000,
            n_finetunes=50,
            best_possible=-np.inf,
            #best_headroom=1.0,
            stop_at=None,
            plot_contours=None,
            ):
    """
    Parameters
    ----------

    """
    if stop_at is not None and stop_at < best_possible:
        raise ValueError(
            ('If stop_at is specified'
             ', it (%f) must be greater than best_possible (%f)') % (
                 stop_at, best_possible))

    if len(trials.trials) <= warmup_cutoff:
        return rand.suggest(new_ids, domain, trials, seed)

    try:
        dgp = _suggest_domain_cache[domain]
    except KeyError:
        dgp = _suggest_domain_cache[domain] = DomainGP_LUCB(domain)

    if stop_at is not None and min(trials.losses()) < stop_at:
        return []

    X, y, var_y = dgp._X_y_var_y(trials)
    dgp.fit_gpr(X, y, var_y)
    dgp._optimism = 1.0 #0.5 * dgp._optimism
    dgp._ubound = np.min(y)
    dgp._lbound = best_possible

    #yy = y + np.sqrt(np.maximum(var_y, dgp.gpr.s_var_min.eval()))
    #dgp._ubound = np.min(yy)
        #max(opt_lbound,  - best_headroom)
    #print 'LUCB interval:', dgp._lbound, dgp._ubound

    print 'LUCB: Best after %i trials: %f' % ( len(y), np.min(y))
    #dgp.gpr._params_list[0][:] = 0
    rng = np.random.RandomState(seed)
    best_pt = dgp.optimize_over_X(
        n_buckshots=n_buckshots,
        n_finetunes=n_finetunes,
        rng=rng,
        )
    if plot_contours:
        plot_contours(dgp, 2, dgp._lbound, best_pt)
    new_id, = new_ids
    #print 'LUCB: Best pt', best_pt
    return SuggestBest(domain, trials, seed, best_pt)(new_id)

