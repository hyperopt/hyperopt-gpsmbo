import numpy as np
import theano.tensor
from hyperopt import rand

from .hpsuggest import SuggestBest, DomainGP
from .gpr import GPR_ML2


class DomainGP_UCB(DomainGP):
    GPR = GPR_ML2

    def init_cost_fns(self):
        try:
            self._cost_fn
        except AttributeError:
            s_ucb_z = theano.tensor.dscalar('ucb_z')

            s_mean_x, s_var_x, s_x, K_new = self.gpr.kernel.s_mean_var(
                self.gpr.s_X,
                self.gpr.s_y,
                self.gpr.s_var_y,
                self.gpr.s_emp_var,
                self.gpr.s_params,
                self.gpr.s_var_min,
                return_K_new=True)
            s_cost = s_mean_x - theano.tensor.sqrt(s_var_x) * s_ucb_z

            s_gx = theano.tensor.grad(s_cost.sum(), s_x)
            self._cost_fn = theano.function(
                [s_x, s_ucb_z, self.gpr.s_params], s_cost)
            self._cost_deriv = theano.function(
                [s_x, s_ucb_z, self.gpr.s_params], [s_cost, s_gx])
            self._K_new = theano.function(
                [s_x, self.gpr.s_params], K_new)


    def crit(self, X):
        self.init_cost_fns()
        if len(self.gpr._params_list) > 1:
            raise NotImplementedError()
        pp, = self.gpr._params_list
        return self._cost_fn(X, self._ucb_z, pp)

    def crit_deriv(self, X):
        self.init_cost_fns()
        if len(self.gpr._params_list) > 1:
            raise NotImplementedError()
        pp, = self.gpr._params_list
        return self._cost_deriv(X, self._ucb_z, pp)

    def optimize_over_X(self, n_buckshots, n_finetunes, rng):
        best_pt = None
        while True:
            results = DomainGP.optimize_over_X(self, n_buckshots,
                                                n_finetunes, rng, ret_results=True)
            Ks = self._K_new(np.asarray([rr[2] for rr in results]),
                             self.gpr._params_list[0]).T
            #order = rng.permutation(len(results))
            order = range(len(results))
            assert len(Ks) == len(results)
            for ii in order:
            #for Ki, rr in zip(Ks, results):
                Ki = Ks[ii]
                rr = results[ii]
                if Ki.max() >  self._K_thresh:
                    #print 'UCB: skipping pt wit h K', Ki.max()
                    continue
                else:
                    #print 'UCB: picking pt wit h K', Ki.max()
                    best_pt = self.best_pt_from_featurevec(rr[2])
                    break
            if best_pt is None:
                self._ucb_z *= 2 + .1
                print 'UCB: raising ucb_z to', self._ucb_z
            else:
                break
        #best_pt = self.best_pt_from_featurevec(rval_raw)
        return best_pt


_suggest_domain_cache = {}
def suggest(new_ids, domain, trials, seed,
            warmup_cutoff=15,
            n_buckshots=10000,
            n_finetunes=50,
            stop_at=None,
            plot_contours=None,
            ):
    """
    Parameters
    ----------

    """
    if len(trials.trials) <= warmup_cutoff:
        return rand.suggest(new_ids, domain, trials, seed)

    # XXX would like to cache on domain, but
    #     fmin(fn, space) always rebuilds a new domain for given fn and space
    key = domain.expr
    try:
        dgp = _suggest_domain_cache[key]
    except KeyError:
        dgp = _suggest_domain_cache[key] = DomainGP_UCB(domain)

    if stop_at is not None and min(trials.losses()) < stop_at:
        return []

    X, y, var_y = dgp._X_y_var_y(trials)
    dgp.fit_gpr(X, y, var_y)
    print 'Fit ->', dgp.gpr._params_list[0]
    dgp._ucb_z = 0.2
    # XXX: radius should depend on dimensionality?
    #     1e-8 worked for branin in case current one doesn't
    dgp._K_thresh =  (1 - 1e-5) # / (1000 + len(y) ** 2))

    print 'UCB: Best after %i trials: %f' % ( len(y), np.min(y))
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
    #print 'REI: Best pt', best_pt
    return SuggestBest(domain, trials, seed, best_pt)(new_id)
# --eof
