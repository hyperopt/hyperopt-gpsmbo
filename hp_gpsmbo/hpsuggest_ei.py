import time
import numpy as np
import theano.tensor

from hyperopt import rand

from .hpsuggest import SuggestBest, DomainGP
from . import gpr_math
from . import op_Kcond
from .gpr import GPR_HMC

class DomainGP_EI(DomainGP):
    _EI_thresh_increment = 0.1
    _min_thresh_inc = 0
    GPR = GPR_HMC

    def init_fns(self):
        try:
            self._cost_deriv
        except AttributeError:
            s_thresh = theano.tensor.dscalar('thresh')
            s_reuse_cholesky = theano.tensor.iscalar('reuse_cholesky')
            s_reuse_cholesky_idx = theano.tensor.iscalar('reuse_cholesky_idx')

            s_mean_x, s_var_x, s_x, K_new = self.gpr.kernel.s_mean_var(
                self.gpr.s_X,
                self.gpr.s_y,
                self.gpr.s_var_y,
                self.gpr.s_emp_var,
                self.gpr.s_params,
                self.gpr.s_var_min,
                return_K_new=True)
            s_logEI = gpr_math.s_normal_logEI(
                -s_thresh,
                -(s_mean_x + self.gpr.s_emp_mean),
                s_var_x,
                quad_approx=True)
            cost = -s_logEI

            assert cost.ndim == 1
            s_gx = theano.tensor.grad(cost.sum(), s_x)

            # -- this hack makes it so that the s_reuse_cholesky
            #    variable is patched in to the graph during optimization
            #    and allows to disable the computation of training
            #    K matrix and it's cholesky factorization
            op_Kcond.use_lazy_cholesky = s_reuse_cholesky
            op_Kcond.use_lazy_cholesky_idx = s_reuse_cholesky_idx
            self._cost_deriv = theano.function(
                [s_x, s_thresh, self.gpr.s_params,
                 s_reuse_cholesky, s_reuse_cholesky_idx],
                [cost, s_gx],
                on_unused_input='ignore',
                allow_input_downcast=True,
                profile=0)
            op_Kcond.use_lazy_cholesky = None
            op_Kcond.use_lazy_cholesky_idx = None

            if 1:
                # /begin hack sanity checking
                #import pdb; pdb.set_trace()
                n_cholesky = 0
                n_lazy_cholesky = 0
                for node in self._cost_deriv.maker.fgraph.toposort():
                    #print node
                    if isinstance(node.op,
                                  theano.sandbox.linalg.ops.Solve):
                        assert node.op.A_structure != 'general'
                    if isinstance(node.op,
                                  theano.sandbox.linalg.ops.Cholesky):
                        n_cholesky += 1
                    if isinstance(node.op, op_Kcond.LazyCholesky):
                        n_lazy_cholesky += 1
                assert n_cholesky == 0
                assert n_lazy_cholesky == 1
                # /end hack sanity checking

            self._cost = theano.function(
                [s_x, s_thresh, self.gpr.s_params],
                cost,
                allow_input_downcast=True)
            self._K_new = theano.function(
                [s_x, self.gpr.s_params], K_new)
        return self._cost_deriv

    def set_thresholds(self, y, var_y, z=1.0, max_ei_thresh=None):
        yy = y - z * np.sqrt(np.maximum(var_y,
                                        max(
                                            self.gpr.s_var_min.eval(),
                                            self._min_thresh_inc ** 2)))
        if max_ei_thresh is not None:
            self._EI_thresh = min(np.min(yy), max_ei_thresh)
        else:
            self._EI_thresh = np.min(yy)

    def crit(self, X):
        self.init_fns()
        #return -self.gpr.logEI(X,
                               #self._EI_thresh,
                               #direction=-1, # below thresh
                               #quad_approx=True)
        gpr = self.gpr
        fs = []
        for pp in gpr._params_list:
            f = self._cost(np.atleast_2d(X),
                               self._EI_thresh,
                               pp)
            fs.append(f)
        mean_f = np.dot(gpr._params_weights, fs)
        return mean_f

    def crit_deriv(self, X):
        self.init_fns()
        gpr = self.gpr
        fs = []
        dfs = []
        for ii, pp in enumerate(gpr._params_list):
            #print 'pp', pp, 'x', X
            f, df = self._cost_deriv(np.atleast_2d(X),
                                     self._EI_thresh,
                                     pp,
                                     self._cost_deriv_reuse_cholesky,
                                     ii)
            assert f.shape == (1,), (f.shape, X.shape)
            fs.append(f[0])
            dfs.append(df.flatten())
        self._cost_deriv_reuse_cholesky = 1
        mean_f = np.dot(gpr._params_weights, fs)
        #import pdb; pdb.set_trace()
        mean_df = np.dot(gpr._params_weights, np.asarray(dfs))
        return [mean_f], [mean_df]

    def optimize_over_X(self, n_buckshots, n_finetunes, rng):
        while True:
            rval_raw = DomainGP.optimize_over_X(self,
                                                n_buckshots,
                                                n_finetunes,
                                                rng,
                                                ret_raw=True)
            if len(self.gpr._params_list) == 1:
                Ks = self._K_new(np.atleast_2d(rval_raw),
                                 self.gpr._params_list[0])
                if (Ks.max() > (1 - 1e-6)):
                    # -- promote exploration with a more aggressive threshold
                    self._EI_thresh -= self._EI_thresh_increment
                    print 'lowering EI thresh to', self._EI_thresh
                else:
                    break
            else:
                break
        best_pt = self.best_pt_from_featurevec(rval_raw)
        return best_pt


_suggest_domain_cache = {}
def suggest(new_ids, domain, trials, seed,
            warmup_cutoff=15, # -- enough for mean & var stats
            n_buckshots=10000,
            n_finetunes=50,
            stop_at=None,
            plot_contours=None,
            gp_fit_method='ml2',
            failure_loss=None,
            max_ei_thresh=None,
            ):
    """
    Parameters
    ----------

    """
    # XXX would like to cache on domain, but
    #     fmin(fn, space) always rebuilds a new domain for given fn and space
    key = domain.expr
    try:
        dgp = _suggest_domain_cache[key]
    except KeyError:
        print  'CREATING GP_EI for', domain
        dgp = _suggest_domain_cache[key] = DomainGP_EI(domain)
    if len(trials.trials):
        X, y, var_y = dgp._X_y_var_y(trials, failure_loss=failure_loss)

    if len(trials.trials) <= warmup_cutoff:
        if len(trials.trials):
            dgp.gpr.prior_mean = np.mean(y)
            dgp.gpr.prior_var = np.var(y)
        return rand.suggest(new_ids, domain, trials, seed)

    if stop_at is not None and min(trials.losses()) < stop_at:
        return []

    dgp.fit_gpr(X, y, var_y, method=gp_fit_method)
    dgp.set_thresholds(y, var_y, max_ei_thresh=max_ei_thresh)
    dgp._cost_deriv_reuse_cholesky = 0

    print 'EI: Best after %i trials: %f' % ( len(y), np.min(y))
    #dgp.gpr._params_list[0][:] = 0
    rng = np.random.RandomState(seed)
    t0 = time.time()
    best_pt = dgp.optimize_over_X(
        n_buckshots=n_buckshots,
        n_finetunes=n_finetunes,
        rng=rng,
        )
    t1 = time.time()
    print 'optimizing surrogate took', (t1 - t0)
    if plot_contours:
        plot_contours(dgp, 2, dgp._lbound, best_pt)
    new_id, = new_ids
    #print 'REI: Best pt', best_pt
    return SuggestBest(domain, trials, seed, best_pt)(new_id)

# --eof
