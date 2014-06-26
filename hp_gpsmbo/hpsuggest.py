from itertools import groupby
import numpy as np
import scipy.optimize

from hyperopt.pyll_utils import expr_to_config
from hyperopt import pyll, STATUS_OK
from hyperopt.algobase import SuggestAlgo

from . import kernels


def loss_variances(trials):
    return [r.get('loss_variance', 0)
            for r in trials.results if r['status'] == STATUS_OK]


class SuggestBest(SuggestAlgo):
    def __init__(self, domain, trials, seed, best_pt):
        SuggestAlgo.__init__(self, domain, trials, seed)
        self.best_pt = best_pt

    def on_node_hyperparameter(self, memo, node, label):
        if label in self.best_pt:
            rval = [self.best_pt[label]]
        else:
            rval = []
        return rval


class ParamHelper(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, name):
        node = self.config[name]['node']
        conditional = self.config[name]['conditions'] != set([()])
        rval = getattr(self, node.name)(node, conditional)
        return rval

    def randint(self, node, conditional):
        upper = int(node.arg['upper'].obj)
        def val_fn(feat):
            rval = np.asarray(feat).astype('int')
            if not np.allclose(rval, feat):
                print 'WARNING: optimizer gave randint val_fn a float'
            return rval

        if upper == 2:
            return {
                'feature_bounds': (0, 1),
                'kernel': kernels.Choice2(0.7, 1e-2, 2.0, conditional),
                'ndim': 1,
                'continuous': False,
                'ordinal': False,
                'feature_fn': np.asarray,
                'val_fn': val_fn,
            }
        else:
            return {
                'feature_bounds': (0, upper),
                'kernel': kernels.ChoiceN(upper, conditional),
                'ndim': 1,
                'continuous': False,
                'ordinal': False,
                'feature_fn': np.asarray,
                'val_fn': val_fn,
            }

    def categorical(self, node, conditional):
        # TODO: bias the choice somehow?
        return self.randint(node, conditional)

    def uniform(self, node, conditional, continuous=True, q=None):
        low = float(node.arg['low'].obj)
        high = float(node.arg['high'].obj)
        def val_fn(feat):
            rval = feat * (high - low) + low
            if q is not None:
                rval = np.round(rval / q) * q
            return rval
        return {
            'feature_bounds': (0, 1),
            'kernel': kernels.SqExp(0.7, 1e-6, 1.5, conditional),
            'ndim': 1,
            'continuous': continuous,
            'ordinal': q is not None,
            'feature_fn': (lambda val: (np.asarray(val) - low) / (high - low)),
            'val_fn': val_fn,
        }

    def quniform(self, node, conditional):
        q = float(node.arg['q'].obj)
        return self.uniform(node, conditional, continuous=False, q=q)

    def loguniform(self, node, conditional, continuous=True, q=None):
        # -- log-scaling has been handled by feature code
        #val = np.exp(featureval) - self.logquantized_feature_epsilon
        low = float(node.arg['low'].obj)
        high = float(node.arg['high'].obj)
        def val_fn(feat):
            rval = np.exp(feat * (high - low) + low)
            if q is not None:
                rval = np.round(rval / q) * q
            return rval
        return {
            'feature_bounds': (0, 1),
            'kernel': kernels.SqExp(0.7, 1e-6, 1.5, conditional),
            'ndim': 1,
            'continuous': continuous,
            'ordinal': q is not None,
            'feature_fn': (lambda val: (np.log(val) - low) / (high - low)),
            'val_fn': val_fn,
        }

    def qloguniform(self, node, conditional):
        q = float(node.arg['q'].obj)
        return self.loguniform(node, conditional, continuous=False, q=q)

    def normal(self, node, conditional, continuous=True, q=None):
        sigma = float(node.arg['sigma'].obj)
        mu = float(node.arg['mu'].obj)
        def val_fn(feat):
            rval = feat * sigma + mu
            if q is not None:
                rval = np.round(rval / q) * q
            return rval
        return {
            'feature_bounds': (-10, 10),
            'kernel': kernels.SqExp(0.7, 1e-6, 1.5, conditional),
            'ndim': 1,
            'continuous': continuous,
            'ordinal': q is not None,
            'feature_fn': (lambda val: (np.asarray(val) - mu) / sigma),
            'val_fn': val_fn,
        }

    def qnormal(self, node, conditional):
        q = float(node.arg['q'].obj)
        return self.normal(node, conditional, continuous=False, q=q)

    def lognormal(self, node, conditional, continuous=True, q=None):
        sigma = float(node.arg['sigma'].obj)
        mu = float(node.arg['mu'].obj)
        def val_fn(feat):
            rval = np.exp(feat * sigma + mu)
            if q is not None:
                rval = np.round(rval / q) * q
            return rval
        return {
            'feature_bounds': (-10, 10),
            'kernel': kernels.SqExp(0.7, 1e-6, 1.5, conditional),
            'ndim': 1,
            'continuous': continuous,
            'ordinal': q is not None,
            'feature_fn': (lambda val: (np.log(val) - mu) / sigma),
            'val_fn': val_fn,
        }

    def qlognormal(self, node, conditional):
        q = float(node.arg['q'].obj)
        return self.normal(node, conditional, continuous=False, q=q)


class DomainGP(object):
    logquantized_feature_epsilon = 1e-3

    def __init__(self, domain, GPR=None):
        self.domain = domain

        # -- hps: list of hyperparameter names
        self.hps = list(sorted(domain.params.keys()))

        # -- config: type and dependency information keyed by hp name
        self.config = {}
        expr_to_config(domain.expr, None, self.config)

        if GPR is None:
            GPR = self.GPR # -- class variable

        kerns, self.hp_slices, self.x_bounds = self.init_param_helpers()
        self.gpr = GPR(kernels.product(kerns, self.hp_slices))
        #kern = self.compress_product(kerns, slices)
        #self.gpr = GPR(kern)

    def init_param_helpers(self):
        # -- called early in constructor before most attributes have been set
        kerns = []
        slices = []
        x_bounds = []
        param_helper = ParamHelper(self.config)
        self.param_helpers = {}
        ndim_offset = 0
        for hpname in self.hps:
            ph = self.param_helpers[hpname] = param_helper(hpname)

            kerns.append(ph['kernel'])

            # slices are for index into featurevec
            ph['feature_slice'] = slice(ndim_offset, ndim_offset + ph['ndim'])
            slices.append(ph['feature_slice'])
            ndim_offset += ph['ndim']

            x_bounds.append(ph['feature_bounds'])

        return kerns, slices, np.asarray(x_bounds)

    def draw_n_feature_vecs(self, N, rng):
        fake_ids = range(N)
        idxs, vals = pyll.rec_eval(
            self.domain.s_idxs_vals,
            memo={
                self.domain.s_new_ids: fake_ids,
                self.domain.s_rng: rng,
            })
        return self.features_from_idxs_vals(fake_ids, idxs, vals)

    def features_from_idxs_vals(self, ids, idxs, vals):
        columns = []
        if not np.allclose(ids, np.arange(len(ids))):
            # -- indexing below is a little more complicated, due
            #    to another step of indirection
            raise NotImplementedError('non-contiguous target ids')
        for hpname in self.hps:
            cX = self.param_helpers[hpname]['feature_fn'](vals[hpname])
            if cX.ndim < 2:
                cX.shape = (len(cX), 1)
            assert cX.ndim == 2
            assert cX.shape[1] == self.param_helpers[hpname]['ndim']
            cc = np.empty((len(ids), cX.shape[1])) + np.nan
            cc[idxs[hpname]] = cX
            columns.append(cc)
        return np.hstack(columns)

    def best_pt_from_featurevec(self, featurevec):
        best_pt = {}
        for hpname in self.hps:
            ph = self.param_helpers[hpname]
            feat = featurevec[ph['feature_slice']]
            if not np.isnan(np.sum(feat)):
                assert len(feat) == 1
                best_pt[hpname] = ph['val_fn'](feat[0])
        return best_pt

    def _X_y_var_y(self, trials, failure_loss=None):
        all_tids = trials.tids
        all_idxs, all_vals = trials.idxs_vals
        X = self.features_from_idxs_vals(all_tids, all_idxs, all_vals)
        def loss(tr):
            if tr['result']['status'] == 'ok':
                return (
                    float(tr['result']['loss']),
                    float(tr['result'].get('loss_variance', 0)))
            else: # TODO in-fill prediction for in-prog jobs?
                return float(failure_loss), 0
        y, var_y = zip(*map(loss, trials.trials))
        #y = trials.losses()
        #var_y = loss_variances(trials)
        assert len(y) == len(X) == len(var_y)
        return X, y, var_y

    def fit_gpr(self, X, y, var_y, method='ml2'):
        assert X.shape[1] == len(self.hps)
        if method == 'ml2':
            self.gpr.fit_ml2(X, y, var_y=var_y)
        elif method == 'hmc':
            self.gpr.fit_hmc(X, y, var_y=var_y)
        else:
            raise NotImplementedError(method)

    def optimize_over_X_finetune(self, vec):
        vec_is_nan = np.isnan(vec)

        vec0 = vec.copy()
        vec0[vec_is_nan] = 0

        to_opt = np.ones_like(vec)
        to_opt[vec_is_nan] = 0
        for kslice, hpname in zip(self.hp_slices, self.hps):
            ph = self.param_helpers[hpname]
            if not (ph['continuous'] or ph['ordinal']):
                to_opt[kslice] = 0
        q_filter = np.ones_like(vec)

        def f_df(_x):
            x = np.clip(_x, self.x_bounds[:, 0], self.x_bounds[:, 1])
            if not np.allclose(x, _x):
                print 'x clipped', abs(x - _x)
            x[vec_is_nan] = np.nan
            f, df = self.crit_deriv(np.atleast_2d(x))
            assert len(f) == len(df) == 1
            f = f[0]
            df = df[0]
            assert len(self.hps) == len(self.hp_slices)
            #print 'OPTIMIZE_IN_X: f_df', f, df

            # -- don't fine-tune the discrete variables
            #    TODO: don't even compute the gradient in the first place
            #for ii, (kslice, hpname) in enumerate(zip(self.hp_slices, self.hps)):
            #    ph = self.param_helpers[hpname]
            #    print '    %40s\t%.3f\t%20s\t%.3f\t%8s\t%8s' % (
            #        hpname, _x[ii], kslice, df[ii], ph['continuous'], ph['q'])

            assert np.all(np.isfinite(df))
            mask = to_opt * q_filter
            df[mask == 0] = 0
            assert np.all(np.isfinite(df))
            assert np.all(np.isfinite(f))
            return f, df

        #print 'OPTIMIZE_IN_X start', vec0
        print 'Info: optimizing', (to_opt * q_filter).sum(), 'vars'
        res = scipy.optimize.minimize(
            fun=f_df,
            x0=vec0,
            jac=True, # -- means f returns cost and jacobian
            method='L-BFGS-B',
            #method='SLSQP',
            tol=1e-10, # XXX delete this after validating file
            #options={} if self.maxiter is None else (
                #{'maxiter': self.maxiter,}),
            bounds=self.x_bounds,
            )
        #print 'OPTIMIZE_IN_X done', res
        res.x = np.clip(res.x, self.x_bounds[:, 0], self.x_bounds[:, 1])
        assert np.all(np.isfinite(res.x))

        for kslice, hpname in zip(self.hp_slices, self.hps):
            ph = self.param_helpers[hpname]
            if ph['ordinal']:
                # -- round quantized variables to nearest valid value
                res.x[kslice] = ph['feature_fn'](ph['val_fn'](res.x[kslice]))
                # -- mask out derivatives from here on
                q_filter[kslice] = 0

        # -- maybe reoptimize with quantized variables frozen
        if (to_opt * q_filter).sum():
            print 'Info: reoptimizing', (to_opt * q_filter).sum(), 'vars'
            res2 = scipy.optimize.minimize(
                fun=f_df,
                x0=res.x,
                jac=True, # -- means f returns cost and jacobian
                method='L-BFGS-B',
                #method='SLSQP',
                tol=1e-10, # XXX delete this after validating file
                #options={} if self.maxiter is None else (
                    #{'maxiter': self.maxiter,}),
                bounds=self.x_bounds,
                )
        else:
            print 'Info: skipping reoptimization step'
            res2 = res
        assert np.all(np.isfinite(res2.x))
        #print 'OPTIMIZE_IN_X done', res
        res2.x = np.clip(res2.x, self.x_bounds[:, 0], self.x_bounds[:, 1])
        res2.x[vec_is_nan] = np.nan
        return res2

    def optimize_over_X(self, n_buckshots, n_finetunes, rng, ret_raw=False,
                       ret_results=False):
        # -- sample a bunch of points
        buckshot = self.draw_n_feature_vecs(n_buckshots, rng)
        buckshot_crit = self.crit(buckshot)
        best_first = np.argsort(buckshot_crit)
        #print 'buckshot stats', buckshot_crit.min(), buckshot_crit.max()

        # -- finetune a few of the best by gradient descent
        results = [
            (buckshot_crit[best_first[0]],
             -1,
             buckshot[best_first[0]].copy(),
             buckshot_crit[best_first[0]],
            )]
        if self._cost_deriv is not None:
            misc_step = int(n_buckshots / (.5 * n_finetunes))
            misc = best_first[n_finetunes::misc_step]
            top_best = best_first[:n_finetunes - len(misc)]
            to_finetune = list(misc) + list(top_best)
            assert len(to_finetune) <= n_finetunes
            for ii in range(n_finetunes):
                vec = buckshot[to_finetune[ii]]
                res = self.optimize_over_X_finetune(vec)
                results.append((res.fun, ii, res.x.copy(),
                                buckshot_crit[to_finetune[ii]]))
            results.sort()
            if results[0][1] == -1:
                print 'Warning: finetuning did no good'
        print 'optimize_X', results[0]
        if ret_results:
            return results
        if ret_raw:
            return results[0][2]
        else:
            # -- return the best one
            best_pt = self.best_pt_from_featurevec(results[0][2])
            return best_pt

# -- flake-8 abhors blank line EOF
