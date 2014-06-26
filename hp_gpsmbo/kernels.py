import numpy as np
import theano
import theano.tensor as TT

from .op_Kcond import zero_diag, isnan as s_isnan
from .kernels_base import Kernel
from .kernels_base import euclidean_sq_distances

from .prodkernels import SqExpProd


def check_K(K, tag=None):
    return K
    import scipy.linalg
    def check_pd(op, xin):
        try:
            scipy.linalg.cholesky(xin + 1e-12 * np.eye(xin.shape[0]))
        except:
            print 'tag', tag
            theano.printing.debugprint(K)
            raise
    return theano.printing.Print('check_K', global_fn=check_pd)(K)


def check_finite(K, tag=None):
    return K
    def check(op, xin):
        try:
            assert np.all(np.isfinite(xin))
        except:
            print 'tag', tag
            theano.printing.debugprint(K)
            raise
    return theano.printing.Print('check_finite', global_fn=check)(K)


class ChoiceN(Kernel):
    def __init__(self, upper, conditional, seed=1):
        # N.B. seed should not need to be changed

        # -- XXX only need upper-triangle worth of values
        #        but need Theano triangle-packing op (already exists??)
        #self.n_params = upper * (upper - 1) / 2
        self.n_idxs = (upper + 1) if conditional else upper
        self.n_params = self.n_idxs ** 2

        self.seed = seed
        self.conditional = conditional

    def prodkey(self):
        return id(self) # -- choices are not mergeable

    def reduce_lenscale(self, params):
        # No-op
        return params

    def s_logprior(self, params, strength=10.0):
        P_shaped = params.reshape((self.n_idxs, self.n_idxs))
        P_norms = TT.sqrt((P_shaped ** 2).sum(axis=1))
        return strength * ((P_norms - 1) ** 2).sum()

    def unit(self, params):
        P_shaped = params.reshape((self.n_idxs, self.n_idxs))
        P_norms = TT.sqrt((P_shaped ** 2).sum(axis=1))
        P_unit = P_shaped / P_norms[:, None]
        return P_unit

    def opt_logK(self, x, params):
        if self.conditional:
            s_x = TT.switch(TT.isnan(x), self.n_idxs - 1, x)
        else:
            s_x = x
        #s_x = theano.printing.Print('x')(s_x)
        lbound = 1e-5
        ubound = 1.0
        params0 = np.random.RandomState(self.seed).uniform(
            low=lbound,
            high=ubound,
            size=(self.n_idxs, self.n_idxs))
        P_unit = self.unit(params)
        idxs = s_x.flatten().astype('int32')
        #def wtf(node, val):
        #    print 'IDXS', val
        #    print 'SELF', self.n_idxs, self.conditional
        #    return val

        #idxs = theano.printing.Print('idxs', global_fn=wtf)(idxs)
        K = TT.dot(P_unit[idxs], P_unit[idxs].T)
        #K = K + 1e-12 * TT.eye(x.shape[0])
        bounds = [(lbound, ubound)] * self.n_params
        return TT.log(K), list(params0.flatten()), bounds

    def predict_logK(self, x, z, params):
        if self.conditional:
            s_x = TT.switch(TT.isnan(x), self.n_idxs - 1, x)
            s_z = TT.switch(TT.isnan(z), self.n_idxs - 1, z)
        else:
            s_x = x
            s_z = z
        P_unit = self.unit(params)
        K = TT.dot(P_unit[s_x.flatten().astype('int32')],
                   P_unit[s_x.flatten().astype('int32')].T)
        #K_reg = K + 1e-12 * TT.eye(x.shape[0])
        K_new = TT.dot(P_unit[s_x.flatten().astype('int32')],
                       P_unit[s_z.flatten().astype('int32')].T)
        return TT.log(K), TT.log(K_new)


class StationaryBase(Kernel):
    """

    K(x,y) = exp(- ||x-y||^2 / (2 l^2))

    N.B. the kernel is parameterized by quantity

        alpha = log( 2 * l^2)

    So that 

        K(x, y) = exp(- ||x - y|| ** 2 / exp(alpha))
        l = sqrt(exp(alpha) / 2)


    """

    @staticmethod
    def _alpha_from_l(l):
        return np.log(2.0 * l ** 2)

    @staticmethod
    def _l_from_alpha(alpha):
        return np.sqrt(np.exp(alpha) / 2.)

    def __init__(self, lenscale, lenscale_min, lenscale_max, conditional):
        self._lenscale0 = lenscale
        self._lenscale_min = lenscale_min
        self._lenscale_max = lenscale_max
        self._conditional = conditional
        self._n_warp_segments = 0
        if conditional:
            self.n_params = 3 + self._n_warp_segments
        else:
            self.n_params = 1 + self._n_warp_segments

    def prodkey(self):
        # -- unique identifier of mergeable product sets
        return (type(self),
                self._conditional,
                self._n_warp_segments)

    def props(self):
        return (
            self._lenscale0,
            self._lenscale_min,
            self._lenscale_max,
            self._conditional,
            self._n_warp_segments,
            )

    def __eq__(self, other):
        return type(self) == type(other) and self.props() == other.props()

    def __hash__(self):
        return hash((type(self), self.props()))

    def reduce_lenscale(self, params):
        new_alpha = params[0] - 1
        new_l = max(self._lenscale_min, self._l_from_alpha(new_alpha))
        rval = list(params)
        rval[0] = self._alpha_from_l(new_l)
        return rval

    def s_logprior(self, params, strength=10.0):
        # -- I don't know what distribution this would be
        #    but I think it makes a nice shape
        alpha = params[0]
        alpha_min = self._alpha_from_l(self._lenscale_min)
        alpha_max = self._alpha_from_l(self._lenscale_max)
        #return strength * (alpha - alpha_min) ** 2
        log0 = -10000
        width = alpha_max - alpha_min
        #alpha_mean = 0.5 * (alpha_max + alpha_min)
        energy = strength * 0.5 * (alpha - alpha_max) ** 2 / width ** 2
        lenscale_logprior = TT.switch(alpha < alpha_min,
                         log0,
                         TT.switch(alpha < alpha_max,
                                   -energy,
                                   log0))
        if self._conditional:
            diff = params[1:3] - np.asarray([0, 1])
            return lenscale_logprior + TT.dot(diff, diff)
        else:
            return lenscale_logprior

    def cond_x(self, x, params):
        # x is a full matrix, but will only have one column

        x = TT.addbroadcast(x, 1)
        if self._conditional:
            missing_x = params[1:3]
            log_scale_x = params[3:3 + self._n_warp_segments]
        else:
            log_scale_x = params[1:1 + self._n_warp_segments]

        if self._n_warp_segments:
            # XXX
            warp_lbound = 0.
            warp_ubound = 1.
            warp_segments = np.linspace(warp_lbound,
                                        warp_ubound,
                                        self._n_warp_segments)
            scale_x = TT.exp(log_scale_x)
            z = TT.sum(
                TT.tanh(scale_x * (x - warp_segments)),
                axis=1)[:, None]
            z_min = TT.sum(
                TT.tanh(scale_x * (np.zeros((1, 1)) - warp_segments)),
                axis=1)[:, None]
            z_max = TT.sum(
                TT.tanh(scale_x * (np.ones((1, 1)) - warp_segments)),
                axis=1)[:, None]
            z = (z - z_min) / (z_max - z_min)
        else:
            z = x
        if self._conditional:
            x2_base = TT.switch(s_isnan(x), missing_x, 0)
            x2 = TT.inc_subtensor(x2_base[:, 0:1], TT.switch(s_isnan(x), 0, z))
            return x2
        else:
            return z

    def opt_logK(self, x, params):
        x2 = self.cond_x(x, params)
        logK = self._logK_of_dist(euclidean_sq_distances(x2, x2), params, True)
        params0 = [self._alpha_from_l(self._lenscale0)]
        if self._conditional:
            params0.extend([0., 1.])
        params0.extend([0.] * self._n_warp_segments)
        amin = None if self._lenscale_min is None else (
            self._alpha_from_l(self._lenscale_min))
        amax = None if self._lenscale_max is None else (
            self._alpha_from_l(self._lenscale_max))
        bounds = [[amin, amax]]
        if self._conditional:
            bounds.extend([(-5., 5.), (1e-5, 5.)])
        bounds.extend([(-.2, 2.)] * self._n_warp_segments)
        return logK, params0, bounds

    def predict_logK(self, x, z, params):
        x2 = self.cond_x(x, params)
        z2 = self.cond_x(z, params)
        logK = self._logK_of_dist(euclidean_sq_distances(x2, x2), params, True)
        logK_new = self._logK_of_dist(euclidean_sq_distances(x2, z2), params, False)
        return logK, logK_new


class SqExp(StationaryBase):
    Product = SqExpProd
    def _logK_of_dist(self, sq_dists, params, self_sim):
        _alpha = params[0]
        ll2 = TT.exp(_alpha) # aka 2 * l ** 2
        return -sq_dists / ll2


class Matern12(SqExp):
    def _K_of_dist(self, sq_dists, params, self_sim):
        _alpha = params[0]
        ll = TT.sqrt(.5 * TT.exp(_alpha))
        return TT.exp(-TT.sqrt(sq_dists) / ll)


class Matern32(StationaryBase):
    def _K_of_dist(self, sq_dists, params, self_sim):
        _alpha = params[0]
        ll2 = .5 * TT.exp(_alpha) # aka l ** 2
        nrmsq = sq_dists / ll2
        if self_sim:
            # -- help grad by suppressing 0/0 -> NaN
            nrmsq = zero_diag(nrmsq)
        nrm_root_3 = TT.sqrt(3 * nrmsq)
        return ((1 + nrm_root_3) * TT.exp(-nrm_root_3))


class Matern52(StationaryBase):
    def _K_of_dist(self, sq_dists, params, self_sim):
        _alpha = params[0]
        ll2 = .5 * TT.exp(_alpha) # aka l ** 2
        nrmsq = sq_dists / ll2
        if self_sim:
            # -- help grad by suppressing 0/0 -> NaN
            nrmsq = zero_diag(nrmsq)
        nrm_root_5 = TT.sqrt(5 * nrmsq)
        coef = 1 + nrm_root_5 + 5. / 3. * nrmsq
        return coef * TT.exp(-nrm_root_5)


Choice2 = SqExp
#class Choice2(StationaryBase):
    #def _logK_of_dist(self, sq_dists, params, self_sim):
        #_alpha = params[0]
        #ll2 = TT.exp(_alpha) # aka 2 * l ** 2
        #return -sq_dists / ll2


def product(kernels, slices):
    from gby import groupby
    # -- there are some kernels whose product can be handled
    #    by the same sort of Theano graph as it takes to handle
    #    just one term of the product. Pre-consolidating such
    #    sub-products saves a huge amount of compilation time
    #    and it runs faster too.
    prod_mergeable = groupby(zip(kernels, slices),
                             lambda ks: ks[0].prodkey())
    kernels_ = []
    slices_ = []
    for key, mergeable in prod_mergeable.items():
        print key, mergeable
        if len(mergeable) > 1:
            kern = mergeable[0][0].Product(mergeable)
            slc = kern.column_idxs
        else:
            (kern, slc), = mergeable
        kernels_.append(kern)
        slices_.append(slc)
    if len(kernels_) == 1:
        # -- XXX ignores slc ... is ok?
        return kernels_[0]
    return Product(kernels_, slices_)


class Product(Kernel):
    def __init__(self, kernels, slices):
        self.kernels = kernels
        self.slices = slices
        self.n_params = sum(k.n_params for k in kernels)

    def reduce_lenscale(self, params):
        rval = np.zeros_like(params)
        offset = 0
        for k in self.kernels:
            rval[offset: offset + k.n_params] = (
                k.reduce_lenscale(params[offset: offset + k.n_params]))
            offset += k.n_params
        return rval

    def s_logprior(self, params):
        offset = 0
        lps = []
        for k in self.kernels:
            lps.append(k.s_logprior(params[offset: offset + k.n_params]))
            offset += k.n_params
        return reduce(lambda a, b: a + b, lps)

    def opt_logK(self, x, params):
        # return a cost, and parameter vector suitable for fitting
        # the GP, and bounds on that parameter vector

        params0 = []
        bounds = []
        offset = 0
        logKs = []
        for kern, slice_k in zip(self.kernels, self.slices):
            params_k = params[offset: offset + kern.n_params]
            #if slice_k is None:
                #logK_k, params0_k, bounds_k = kern.opt_logK(x, params_k)
            #else:
            logK_k, params0_k, bounds_k = kern.opt_logK(x[:, slice_k],
                                                        params_k)
            logKs.append(check_K(logK_k))
            params0.extend(params0_k)
            bounds.extend(bounds_k)
            offset += kern.n_params

        if len(self.kernels) == 1:
            return logKs[0], params0, bounds
        else:
            Kstack = TT.stack(*logKs)
            logK = TT.sum(Kstack, axis=0)
            return logK, params0, bounds

    def predict_logK(self, x, z, params):
        # s_mean, s_x for computing mean from s_x
        logKs = []
        logKs_new = []
        offset = 0
        for kern, slice_k in zip(self.kernels, self.slices):
            params_k = params[offset: offset + kern.n_params]
            #if slice_k is None:
                #logK_k, logK_new_k = kern.predict_logK(x, z, params_k)
            #else:
            logK_k, logK_new_k = kern.predict_logK(
                x[:, slice_k], z[:, slice_k], params_k)
            logKs.append(logK_k)
            logKs_new.append(logK_new_k)
            offset += kern.n_params

        if len(self.kernels) == 1:
            return logKs[0], logKs_new[0]
        else:
            logK = TT.sum(TT.stack(*logKs), axis=0)
            logK_new = TT.sum(TT.stack(*logKs_new), axis=0)
            return logK, logK_new


def prod_of(Kcls, slices):
    kernels = [Kcls() for ii in range(len(slices))]
    return Product(kernels, slices)


class Mixture(Kernel):
    def __init__(self, kernels, slices):
        self.kernels = kernels
        self.slices = slices
        self.n_my_params = len(kernels) - 1
        self.n_params = sum(k.n_params for k in kernels) + self.n_my_params
        self.prior_strength = 2.0

    def reduce_lenscale(self, params):
        rval = np.zeros_like(params)
        offset = 0
        for k in self.kernels:
            rval[offset: offset + k.n_params] = (
                k.reduce_lenscale(params[offset: offset + k.n_params]))
            offset += k.n_params
        # shrink weights back to even weighting
        rval[offset: offset + len(self.kernels) - 1] *= 0.75
        return rval

    def s_logprior(self, params):
        offset = 0
        lps = []
        for k in self.kernels:
            lps.append(k.s_logprior(params[offset: offset + k.n_params]))
            offset += k.n_params
        # -- multiplicative because they are independent
        lp = reduce(lambda a, b: a + b, lps)
        log_weights = params[offset: offset + self.n_my_params]
        return lp - self.prior_strength * TT.dot(log_weights, log_weights)

    def opt_K(self, x, params):
        # return a cost, and parameter vector suitable for fitting
        # the GP, and bounds on that parameter vector

        params0 = []
        bounds = []
        offset = 0
        Ks = []
        for kern, slice_k in zip(self.kernels, self.slices):
            params_k = params[offset: offset + kern.n_params]
            K_k, params0_k, bounds_k = kern.opt_K(x[:, slice_k], params_k)
            Ks.append(K_k)
            params0.extend(params0_k)
            bounds.extend(bounds_k)
            offset += kern.n_params

        params0.extend([0.0] * self.n_my_params)
        bounds.extend([(-4, 4)] * self.n_my_params)

        log_weights = TT.concatenate((np.asarray([0.0]),
                                      params[offset:offset + self.n_my_params]))
        weights = TT.exp(log_weights) / TT.exp(log_weights).sum()

        if len(self.kernels) == 1:
            return Ks[0], params0, bounds
        else:
            Kstack = TT.stack(*Ks)
            weighted_Kstack = weights[:, None, None] * Kstack
            K = TT.sum(weighted_Kstack, axis=0)
            # XXX: log_K, should be logadd here (#11)
            return K, params0, bounds

    def predict_K(self, x, z, params):
        # s_mean, s_x for computing mean from s_x
        Ks = []
        Ks_new = []
        offset = 0
        for kern, slice_k in zip(self.kernels, self.slices):
            params_k = params[offset: offset + kern.n_params]
            K_k, K_new_k = kern.predict_K(
                x[:, slice_k], z[:, slice_k], params_k)
            Ks.append(K_k)
            Ks_new.append(K_new_k)
            offset += kern.n_params

        log_weights = TT.concatenate((np.asarray([0]),
                                      params[offset:offset + self.n_my_params]))
        weights = TT.exp(log_weights) / TT.exp(log_weights).sum()

        if len(self.kernels) == 1:
            return Ks[0], Ks_new[0]
        else:
            # XXX: log_K, should be logadd here (#11)
            wK = TT.sum(
                weights[:, None, None] * TT.stack(*Ks), axis=0)
            wK_new = TT.sum(
                weights[:, None, None] * TT.stack(*Ks_new), axis=0)
            return wK, wK_new

def mix_of(Kcls, slices):
    kernels = [Kcls() for ii in range(len(slices))]
    return Mixture(kernels, slices)
