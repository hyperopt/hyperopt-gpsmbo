import numpy as np
from .kernels_base import Kernel

import theano.tensor as TT
from .op_Kcond import zero_diag, isnan as s_isnan
from .kernels_base import euclidean_sq_distances

class SqExpProd(Kernel):
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

    def __init__(self,
                 seq_kern_slice):
                 #lenscales_0,
                 #lenscales_min,
                 #lenscales_max,
                 #conditional):
        kerns, slices = zip(*seq_kern_slice)
        self._conditional = kerns[0]._conditional
        assert all(self._conditional == kern._conditional
                   for kern, slc in seq_kern_slice)
        self._lenscales_0 = np.asarray([kern._lenscale0 for kern in kerns])
        self._lenscales_min = np.asarray([kern._lenscale_min for kern in kerns])
        self._lenscales_max = np.asarray([kern._lenscale_max for kern in kerns])

        self._n_warp_segments_per_X = 0
        if self._conditional:
            self.n_params = 3 + self._n_warp_segments_per_X
        else:
            self.n_params = 1 + self._n_warp_segments_per_X
        self.n_params *= len(kerns)
        self.N = len(kerns)
        def getidx(slc):
            assert slc.start + 1 == slc.stop and slc.step == None
            return slc.start
        self.column_idxs = np.asarray(map(getidx, slices))
        self.s_idxs = TT.as_tensor_variable(self.column_idxs)

    def prodkey(self):
        # -- unique identifier of mergeable product sets
        return (type(self),
                self._conditional,
                self._n_warp_segments_per_X)

    def reduce_lenscale(self, params):
        new_l = np.maximum(self._lenscales_min,
                           self._l_from_alpha(np.asarray(params[0:self.N]) - 1))
        rval = list(params)
        rval[0:self.N] = self._alpha_from_l(new_l)
        return rval

    def unpack(self, params):
        alpha = params[0:self.N]
        cond_x = params[self.N:2 * self.N]
        cond_y = params[2 * self.N: 3 * self.N]
        return alpha, cond_x, cond_y

    def s_logprior(self, s_params, strength=10.0):
        # -- I don't know what distribution this would be
        #    but I think it makes a nice shape
        s_alpha, s_cond_x, s_cond_y = self.unpack(s_params)
        n_alpha_min = self._alpha_from_l(self._lenscales_min)
        n_alpha_max = self._alpha_from_l(self._lenscales_max)
        #return strength * (alpha - alpha_min) ** 2
        log0 = -10000
        width = n_alpha_max - n_alpha_min
        #alpha_mean = 0.5 * (alpha_max + alpha_min)
        energy = strength * 0.5 * (s_alpha - n_alpha_max) ** 2 / width ** 2
        lenscale_logprior = TT.switch(s_alpha < n_alpha_min,
                                      log0,
                                      TT.switch(s_alpha < n_alpha_max,
                                                -energy,
                                                log0)).sum()
        if self._conditional:
            diff_x = s_cond_x
            diff_y = s_cond_y - 1
            rval = (lenscale_logprior
                    + TT.dot(diff_x, diff_x)
                    + TT.dot(diff_y, diff_y))
        else:
            rval = lenscale_logprior
        assert rval.ndim == 0
        return rval

    def cond_x(self, s_x, s_params):
        #import theano
        #s_x_all = theano.printing.Print('x_all')(s_x_all)
        #s_x = s_x_all.T[self.s_idxs].T
        s_alpha, s_missing_x, s_missing_y = self.unpack(s_params)
        assert s_x.ndim == 2
        #s_x = TT.addbroadcast(s_x, 1)
        if self._conditional:
            filled_x = TT.switch(s_isnan(s_x), s_missing_x, s_x)
            filled_y = TT.switch(s_isnan(s_x), s_missing_y, 0)
        else:
            filled_x = s_x
            filled_y = None
        assert filled_x.ndim == 2
        return filled_x, filled_y


    def opt_logK(self, s_x, s_params):
        s_alpha, s_missing_x, s_missing_y = self.unpack(s_params)
        filled_x, filled_y = self.cond_x(s_x, s_params)

        lenscales = TT.sqrt(.5 * TT.exp(s_alpha))

        dist_sq = euclidean_sq_distances(filled_x / lenscales,
                                         filled_x / lenscales)
        if filled_y is not None:
            dist_sq += euclidean_sq_distances(filled_y / lenscales,
                                              filled_y / lenscales)
        # Geometric
        logK = -0.5 * dist_sq

        params0 = list(self._alpha_from_l(self._lenscales_0))
        if self._conditional:
            params0.extend([0.] * self.N)
            params0.extend([1.] * self.N)
        params0.extend([0.] * self._n_warp_segments_per_X)
        amin = self._alpha_from_l(self._lenscales_min)
        amax = self._alpha_from_l(self._lenscales_max)
        bounds = zip(amin, amax)
        if self._conditional:
            bounds.extend([(-5., 5.)] * self.N)
            bounds.extend([(1e-5, 5.)] * self.N)
        if self._n_warp_segments_per_X:
            #bounds.extend([(-.2, 2.)] * self._n_warp_segments)
            raise NotImplementedError()
        return logK, params0, bounds

    def predict_logK(self, s_x, s_z, s_params):
        filled_x_x, filled_x_y = self.cond_x(s_x, s_params)
        filled_z_x, filled_z_y = self.cond_x(s_z, s_params)

        s_alpha, s_missing_x, s_missing_y = self.unpack(s_params)
        lenscales = TT.sqrt(.5 * TT.exp(s_alpha))

        dist_xx_sq = euclidean_sq_distances(filled_x_x / lenscales,
                                            filled_x_x / lenscales)
        dist_xz_sq = euclidean_sq_distances(filled_x_x / lenscales,
                                            filled_z_x / lenscales)
        if filled_x_y is not None:
            dist_xx_sq += euclidean_sq_distances(filled_x_y / lenscales,
                                                 filled_x_y / lenscales)
            dist_xz_sq += euclidean_sq_distances(filled_x_y / lenscales,
                                                 filled_z_y / lenscales)
        logK = -0.5 * dist_xx_sq
        logK_new = -0.5 * dist_xz_sq

        #x2 = self.cond_x(s_x, s_params)
        #z2 = self.cond_x(s_z, s_params)
        #logK = self._logK_of_dist(
            #euclidean_sq_distances(x2, x2), s_params, True)
        #logK_new = self._logK_of_dist(
            #euclidean_sq_distances(x2, z2), s_params, False)
        return logK, logK_new
