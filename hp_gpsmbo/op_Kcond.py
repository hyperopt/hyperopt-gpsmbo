import numpy as np
from theano import Op, Apply, gradient
from theano import tensor as TT

class KCond(Op):
    """
    Return a vector of indexes of K to keep
    """
    def __init__(self):
        self.destructive = False

        self.props = (self.destructive,)

    def __hash__(self):
        return hash((type(self), self.props))

    def __eq__(self, other):
        return (type(self) == type(other) and self.props == other.props)

    #def infer_shape(self, node, shapes):
        #return [shapes[0]]

    def __str__(self):
        return 'KCond'

    def make_node(self, K, y, eps):
        K = TT.as_tensor_variable(K)
        y = TT.as_tensor_variable(y)
        eps = TT.as_tensor_variable(eps)
        return Apply(self, [K, y, eps], [TT.ivector()])

    def perform(self, node, inputs, outputs):
        K, y, eps = inputs
        M = K.shape[0]
        assert (M, M) == K.shape
        assert (M,) == y.shape
        order = np.argsort(y)  # best to worst
        keep = np.ones_like(y).astype(np.int32) # order matches K, y
        assert np.allclose(np.diag(K), 1.0)
        max_similarity = (K - np.eye(M)).max()
        if max_similarity + eps > 1.0:
            print 'max_similarity', max_similarity

        for ii in xrange(M - 1):
            this = order[ii]
            if not keep[this]:
                continue
            # -- we have committed to using row `this`
            # -- Now, delete all worse points within epsilon of row `this`
            #    (all pts remaining in `order` are worse by definition)
            K_this = K[this]
            for jj in xrange(ii + 1, M):
                other = order[jj]
                if not keep[other]:  # -- other's already gone
                    continue
                if (1 - K_this[other]) < eps:
                    keep[other] = 0
        keep_idxs = np.where(keep)[0].astype(np.int32)
        if len(keep_idxs) < M:
            print 'Dropping %i rows to condition K' % (
                M - len(keep_idxs))
        outputs[0][0] = keep_idxs

    def grad(self, inputs, gradients):
        return [inp.zeros_like() for inp in inputs]


def K_cond(K, y, eps):
    keep_idxs = KCond()(K, y, eps)
    keep_y = y[keep_idxs]
    # -- we want to keep the given rows and cols, hence:
    keep_K = K[keep_idxs].T[keep_idxs].T
    assert keep_K.type == K.type
    assert keep_y.type == y.type
    return keep_K, keep_y, keep_idxs

class ZeroDiag(Op):
    """ Return a square matrix with the diagonal zero-d out.

    The advantage of this Op over masking techniques based on arithmetic
    is that this Op can remove NaNs from the diagonal.
    """
    def __init__(self):
        self.destructive = False
        self.props = (self.destructive,)

    def __hash__(self):
        return hash((type(self), self.props))

    def __eq__(self, other):
        return (type(self) == type(other) and self.props == other.props)

    def infer_shape(self, node, shapes):
        return shapes

    def __str__(self):
        return 'ZeroDiag'

    def make_node(self, K):
        K = TT.as_tensor_variable(K)
        return Apply(self, [K], [K.type()])

    def perform(self, node, inputs, outputs):
        K, = inputs
        rval = K.copy()
        idxs = np.arange(K.shape[0])
        rval[idxs, idxs] = 0
        outputs[0][0] = rval

    def connection_pattern(self, node):
        return [[True]]

    def grad(self, inputs, gradients):
        gY = gradients[0]
        return [zero_diag(gY)]

zero_diag = ZeroDiag()


class ZeroForNan(Op):
    """ Return a square matrix with the diagonal zero-d out.

    The advantage of this Op over masking techniques based on arithmetic
    is that this Op can remove NaNs from the diagonal.
    """
    def __init__(self):
        self.destructive = False
        self.props = (self.destructive,)

    def __hash__(self):
        return hash((type(self), self.props))

    def __eq__(self, other):
        return (type(self) == type(other) and self.props == other.props)

    def infer_shape(self, node, shapes):
        return shapes

    def __str__(self):
        return 'ZeroForNan'

    def make_node(self, K):
        K = TT.as_tensor_variable(K)
        return Apply(self, [K], [K.type()])

    def perform(self, node, inputs, outputs):
        K, = inputs
        rval = K.copy()
        rval[np.isnan(rval)] = 0
        outputs[0][0] = rval

    def connection_pattern(self, node):
        return [[True]]

    def grad(self, inputs, gradients):
        #K, = inputs
        gY, = gradients
        return [gY]

zero_for_nan = ZeroForNan()


class IsNan(Op):
    """ Return a square matrix with the diagonal zero-d out.

    The advantage of this Op over masking techniques based on arithmetic
    is that this Op can remove NaNs from the diagonal.
    """
    def __init__(self):
        self.destructive = False
        self.props = (self.destructive,)

    def __hash__(self):
        return hash((type(self), self.props))

    def __eq__(self, other):
        return (type(self) == type(other) and self.props == other.props)

    def infer_shape(self, node, shapes):
        return shapes

    def __str__(self):
        return 'IsNan'

    def make_node(self, K):
        K = TT.as_tensor_variable(K)
        otype = TT.TensorType(dtype='int8',
                              broadcastable=K.broadcastable)
        return Apply(self, [K], [otype()])

    def perform(self, node, inputs, outputs):
        outputs[0][0] = np.isnan(inputs[0]).astype('int8')

    #def connection_pattern(self, node):
        #return [[False]]

    def grad(self, inputs, gradients):
        return [gradient.DisconnectedType()()]

isnan = IsNan()

import scipy.linalg
import theano
from theano.gof import local_optimizer, PureOp
from theano.tensor.opt import (register_stabilize,
        register_specialize, register_canonicalize)
from theano.sandbox.linalg.ops import Cholesky

class LazyCholesky(PureOp):
    def __init__(self, lower):
        self.lower = lower
        self.props = (lower,)

    def __hash__(self):
        return hash((type(self), self.props))

    def __eq__(self, other):
        return (type(self) == type(other) and self.props == other.props)

    def make_node(self, X, use_buf, buf_idx):
        return Apply(self,
                     [X, use_buf, buf_idx],
                     [X.type(), theano.gof.type.generic()])

    def infer_shape(self, node, shapes):
        return [shapes[0], None]

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        s_X, s_use_buf, s_buf_idx = node.inputs
        s_chol, s_buf = node.outputs
        comp_X = compute_map[s_X]
        comp_use_buf = compute_map[s_use_buf]
        comp_buf_idx = compute_map[s_buf_idx]
        comp_chol = compute_map[s_chol]
        #comp_buf = compute_map[s_buf]

        stor_X = storage_map[s_X]
        stor_use_buf = storage_map[s_use_buf]
        stor_buf_idx = storage_map[s_buf_idx]
        stor_chol = storage_map[s_chol]
        stor_buf = storage_map[s_buf]
        def thunk():
            # -- compute the use_buf flag
            if not comp_use_buf[0]:
                return [1]
            if not comp_buf_idx[0]:
                return [2]
            buf_idx = int(stor_buf_idx[0])
            use_buf = stor_use_buf[0]
            if use_buf:
                buf_dict = stor_buf[0]
                assert buf_dict is not None, 'buf output is empty'
                chol = buf_dict[buf_idx]
            else:
                # -- compute a cholesky and store to buffer
                if not comp_X[0]:
                    return [0]
                X = stor_X[0]
                chol = scipy.linalg.cholesky(X, lower=self.lower)
                print 'computing cholesky', buf_idx
                if stor_buf[0] is None:
                    stor_buf[0] = {}
                chol = chol.astype(X.dtype)
                buf_dict = stor_buf[0]
                buf_dict[buf_idx] = chol

            stor_chol[0] = chol.copy()
            comp_chol[0] = 1
            return []

        thunk.lazy = True
        thunk.inputs = [storage_map[v] for v in node.inputs]
        thunk.outputs = [storage_map[v] for v in node.outputs]
        return thunk

use_lazy_cholesky = False
use_lazy_cholesky_idx = None

@register_specialize
@local_optimizer(None)
def lazy_cholesky(node):
    """
    If a general solve() is applied to the output of a cholesky op, then
    replace it with a triangular solve.
    """
    if not use_lazy_cholesky:
        return

    if isinstance(node.op, Cholesky):
        assert use_lazy_cholesky.name
        for var in node.fgraph.variables:
            if var.name == use_lazy_cholesky.name:
                break
        else:
            raise Exception('var not found in graph', use_lazy_cholesky)
        buf_flag = var

        for var in node.fgraph.variables:
            if var.name == use_lazy_cholesky_idx.name:
                break
        else:
            raise Exception('var not found in graph', use_lazy_cholesky_idx)
        buf_idx = var
        assert buf_idx is not buf_flag
        X, = node.inputs
        chol, buf = LazyCholesky(node.op.lower)(X, buf_flag, buf_idx)
        assert chol.type == node.outputs[0].type
        return [chol]


from scipy.stats import norm

class NormalLogEIDiffSigmaScalar(theano.scalar.basic.ScalarOp):
    nin = 2
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def impl(self, diff, sigma):
        z = diff / sigma
        if z < 34:
            a = -diff * norm.cdf(-z)
            b = sigma * norm.pdf(-z)
            rval = np.log(a + b)
        else:
            rval = (-4.86466981
                    -0.12442506 * z
                    -0.49903031 * z ** 2)
        return rval

    def c_code(self, node, name, inp, out, sub):
        diff, sigma = inp
        y, = out
        z = y + '_z'
        a = y + '_a'
        b = y + '_b'
        cdf = y + '_cdf'
        pdf = y + '_pdf'
        #root_2pi = '%' % np.sqrt(2 * np.pi)
        if node.inputs[0].type in theano.scalar.basic.float_types:
            return """
                double %(z)s = %(diff)s / %(sigma)s;
                if (%(z)s < 34)
                {
                    double %(cdf)s = .5 * erfc(%(z)s / sqrt(2.));
                    double %(pdf)s = exp(-.5 * %(z)s * %(z)s) / sqrt(2 * M_PI);
                    double %(a)s = -%(diff)s * %(cdf)s;
                    double %(b)s = %(sigma)s * %(pdf)s;
                    %(y)s = log(%(a)s + %(b)s);
                }
                else
                {
                    %(y)s = -4.86466981
                            -0.12442506 * %(z)s
                            -0.49903031 * %(z)s * %(z)s;
                }
                """ % locals()
        raise NotImplementedError('only floating point is implemented')

    def c_code_cache_version(self):
        return (1,)

    def grad(self, inp, grads):
        y = self(*inp)
        gy, = grads
        float_out = theano.scalar.basic.float_out
        gd = NormalLogEIDiffSigmaScalarGrad0(float_out)(y, gy, *inp)
        gs = NormalLogEIDiffSigmaScalarGrad1(float_out)(y, gy, *inp)
        return gd, gs

class NormalLogEIDiffSigmaScalarGrad0(theano.scalar.basic.ScalarOp):
    nin = 4
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def impl(self, logEI, glogEI, diff, sigma):
        z = diff / sigma
        if z < 34:
            logcdf = norm.logcdf(-z, 0, 1)
            ddiff = -np.exp(logcdf - logEI)     # aka: -cdf / EI
        else:
            foo = 2 * .49903031
            dz = (-0.12442506 - foo * z)
            ddiff = dz / sigma
        return ddiff * glogEI

    def c_code(self, node, name, inp, out, sub):
        logEI, glogEI, diff, sigma = inp
        y, = out
        z = y + '_z'
        logcdf = y + '_logcdf'
        #root_2pi = '%' % np.sqrt(2 * np.pi)
        if node.inputs[0].type in theano.scalar.basic.float_types:
            return """
                double %(z)s = %(diff)s / %(sigma)s;
                if (%(z)s < 34)
                {
                    double %(logcdf)s = log(.5) + log(erfc(%(z)s / sqrt(2.)));
                    %(y)s = -exp(%(logcdf)s - %(logEI)s) * %(glogEI)s;
                }
                else
                {
                    %(y)s = (-0.12442506 - 2 * .49903031 * %(z)s)
                        / %(sigma)s
                        * %(glogEI)s;
                }
                """ % locals()
        raise NotImplementedError('only floating point is implemented')

    def c_code_cache_version(self):
        return (1,)

class NormalLogEIDiffSigmaScalarGrad1(theano.scalar.basic.ScalarOp):
    nin = 4
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def impl(self, logEI, glogEI, diff, sigma):
        z = diff / sigma
        if z < 34:
            logpdf = norm.logpdf(-z, 0, 1)
            dsigma = np.exp(logpdf - logEI)  # aka: pdf / EI
        else:
            foo = 2 * .49903031
            dz = (-0.12442506 - foo * z)
            dsigma = dz * (-z / sigma)
                #(foo * z) ** 2 / sigma
        return dsigma * glogEI

    def c_code(self, node, name, inp, out, sub):
        logEI, glogEI, diff, sigma = inp
        y, = out
        z = y + '_z'
        logpdf = y + '_logpdf'
        #root_2pi = '%' % np.sqrt(2 * np.pi)
        if node.inputs[0].type in theano.scalar.basic.float_types:
            return """
                double %(z)s = %(diff)s / %(sigma)s;
                if (%(z)s < 34)
                {
                    double %(logpdf)s = -.5 * (log(2 * M_PI) + %(z)s * %(z)s);
                    %(y)s = exp(%(logpdf)s - %(logEI)s) * %(glogEI)s;
                }
                else
                {
                    %(y)s = (-0.12442506 - 2 * .49903031 * %(z)s)
                        * (-%(z)s / %(sigma)s)
                        * %(glogEI)s;
                }
                """ % locals()
        raise NotImplementedError('only floating point is implemented')

    def c_code_cache_version(self):
        return (1,)

normal_logEI_diff_sigma_scalar = NormalLogEIDiffSigmaScalar(
    theano.scalar.upgrade_to_float_no_complex,
    name='normal_logEI_diff_sigma_elemwise')

normal_logEI_diff_sigma_elemwise = theano.tensor.Elemwise(
    normal_logEI_diff_sigma_scalar)

class NormalLogEIDiffSigma(theano.Op):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash((type(self),))

    def make_node(self, diff, sigma):
        diff = theano.tensor.as_tensor_variable(diff)
        sigma = theano.tensor.as_tensor_variable(sigma)
        foo = diff + sigma
        return theano.Apply(self, [diff, sigma], [foo.type()])

    def perform(self, node, inputs, output_storage):
        diff, sigma = inputs
        z = diff / sigma
        # -- the following formula is cuter, but
        #    Theano doesn't produce as stable a gradient I think?
        #return sigma * (z * s_normal_cdf(z, 0, 1) + s_normal_pdf(z, 0, 1))
        a = -diff * norm.cdf(-z, 0, 1)
        b = sigma * norm.pdf(-z, 0, 1)
        rval_naive = np.log(a + b)
        zz = z[z > 34]
        interp = (-4.86466981
                  -0.12442506 * zz
                  -0.49903031 * zz ** 2)
        rval_naive[z > 34] = interp
        output_storage[0][0] = rval_naive

    def grad(self, inputs, output_gradients):
        y = NormalLogEIDiffSigma()(*inputs)
        gy, = output_gradients
        return NormalLogEIGrad()(y, gy, *inputs)

normal_logEI_diff_sigma = NormalLogEIDiffSigma()


class NormalLogEIGrad(theano.Op):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash((type(self),))

    def make_node(self, logEI, gEI, diff, sigma):
        return theano.Apply(self,
                            [logEI, gEI, diff, sigma],
                            [diff.type(), sigma.type()])

    def perform(self, node, inputs, output_storage):
        logEI, gEI, diff, sigma = inputs
        z = diff / sigma
        logcdf = norm.logcdf(-z, 0, 1)
        logpdf = norm.logpdf(-z, 0, 1)
        #for zi, a, b, c in zip(z, logcdf, logpdf, logEI):
            #print zi, 'cdf', a, 'pdf', b, 'EI', c, 'logdz', a - c, 'logsig', b - c
        dz = -np.exp(logcdf - logEI)     # aka: -cdf / EI
        dsigma = np.exp(logpdf - logEI)  # aka: pdf / EI

        #if np.any(z > 20):
        #    print 'NormalLogEIGrad: bigz', z[z > 20]

        foo = 2 * .49903031
        dz[z > 34] = -0.12442506 - foo * z[z > 34]
        dsigma[z > 34] = dz[z > 34] * (-z[z > 34] / sigma[z > 34])
        dz[z > 34] /= sigma[z > 34]

        output_storage[0][0] = dz * gEI
        output_storage[1][0] = dsigma * gEI
        #if np.any(np.isnan(dz)):
        #    import pdb; pdb.set_trace()
        #print ('logEI grad: gEI=%s dz=%s dsigma=%s' % (gEI, dz, dsigma))


# -- eof
