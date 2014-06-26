import numpy as np
from theano.gradient import verify_grad
import theano.tensor
from hp_gpsmbo.op_Kcond import normal_logEI_diff_sigma
from hp_gpsmbo.op_Kcond import normal_logEI_diff_sigma_elemwise
from hyperopt.criteria import logEI_gaussian

def test_normal_logEI():
    rng = np.random.RandomState(123)

    N = 2000
    thresh = np.linspace(-50, 500, N)
    #N = 100
    #thresh = np.linspace(37, 38, N)
    mean = thresh * 0
    var = 1e-1 + rng.rand(N)
    sigma = np.sqrt(var)

    s_t, s_m, s_v = theano.tensor.dvectors('tmv')

    fn = theano.function([s_t, s_m, s_v],
                         normal_logEI_diff_sigma(s_t - s_m,
                                                 theano.tensor.sqrt(s_v)))

    my = fn(thresh, mean, var)
    ref = logEI_gaussian(mean, var, thresh)
    for xi, myv, spv in zip(thresh, my, ref):
        print xi, 'my', myv, 'sp', spv, 'diff', (myv - spv)

    assert np.any(thresh / sigma > 34)
    assert np.all(np.isfinite(my))
    assert np.allclose(my[thresh/sigma < 34], ref[thresh/sigma < 34])
    assert np.allclose(my, ref, rtol=.1)


def explore_grad():
    N = 15
    ubound = 6e2
    diff = np.ones(N) * 100
    rng = np.random.RandomState(123)
    #diff = np.linspace(0, ubound, N).astype('float64')
    #var = np.random.rand(N) * .1 + 1 #1e-8 +
    #var = np.ones(N) * .01
    var = np.exp(rng.randn(N) * 10) ** 2
    var = np.sort(var)

    s_d, s_v = theano.tensor.dvectors('dv')
    s_y = normal_logEI_diff_sigma(s_d, theano.tensor.sqrt(s_v))
    s_gd, s_gv = theano.tensor.grad(s_y.sum(), [s_d, s_v])

    fn = theano.function([s_d, s_v], [s_y, s_gd, s_gv])

    eps = ubound / 1e8 # 1e1 # 1e-4
    y, gd, gv = fn(diff, var)
    y_eps, _, _ = fn(diff + eps, var)
    y_eps2, _, _ = fn(diff, var + eps)
    for di, yi, yi_eps, yi2, gdi, gvi in zip(diff, y, y_eps, y_eps2, gd, gv):
        print 'di %.6f\tyi:%.6f\tgi:%.6f\tref:%.6f\tgv:%s\tref:%s' % (
            di, yi, gdi, (yi_eps - yi) / eps, gvi, (yi2 - yi) / eps
        )

def test_grad_arg0():
    N = 50
    def f_arg01(x):
        return normal_logEI_diff_sigma(x, np.ones(1))
    def f_arg0(x):
        return normal_logEI_diff_sigma(x, np.ones(N))

    rng = np.random.RandomState(123)
    diffvec = (rng.rand(N) - .5) * 200

    verify_grad(f_arg01, [np.asarray([-50.])], rng=rng)
    verify_grad(f_arg01, [np.asarray([50.])], rng=rng)
    verify_grad(f_arg0, [diffvec], rng=rng, rel_tol=1e-3)

def test_grad_arg1():
    N = 50
    def f_arg1(x):
        return normal_logEI_diff_sigma(np.ones(N) * 100,
                                       x)
    rng = np.random.RandomState(123)
    #sigmavec = np.exp(np.linspace(N) * 10)
    sigmavec = np.linspace(.1, 10, N)

    verify_grad( f_arg1, [sigmavec], rng=rng, rel_tol=1e-3)

def test_normal_logEI_elemwise():
    rng = np.random.RandomState(123)

    N = 2000
    thresh = np.linspace(-50, 500, N)
    #N = 100
    #thresh = np.linspace(37, 38, N)
    mean = thresh * 0
    var = 1e-1 + rng.rand(N)
    sigma = np.sqrt(var)

    s_t, s_m, s_v = theano.tensor.dvectors('tmv')

    fn = theano.function([s_t, s_m, s_v],
                         normal_logEI_diff_sigma_elemwise(
                             s_t - s_m,
                             theano.tensor.sqrt(s_v)))

    my = fn(thresh, mean, var)
    ref = logEI_gaussian(mean, var, thresh)
    for xi, myv, spv in zip(thresh, my, ref):
        print xi, 'my', myv, 'sp', spv, 'diff', (myv - spv)

    assert np.any(thresh / sigma > 34)
    assert np.all(np.isfinite(my))
    assert np.allclose(my[thresh/sigma < 34], ref[thresh/sigma < 34])
    assert np.allclose(my, ref, rtol=.1)

def test_grad_arg0_elemwise():
    N = 50
    def f_arg01(x):
        return normal_logEI_diff_sigma_elemwise(x, np.ones(1))
    def f_arg0(x):
        return normal_logEI_diff_sigma_elemwise(x, np.ones(N))

    rng = np.random.RandomState(123)
    diffvec = (rng.rand(N) - .5) * 200

    verify_grad(f_arg01, [np.asarray([-50.])], rng=rng)
    verify_grad(f_arg01, [np.asarray([50.])], rng=rng)
    verify_grad(f_arg0, [diffvec], rng=rng, rel_tol=1e-3)

def test_grad_arg1_elemwise():
    N = 50
    def f_arg1(x):
        return normal_logEI_diff_sigma_elemwise(
            np.ones(N) * 100,
            x)
    rng = np.random.RandomState(123)
    #sigmavec = np.exp(np.linspace(N) * 10)
    sigmavec = np.linspace(.1, 10, N)

    verify_grad( f_arg1, [sigmavec], rng=rng, rel_tol=1e-3)

