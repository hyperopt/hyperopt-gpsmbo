import numpy as np
from hp_gpsmbo import gpr_math
import theano
import scipy.stats


def test_normal_pdf():
    rng = np.random.RandomState(123)
    norm = scipy.stats.norm

    N = 50
    x = rng.randn(N)
    mean = rng.randn(N)
    var = rng.randn(N) ** 2

    s_x, s_m, s_v = theano.tensor.dvectors('xmv')

    fn = theano.function([s_x, s_m, s_v],
                         gpr_math.s_normal_pdf(s_x, s_m, s_v))


    assert np.allclose(norm.pdf(x, mean, np.sqrt(var)),
                       fn(x, mean, var))


def test_normal_logpdf():
    rng = np.random.RandomState(123)
    norm = scipy.stats.norm

    N = 50
    x = rng.randn(N) * 10 - 50
    mean = rng.randn(N)
    var = rng.randn(N) ** 2

    s_x, s_m, s_v = theano.tensor.dvectors('xmv')

    logfn = theano.function([s_x, s_m, s_v],
                         gpr_math.s_normal_logpdf(s_x, s_m, s_v))

    assert np.allclose(norm.logpdf(x, mean, np.sqrt(var)),
                       logfn(x, mean, var))


def test_normal_cdf():
    rng = np.random.RandomState(123)
    norm = scipy.stats.norm

    N = 50
    x = rng.randn(N)
    mean = rng.randn(N)
    var = rng.randn(N) ** 2

    #x = np.sort(x)
    #mean = np.zeros(N)
    #var = np.ones(N)

    s_x, s_m, s_v = theano.tensor.dvectors('xmv')

    fn = theano.function([s_x, s_m, s_v],
                         gpr_math.s_normal_cdf(s_x, s_m, s_v))
    myval = fn(x, mean, var)
    spval = norm.cdf(x, mean, np.sqrt(var))
    for xi, myv, spv in zip(x, myval, spval):
        print xi, 'my', myv, 'sp', spv, 'diff', (myv - spv)

    assert np.allclose(norm.cdf(x, mean, np.sqrt(var)),
                       fn(x, mean, var))

def test_normal_logcdf():
    rng = np.random.RandomState(123)
    norm = scipy.stats.norm

    N = 50
    x = rng.randn(N) * 200
    mean = rng.randn(N)
    var = rng.randn(N) ** 2

    #mean = np.zeros(N)
    #var = np.ones(N)
    #x = np.sort(x)

    s_x, s_m, s_v = theano.tensor.dvectors('xmv')

    lcdf = gpr_math.s_normal_logcdf(s_x, s_m, s_v)

    fn = theano.function([s_x, s_m, s_v], lcdf)

    myval= fn(x, mean, var)
    spval = norm.logcdf(x, mean, np.sqrt(var))
    for xi, myv, spv in zip(x, myval, spval):
        print xi, 'my', myv, 'sp', spv, 'diff', (myv - spv)
    assert np.allclose(norm.logcdf(x, mean, np.sqrt(var)),
                       myval)


def test_normal_logEI():
    #rng = np.random.RandomState(123)

    N = 2000
    thresh = np.linspace(-10, 50, N)
    #N = 100
    #thresh = np.linspace(37, 38, N)
    mean = thresh * 0
    var = thresh * 0 + 1

    s_t, s_m, s_v = theano.tensor.dvectors('tmv')

    fn = theano.function([s_t, s_m, s_v],
                         gpr_math.s_normal_logEI(s_t, s_m, s_v))

    if 0:
        #print zip(thresh, fn(thresh, mean, var))
        #print 
        a = theano.tensor.dvector()
        y = s_t ** 2 * a[2] + s_t * a[1] + a[0]
        cost = ((y - gpr_math.s_normal_logEI(s_t, s_m, s_v)) ** 2).sum()
        da = theano.grad(cost, a)
        foo = theano.function([a, s_t, s_m, s_v], [cost, da])
        res = scipy.optimize.minimize(foo, [0, -1, -1], jac=True,
                                      args=(thresh, mean, var),
                                      method='L-BFGS-B')
        print res.x

    from hyperopt.criteria import logEI_gaussian
    if 0:
        import matplotlib.pyplot as plt
        y = fn(thresh, mean, var)
        z = logEI_gaussian(mean, var, thresh)
        plt.plot(thresh, y)
        plt.plot(thresh, z)
        plt.show()

    # -- the gpr_math logEI uses a quadratic approximation for very
    #    hopeless points, which gives the right derivative, but the
    #    slightly wrong value
    assert np.allclose(logEI_gaussian(mean, var, thresh),
                       fn(thresh, mean, var),
                       atol=1e-3, rtol=1e-4)

    if 0:
        d_t = theano.grad(gpr_math.s_normal_logEI(s_t, s_m, s_v).sum(), s_t)
        d_fn = theano.function([s_t, s_m, s_v], d_t)

        import matplotlib.pyplot as plt
        plt.plot(thresh, d_fn(thresh, mean, var))
        plt.show()


def test_logEBI():

    def EBI_from_sample(sample, l, u):
        sample = sample - l
        sample[sample < 0] = 0
        sample[sample > (u - l)] = 0
        return sample.mean()

    def normal_EBI_numeric(l, u, m, sigma, N, rng):
        return EBI_from_sample(rng.randn(N) * sigma + m, l, u)

    def normal_EBI_analytic(l, u, m, sigma):
        from scipy.stats import norm
        from hyperopt.criteria import EI_gaussian
        EI_l = EI_gaussian(m, sigma ** 2, l)
        EI_u = EI_gaussian(m, sigma ** 2, u)
        term = (l - u) * norm.cdf((m - u) / sigma)
        return EI_l - EI_u + term

    s_l, s_u, s_m, s_sigma = theano.tensor.dscalars('lums')
    s_EBI = gpr_math.s_normal_EBI(s_l, s_u, s_m, s_sigma ** 2)
    normal_EBI_theano = theano.function([s_l, s_u, s_m, s_sigma], s_EBI)


    def assert_match(l, u, m, sigma, N=100000, seed=123):
        l, u, m, sigma = map(float, (l, u, m, sigma))
        num = normal_EBI_numeric(l, u, m, sigma, N, np.random.RandomState(seed))
        ana = normal_EBI_analytic(l, u, m, sigma)
        thn = normal_EBI_theano(l, u, m, sigma)
        if not np.allclose(num, ana, atol=0.01, rtol=.01):
            print 'test_EBI mismatch', l, u, m, sigma, '->', num, ana
            assert 0
        if not np.allclose(thn, ana, atol=0.0001, rtol=.0001):
            print 'test_EBI theano mismatch', l, u, m, sigma, '->', thn, ana
            assert 0

    assert_match(0, 100, 0, 1)
    assert_match(0, 0.2, 0, 1)
    assert_match(0, 1.2, 0, 1)
    assert_match(0, 100, 0.5, 1.5)
    assert_match(0, 0.2, 0.5, 1.5)
    assert_match(0, 1.2, 0.5, 1.5)


# -- eof flake8
