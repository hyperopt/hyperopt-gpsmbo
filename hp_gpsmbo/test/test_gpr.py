import numpy as np
from hp_gpsmbo import GPR_ML2, kernels

#class Test1(unittest.TestCase):

def test_prior_mean(GPR=GPR_ML2):
    # Test that the prior mean and prior variance are respected
    # in a simple case where there is just a single data point at 0.
    for prior_mean in (-5, 0, 5):
        for prior_var in (.1, 1):
            gpr = GPR(kernels.SqExp(1.0, 1e-4, 10, conditional=False),
                    maxiter=1,
                    prior_var=prior_var,
                    prior_mean=prior_mean)
            gpr.fit([[0]], [1])
            m, v = gpr.predict([[-10], [0], [10]], eval_MSE=True)
            assert np.allclose(m[0], prior_mean)
            assert np.allclose(m[1], 1)
            assert np.allclose(m[2], prior_mean)
            assert np.allclose(v[0], prior_var)
            assert np.allclose(v[1], 0)
            assert np.allclose(v[2], prior_var)


def test_data_pts_respected(GPR=GPR_ML2):
    X = np.asarray([[-1], [0], [1.5]])
    y = np.asarray([-4, 1, 0.5])
    for prior_mean in (-5, 0, 5):
        for prior_var in (.1, 1):
            gpr = GPR(kernels.SqExp(1.0, 1e-4, 10, conditional=False),
                    maxiter=1,
                    prior_var=prior_var,
                    prior_mean=prior_mean)
            gpr.fit(X, y)
            m, v = gpr.predict(X, eval_MSE=True)
            assert np.all(v < 1e-7)
            assert np.allclose(m, y)


# -- flake8 eof
