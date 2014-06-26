import numpy as np
from hp_gpsmbo import GPR, SqExp, Product


def test_lenscale_wider():
    # Smoke test that changing lenscale changes fit
    pass


def test_product_smoke():
    X = np.random.randn(10, 2)
    y = np.random.randn(10)
    model = GPR(
        Product(
            [SqExp(), SqExp()],
            [slice(0, 1), slice(1, 2)]),
        )
    model.fit(X, y)


