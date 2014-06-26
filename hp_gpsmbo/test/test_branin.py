from functools import partial
import numpy as np
import hyperopt
from hyperopt.tests.test_domains import branin
import hp_gpsmbo.hpsuggest

def test_branin(suggest=hp_gpsmbo.hpsuggest.suggest, seed=1, iters=10):
    import matplotlib.pyplot as plt
    plt.ion()
    mins = []
    all_ys = []
    for ii in range(int(seed), int(seed) + int(iters)):
        print 'SEED', ii
        space = branin()
        trials = hyperopt.Trials()
        hyperopt.fmin(
            fn=lambda x: x,
            space=space.expr,
            trials=trials,
            algo=partial(suggest, stop_at=0.398),
            rstate=np.random.RandomState(ii),
            max_evals=50)
        plt.subplot(2, 1, 1)
        plt.cla()
        ys = trials.losses()
        all_ys.append(ys)
        for ys_jj in all_ys:
            plt.plot(ys_jj)
        plt.plot(trials.losses())
        plt.subplot(2, 1, 2)
        plt.cla()
        for ys_jj in all_ys:
            plt.plot(ys_jj)
        plt.ylim(0, 1)
        plt.axhline(np.min(ys))
        plt.annotate('min=%f' % np.min(ys), xy=(1, np.min(ys)))
        plt.draw()
        mins.append(min(ys))
        print 'MINS', mins
    assert np.max(mins) < 0.398
