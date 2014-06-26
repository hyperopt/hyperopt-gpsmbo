from functools import partial
import numpy as np

import hyperopt
from hyperopt import hp
from hypertree import har6

import hp_gpsmbo.hpsuggest

def test_har6(suggest=hp_gpsmbo.hpsuggest.suggest, seed=1, iters=10):
    # -- see shovel/hps.py for this test with debugging scaffolding
    #    run it by typing e.g.
    # 
    #       shovel hps.run_har6 --seed=9
    #
    #    That should do a run that fails by only getting to -3.2
    mins = []
    for ii in range(int(seed), int(seed) + int(iters)):
        print 'SEED', ii
        space = {
            'a': hp.uniform('a', 0, 1),
            'b': hp.uniform('b', 0, 1),
            'c': hp.uniform('c', 0, 1),
            'x': hp.uniform('x', 0, 1),
            'y': hp.uniform('y', 0, 1),
            'z': hp.uniform('z', 0, 1),
        }
        trials = hyperopt.Trials()
        hyperopt.fmin(
            fn=har6.har6,
            space=space,
            trials=trials,
            algo=partial(suggest, stop_at=-3.32),
            rstate=np.random.RandomState(ii),
            max_evals=100)
        mins.append(min(trials.losses()))

    assert np.sum(mins > -3.32) < 3

    # XXX ideally this sum should be 0, but our optimizer
    #     isn't that good :(

