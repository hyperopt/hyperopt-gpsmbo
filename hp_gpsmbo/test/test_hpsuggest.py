from functools import partial
import unittest
import numpy as np
from hyperopt import rand
from hyperopt import Trials, fmin

from hyperopt.tests.test_domains import CasePerDomain
from hp_gpsmbo import suggest_algos

def passthrough(x):
    return x

class TestSmoke(unittest.TestCase, CasePerDomain):
    def work(self):
        fmin(
            fn=passthrough,
            space=self.bandit.expr,
            algo=partial(suggest_algos.ei,
                         warmup_cutoff=3),
            max_evals=10)


class TestOpt(unittest.TestCase, CasePerDomain):
    # -- these thresholds are pretty low
    #    but they are set to that random does not pass them
    #    (at least, probably not)
    thresholds = dict(
            quadratic1=1e-5,
            q1_lognormal=0.0002,
            distractor=-2.0,
            gauss_wave=-2.8,
            gauss_wave2=-2.20,
            n_arms=-3.0,
            many_dists=-1.,
            branin=0.5,
            )

    LEN = dict(
            # -- running a long way out tests overflow/underflow
            #    to some extent
            twoarms=15,
            gausswave=50, 
            quadratic1=1000,
            many_dists=200,
            distractor=35,
            #q1_lognormal=100,
            branin=200,
            )

    def setUp(self):
        self.olderr = np.seterr('raise')
        np.seterr(under='ignore')

    def tearDown(self, *args):
        np.seterr(**self.olderr)

    def work(self):
        np.random.seed(1234)
        bandit = self.bandit
        LEN = self.LEN.get(bandit.name, 100)
        thresh = self.thresholds[bandit.name]

        print 'STARTING TEST', bandit.name
        rtrials = Trials()
        fmin(fn=passthrough,
            space=self.bandit.expr,
            trials=rtrials,
            algo=rand.suggest,
            max_evals=LEN,
            rstate=np.random)
        print 'RANDOM BEST 6:', list(sorted(rtrials.losses()))[:6]

        if bandit.name != 'n_arms':
            # -- assert that our threshold is meaningful
            assert min(rtrials.losses()) > thresh

        assert bandit.name is not None
        algo = partial(
            suggest_algos.ei,
            stop_at=self.thresholds[bandit.name])

        trials = Trials()
        fmin(fn=passthrough,
            space=self.bandit.expr,
            trials=trials,
            algo=algo,
            max_evals=LEN,
            rstate=np.random)
        assert len(trials) <= LEN


        if 0:
            plt.subplot(2, 2, 1)
            plt.scatter(range(LEN), trials.losses())
            plt.title('TPE losses')
            plt.subplot(2, 2, 2)
            plt.scatter(range(LEN), ([s['x'] for s in trials.specs]))
            plt.title('TPE x')
            plt.subplot(2, 2, 3)
            plt.title('RND losses')
            plt.scatter(range(LEN), rtrials.losses())
            plt.subplot(2, 2, 4)
            plt.title('RND x')
            plt.scatter(range(LEN), ([s['x'] for s in rtrials.specs]))
            plt.show()
        if 0:
            plt.hist(
                    [t['x'] for t in self.experiment.trials],
                    bins=20)


        #print trials.losses()
        print 'SUGGEST BEST 6:', list(sorted(trials.losses()))[:6]
        #logx = np.log([s['x'] for s in trials.specs])
        #print 'TPE MEAN', np.mean(logx)
        #print 'TPE STD ', np.std(logx)
        print 'Thresh', thresh
        assert min(trials.losses()) < thresh



