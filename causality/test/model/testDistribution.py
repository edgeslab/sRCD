import unittest
import random
import numpy.random as numpy_random
from collections import OrderedDict, defaultdict
from causality.test import TestUtil
from causality.model.Distribution import ConstantDistribution
from causality.model.Distribution import DiscreteMarginalDistribution
from causality.model.Distribution import ExponentialMarginalDistribution
from causality.model.Distribution import GaussianMarginalDistribution
from causality.model.Distribution import ListDistribution
from causality.model.Distribution import MarginalDistribution
from causality.model.Distribution import PoissonMarginalDistribution
from causality.model.Distribution import UniformMarginalDistribution

class TestDistribution(unittest.TestCase):

    def testConstantDistribution(self):
        distribution = ConstantDistribution(0)
        self.assertEqual(0, distribution.sample(None))

        distribution = ConstantDistribution('red')
        self.assertEqual('red', distribution.sample(None))

        distribution = ConstantDistribution(1.5)
        self.assertEqual(1.5, distribution.sample(None))


    def testListDistribution(self):
        distribution = ListDistribution([1])
        self.assertEqual(1, distribution.sample(None))

        distribution = ListDistribution([2])
        self.assertEqual(2, distribution.sample(None))

        distribution = ListDistribution([1, 2])
        self.assertEqual(1, distribution.sample(None))
        self.assertEqual(2, distribution.sample(None))
        self.assertEqual(1, distribution.sample(None))

        # check [] errors
        TestUtil.assertRaisesMessage(self, Exception, "listVals cannot be []",
            ListDistribution, [])


    def testGaussianMarginalDistribution(self):
        gmd = GaussianMarginalDistribution(0, 1)
        self.assertTrue(isinstance(gmd, MarginalDistribution))
        randomSeed = random.random()
        random.seed(randomSeed)
        actualSample = [gmd.sample(None) for _ in range(100)]

        random.seed(randomSeed)
        expectedSample = [random.gauss(0, 1) for _ in range(100)]

        self.assertListEqual(expectedSample, actualSample)

        # test error: bad mu and/or sigma
        TestUtil.assertRaisesMessage(self, Exception, "mu must be numeric but found 'xx'", GaussianMarginalDistribution, 'xx', 1)
        TestUtil.assertRaisesMessage(self, Exception, "sigma must be numeric but found 'xx'", GaussianMarginalDistribution, 0, 'xx')


    def testUniformMarginalDistribution(self):
        umd = UniformMarginalDistribution(0, 1)
        self.assertTrue(isinstance(umd, MarginalDistribution))
        randomSeed = random.random()
        random.seed(randomSeed)
        actualSample = [umd.sample(None) for _ in range(100)]

        random.seed(randomSeed)
        expectedSample = [random.uniform(0, 1) for _ in range(100)]

        self.assertListEqual(expectedSample, actualSample)

        # test error: bad lower and upper parameters
        TestUtil.assertRaisesMessage(self, Exception, "lower must be numeric but found 'xx'", UniformMarginalDistribution, 'xx', 0)
        TestUtil.assertRaisesMessage(self, Exception, "upper must be numeric but found 'xx'", UniformMarginalDistribution, 0, 'xx')
        TestUtil.assertRaisesMessage(self, Exception, "lower must be less than upper", UniformMarginalDistribution, 1, 0)


    def testPoissonMarginalDistribution(self):
        pmd = PoissonMarginalDistribution(2.5)
        self.assertTrue(isinstance(pmd, MarginalDistribution))
        # numpy's seed requires an iteger
        randomSeed = int(numpy_random.random())
        numpy_random.seed(randomSeed)
        actualSample = [pmd.sample(None) for _ in range(100)]

        numpy_random.seed(randomSeed)
        expectedSample = [numpy_random.poisson(2.5) for _ in range(100)]

        self.assertListEqual(expectedSample, actualSample)

        # test error: bad lambda (mean) parameter
        TestUtil.assertRaisesMessage(self, Exception, "lambda must be numeric but found 'xx'", PoissonMarginalDistribution, 'xx')
        TestUtil.assertRaisesMessage(self, Exception, "lambda must be greater than 0", PoissonMarginalDistribution, -1)


    def testExponentialMarginalDistribution(self):
        emd = ExponentialMarginalDistribution(2.0)
        self.assertTrue(isinstance(emd, MarginalDistribution))
        # numpy's seed requires an iteger
        randomSeed = int(numpy_random.random())
        numpy_random.seed(randomSeed)
        actualSample = [emd.sample(None) for _ in range(100)]

        numpy_random.seed(randomSeed)
        expectedSample = [numpy_random.exponential(2.0) for _ in range(100)]

        self.assertListEqual(expectedSample, actualSample)

        # test error: bad lambda (mean) parameter
        TestUtil.assertRaisesMessage(self, Exception, "scale must be numeric but found 'xx'", ExponentialMarginalDistribution, 'xx')
        TestUtil.assertRaisesMessage(self, Exception, "scale must be greater than 0", ExponentialMarginalDistribution, -1)


    def sampleTestRange(self, distribution, valToExpectedCount):
        valToActualCount = defaultdict(int)
        for flip in range(100):
            distribution.rng = lambda: flip / 100
            valToActualCount[distribution.sample(None)] += 1
        for val in valToExpectedCount:
            self.assertEqual(valToExpectedCount[val], valToActualCount[val])


    def testDiscreteMarginalDistribution(self):
        self.sampleTestRange(DiscreteMarginalDistribution({0: 1.0, 1: 0.0}), {0: 100, 1: 0})
        self.sampleTestRange(DiscreteMarginalDistribution({0: 0.0, 1: 1.0}), {0: 0, 1: 100})

        # Using ordered dicts so that we can ensure the sample() output (regular dicts have non-deterministic order)
        self.sampleTestRange(DiscreteMarginalDistribution(OrderedDict({0: 0.3, 1: 0.7})), {0: 30, 1: 70})
        self.sampleTestRange(DiscreteMarginalDistribution(OrderedDict({0: 0.3, 1.5: 0.3, 'red': 0.4})), {0: 30, 1.5: 30, 'red': 40})

        # test boundaries: What if probabilities sum to less than or more than 1.0
        self.sampleTestRange(DiscreteMarginalDistribution(OrderedDict({0: 0.3, 1: 0.3, 2: 0.3})), {0: 34, 1: 33, 2: 33})
        self.sampleTestRange(DiscreteMarginalDistribution(OrderedDict({0: 3, 1: 4, 2: 5})), {0: 25, 1: 34, 2: 41})

        # test bad input: Probabilities that sum to <= 0.0 or negative probabilities
        TestUtil.assertRaisesMessage(self, Exception, "total 0.0 <= 0.0",
            DiscreteMarginalDistribution, {0: 0.0, 1: 0.0})
        TestUtil.assertRaisesMessage(self, Exception, "value -1 < 0.0",
            DiscreteMarginalDistribution, {0: -1, 1: 0.0})
        TestUtil.assertRaisesMessage(self, Exception, "value -1 < 0.0",
            DiscreteMarginalDistribution, {0: -1, 1: 2})


if __name__ == '__main__':
    unittest.main()
