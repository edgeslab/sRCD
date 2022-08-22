import numbers
import random
import numpy.random as numpy_random

class MarginalDistribution(object):

    def sample(self, ignoredValue):
        raise NotImplementedError()


# Note: Could be implemented as a degenerate case of discrete marginal or Equation-based distributions
class ConstantDistribution(MarginalDistribution):
    """
    A constant marginal distribution. Always returns the same value for each call.
    """

    def __init__(self, distConst):
        self.distConst = distConst


    def __eq__(self, other):
        return isinstance(other, ConstantDistribution) and self.distConst ==  other.distConst


    def __hash__(self):
        return hash(self.distConst)


    def sample(self, ignoredValue):
        return self.distConst


class ListDistribution(MarginalDistribution):
    """
    A marginal distribution for testing. Returns items in sequence from the passed list. Cycles to the start as needed.
    """

    def __init__(self, listVals):
        if not listVals:
            raise Exception("listVals cannot be []")

        self.listVals = listVals
        self.currentIndex = -1


    def sample(self, ignoredValue):
        self.currentIndex = self.currentIndex + 1 if self.currentIndex < len(self.listVals) - 1 else 0
        return self.listVals[self.currentIndex]


# Note: Could be implemented as a degenerate case of discrete marginal or Equation-based distributions
class GaussianMarginalDistribution(MarginalDistribution):

    def __init__(self, mu, sigma):
        if not isinstance(mu, numbers.Number):
            raise Exception("mu must be numeric but found {!r}".format(str(mu)))
        if not isinstance(sigma, numbers.Number):
            raise Exception("sigma must be numeric but found {!r}".format(str(sigma)))
        self.mu = mu
        self.sigma = sigma


    def sample(self, ignoredValue):
        return random.gauss(self.mu, self.sigma)


    def __repr__(self):
        return "<{}: (mu={}, sigma={})>".format(self.__class__.__name__, self.mu, self.sigma)


class UniformMarginalDistribution(MarginalDistribution):

    def __init__(self, lower, upper):
        if not isinstance(lower, numbers.Number):
            raise Exception("lower must be numeric but found {!r}".format(str(lower)))
        if not isinstance(upper, numbers.Number):
            raise Exception("upper must be numeric but found {!r}".format(str(upper)))
        if lower >= upper:
            raise Exception("lower must be less than upper")
        self.lower = lower
        self.upper = upper


    def sample(self, ignoredValue):
        return random.uniform(self.lower, self.upper)


class PoissonMarginalDistribution(MarginalDistribution):

    def __init__(self, lambdaMean):
        if not isinstance(lambdaMean, numbers.Number):
            raise Exception("lambda must be numeric but found {!r}".format(str(lambdaMean)))
        if lambdaMean <= 0:
            raise Exception("lambda must be greater than 0")
        self.lambdaMean = lambdaMean


    def sample(self, ignoredValue):
        return numpy_random.poisson(self.lambdaMean)


    def __repr__(self):
        return "<{}: (lambda={})>".format(self.__class__.__name__, self.lambdaMean)


class ExponentialMarginalDistribution(MarginalDistribution):

    def __init__(self, scale):
        if not isinstance(scale, numbers.Number):
            raise Exception("scale must be numeric but found {!r}".format(str(scale)))
        if scale <= 0:
            raise Exception("scale must be greater than 0")
        self.scale = scale


    def sample(self, ignoredValue):
        return numpy_random.exponential(self.scale)


    def __repr__(self):
        return "<{}: (scale={})>".format(self.__class__.__name__, self.scale)


class DiscreteMarginalDistribution(MarginalDistribution):
    """
    A discrete marginal distribution.  Will normalize the input probabilities to sum to 1.0.
    """

    def __init__(self, valueToProbability, randomNumGenerator=random.random):
        self.valueToProbability = valueToProbability
        # make sure there are no negative probabilities
        for value in self.valueToProbability.values():
            if value < 0.0:
                raise Exception("value {} < 0.0".format(value))
        # normalize probabilities
        total = sum(self.valueToProbability.values())
        if total <= 0.0:
            raise Exception("total {} <= 0.0".format(total))
        for value in self.valueToProbability:
            self.valueToProbability[value] /= total
        self.rng = randomNumGenerator


    def sample(self, ignoredValue):
        flip = self.rng()
        total = 0.0
        for value, prob in self.valueToProbability.items():
            if flip < total + prob:
                return value
            total += prob