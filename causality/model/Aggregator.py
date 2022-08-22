from ..model import ParserUtil

class Aggregator(object):
    """
    Wrapper around relational variables, pairing with an aggregator function.  Each aggregator type is paired
    with a DataStore implementation that defines its behavior.
    """
    def __init__(self, relVarStr):
        self.relVar = ParserUtil.parseRelVar(relVarStr)


    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.relVar == other.relVar


    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.relVar)


class NoneAggregator(Aggregator):
    """
    Test Aggregator that always returns None.  Only used when aggregator is just a placeholder.
    Unsupported by DataStores.
    """
    pass


class IdentityAggregator(Aggregator):
    """
    Returns the input list without aggregating
    """
    pass


class NonRandomChoiceAggregator(Aggregator):
    """
    Test Aggregator that always returns the first value in the list
    """
    pass


class AverageAggregator(Aggregator):
    """
    Computes the average of the values in valueList. Returns None on empty
    """
    pass


class CountAggregator(Aggregator):
    """
    Counts the number of values in valueList. Returns 0 on empty (not None).
    """
    pass


class MaxAggregator(Aggregator):
    """
    Computes the maximum of the values in valueList. Returns None on empty.
    """
    pass


class CaseAggregator(Aggregator):
    """
    Inputs an aggregator and checks its value against the input cases.

    caseStatements: a list of 3-tuples of the form (CaseAggregator.OPERATOR, compareVal, returnVal)
    elseValue: the return value if no cases pass

    """
    EQUALS, NOT_EQUALS, LESS_THAN, LESS_THAN_OR_EQUAL, GREATER_THAN, GREATER_THAN_OR_EQUAL = range(6)     # fake enum via http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python

    OPERATOR_TO_NAME = {EQUALS: 'EQUALS', NOT_EQUALS: 'NOT_EQUALS', LESS_THAN: 'LESS_THAN',
                        LESS_THAN_OR_EQUAL: 'LESS_THAN_OR_EQUAL', GREATER_THAN: 'GREATER_THAN',
                        GREATER_THAN_OR_EQUAL: 'GREATER_THAN_OR_EQUAL'}


    def __init__(self, agg, caseStatements, elseValue=None):
        if not caseStatements:
            raise Exception("CaseAggregator requires at least one case statement")
        for caseStatement in caseStatements:
            operator = caseStatement[0]
            if not isinstance(operator, int) or not operator in CaseAggregator.OPERATOR_TO_NAME:
                raise Exception("Unknown operator {!r} in case statement {}".format(operator, caseStatement))
        self.agg = agg
        self.relVar = agg.relVar
        self.caseStatements = caseStatements
        self.elseValue = elseValue


    def __repr__(self):
        return "<{} {} {} {}: {}>".format(self.__class__.__name__, self.relVar, self.caseStatements, self.elseValue, self.agg)
