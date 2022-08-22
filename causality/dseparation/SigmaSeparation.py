import collections
import networkx as nx

from ..dseparation.AbstractGroundGraph import AbstractGroundGraph
from ..model import ParserUtil, RelationalValidity
from .sigma_helper import is_blocked_expand


class SigmaSeparation(object):

    def __init__(self, model):
        self.model = model
        self.perspectiveHopThresholdToAgg = {}

    def separated(self, hopThreshold, relVar1Strs, relVar2Strs, condRelVarStrs,
                   relationalVariableSetChecker=RelationalValidity.checkValidityOfRelationalVariableSet):
        """
        relVar1Strs, relVar2Strs, and condRelVarStrs are sequences of parseable RelationalVariable strings
        Method checks if, in model, are relVars1 and relVars2 d-separated? Constructs the abstract ground graph (AGG) for
        the model, and checks to see if all paths are d-separated.
        """
        if not isinstance(relVar1Strs, collections.Iterable) or not relVar1Strs:
            raise Exception("relVars1 must be a non-empty sequence of parseable RelationalVariable strings")
        relVars1 = {ParserUtil.parseRelVar(relVarStr) for relVarStr in relVar1Strs}

        if not isinstance(relVar2Strs, collections.Iterable) or not relVar2Strs:
            raise Exception("relVars2 must be a non-empty sequence of parseable RelationalVariable strings")
        relVars2 = {ParserUtil.parseRelVar(relVarStr) for relVarStr in relVar2Strs}

        if not isinstance(condRelVarStrs, collections.Iterable):
            raise Exception("condRelVars must be a sequence of parseable RelationalVariable strings")
        condRelVars = {ParserUtil.parseRelVar(condRelVar) for condRelVar in condRelVarStrs}

        # check consistency of all three relational variable sets (perspectives, hop threshold, against schema)
        relationalVariableSetChecker(self.model.schema, hopThreshold, relVars1 | relVars2 | condRelVars)

        perspective = list(relVars1)[0].getBaseItemName()
        if (perspective, hopThreshold) not in self.perspectiveHopThresholdToAgg:
            agg = AbstractGroundGraph(self.model, perspective, hopThreshold)
            self.perspectiveHopThresholdToAgg[(perspective, hopThreshold)] = agg
        else:
            agg = self.perspectiveHopThresholdToAgg[(perspective, hopThreshold)]

        # expand relVars1, relVars2, condRelVars with all intersection variables they subsume
        relVars1 = {relVar for relVar1 in relVars1 for relVar in agg.getSubsumedVariables(relVar1)}
        relVars2 = {relVar for relVar2 in relVars2 for relVar in agg.getSubsumedVariables(relVar2)}
        condRelVars = {relVar for condRelVar in condRelVars for relVar in agg.getSubsumedVariables(condRelVar)}

        relVars1 -= condRelVars
        relVars2 -= condRelVars

        if relVars1 & relVars2 != set():
            return False

        if not relVars1 or not relVars2:
            return True

        _agg = agg.copy()
        _agg.remove_nodes_from(list(nx.isolates(agg)))

        return is_blocked_expand(_agg, list(relVars1)[0], list(relVars2)[0], condRelVars, sep_type='sigma')