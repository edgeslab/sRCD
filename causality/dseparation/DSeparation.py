import collections
import networkx as nx

from ..dseparation.AbstractGroundGraph import AbstractGroundGraph
from ..model import ParserUtil, RelationalValidity


class DSeparation(object):

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

        return bfsReachability(relVars1, relVars2, condRelVars, agg)


def bfsReachability(relVars1, relVars2, condRelVars, agg):
    # Adapted from Algorithm 2 in "Identifying Independence in Bayesian Networks", Geiger, Verma, & Pearl 1990
    # Given a set of variables (relVars1, or X) and conditioning set (condRelVars, or Z),
    # identifies _all_ variables d-separated from X given Z using illegal link pairs and reachability
    # Exits early if finds a path that reaches a member of relVars2 (Y).

    # (i) Construct table that says whether each node is determined by the conditioning variables
    determined = condRelVars

    # (i) Construct table that says whether each node has a descendant that is a conditioning variables
    descendant = {}
    for condRelVar in condRelVars:
        descendant[condRelVar] = True
        for ancestor in agg.getAncestors(condRelVar):
            descendant[ancestor] = True

    # (ii) Construct directed graph where all edges have their reverse included
    undirectedAGG = nx.DiGraph()
    undirectedAGG.add_nodes_from(agg.nodes())
    for (aggNode1, aggNode2) in agg.edges_iter():
        undirectedAGG.add_edge(aggNode1, aggNode2)
        undirectedAGG.add_edge(aggNode2, aggNode1)

    # (i) Add node s and add edge from s to all nodes in sourceNodes (X)
    # Label those links with a 1, those nodes as reachable, and all other nodes as not reached
    undirectedAGG.add_node('source', reach=True)
    for node in relVars1:
        undirectedAGG.add_edge('source', node)
        undirectedAGG['source'][node]['label'] = 1

    # (ii) iteration:=1
    iteration = 1

    # (iii) Find all unlabeled links aggNode2 -> aggNode3 adjacent to at least one linke aggNode1 -> aggNode2
    # labeled iteration, such that (aggNode1 -> aggNode2, aggNode2 -> aggNode3) is legal. If no such link exists, stop.
    keepGoing = True
    while keepGoing:
        keepGoing = False
        for (aggNode1, aggNode2) in undirectedAGG.edges_iter():
            if 'label' in undirectedAGG[aggNode1][aggNode2] and undirectedAGG[aggNode1][aggNode2]['label'] == iteration:
                # for each _unlabeled_ neighbor of aggNode2 (except aggNode1)
                for aggNode3 in [neighbor for neighbor in undirectedAGG[aggNode2] if 'label' not in
                        undirectedAGG[aggNode2][neighbor] and not aggNode1 == neighbor]:
                    legal = False
                    # Legal if either aggNode2 is head-to-head node with descendant[aggNode2] = True
                    # OR
                    # aggNode2 is not a head-to-head node and determined[aggNode2] = False
                    if aggNode1 in agg and aggNode2 in agg[aggNode1] and aggNode3 in agg and aggNode2 in agg[aggNode3]:
                        if aggNode2 in descendant:
                            legal = True
                    else:
                        if not aggNode2 in determined:
                            legal = True

                    if legal:
                        if aggNode3 in relVars2:
                            return False
                        undirectedAGG[aggNode2][aggNode3]['label'] = iteration + 1
                        keepGoing = True
        iteration += 1

    return True
