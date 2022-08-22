import numpy as np
import networkx as nx

from ..model.RelationalDependency import RelationalVariable, RelationalDependency
from ..dseparation.AbstractGroundGraph import AbstractGroundGraph


EMPTY = RelationalDependency.TAIL_MARK_EMPTY
CIRCLE = RelationalDependency.TAIL_MARK_CIRCLE
LEFT_ARROW = RelationalDependency.TAIL_MARK_LEFT_ARROW
RIGHT_ARROW = RelationalDependency.TAIL_MARK_RIGHT_ARROW



def isPossibleParent(agg, trueAgg=None):
    A = np.zeros((len(agg), len(agg)), dtype=int)
    nodes = sorted(agg.nodes())
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            u,v = nodes[i], nodes[j]
            if u == v:
                continue
            if trueAgg and not trueAgg.has_edge(u, v):
                continue
            if not agg.has_edge(u, v):
                continue
            relDep = next(iter(agg[u][v][AbstractGroundGraph.UNDERLYING_DEPENDENCIES]))
            if relDep.tailMarkFrom != RelationalDependency.TAIL_MARK_LEFT_ARROW:
                A[i][j] = 1
            elif relDep.tailMarkTo == RelationalDependency.TAIL_MARK_RIGHT_ARROW:
                A[i][j] = 1
    return A


def isNoParent(agg):
    A = isPossibleParent(agg)
    return np.ones(A.shape) - A


def isPossibleAncestor(agg, trueAgg=None, isPAG=False):
    A = np.zeros((len(agg), len(agg)), dtype=int)
    nodes = sorted(agg.nodes())
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            u,v = nodes[i], nodes[j]
            if u == v:
                continue
            if nx.has_path(agg, u, v):
                A[i][j] = 1
            # for path in nx.all_simple_paths(agg, u, v):
            #     if isPAG and not _isPotentiatllyDirectedPath(agg, path):
            #         continue
            #     A[i][j] = 1
            #     break
    return A


def isNoAncestor(agg):
    A = isPossibleAncestor(agg)
    return np.ones(A.shape, dtype=int) - A







def _getRelDep(agg, relVar1, relVar2):
    if relVar1 not in agg or relVar2 not in agg[relVar1]:
        return set()
    return sorted(agg[relVar1][relVar2][AbstractGroundGraph.UNDERLYING_DEPENDENCIES])


def _isUncoveredPath(graph, path):
    if len(path) < 3:
        return True
    for u,v,w in zip(path, path[1:], path[2:]):
        if graph.has_edge(u, w) or graph.has_edge(w, u):
            return False
    return True


def _isCirclePath(graph, path):
    for u,v in zip(path, path[1:]):
        relDep = next(iter(_getRelDep(graph, u, v)))
        if relDep.tailMarkFrom != CIRCLE or relDep.tailMarkTo != CIRCLE:
            return False
    return True


def _isPotentiatllyDirectedPath(graph, path):
    for u,v in zip(path, path[1:]):
        relDep = next(iter(_getRelDep(graph, u, v)))
        if relDep.isFeedbackLoop() or (relDep.tailMarkFrom == LEFT_ARROW or relDep.tailMarkTo == EMPTY):
            return False
    return True


def _isDiscriminatingPath(graph, path, node):
    if len(path) < 4:
        return False
    x,v,y = path[0],path[-2],path[-1]
    if v != node:
        return False
    if graph.has_edge(x, y) or graph.has_edge(y, x):
        return False    

    for a,b,c in zip(path[:-1], path[1:-1], path[2:-1]):
        if not isinstance(a, RelationalVariable) or \
            not isinstance(b, RelationalVariable) or \
                not isinstance(c, RelationalVariable):
                return False
        relDepAB = next(iter(_getRelDep(graph, a, b)))
        relDepBC = next(iter(_getRelDep(graph, b, c)))
        if not relDepAB or not relDepBC:
            return False
        tailAB = relDepAB.tailMarkTo
        tailBC = relDepBC.tailMarkFrom
        if tailAB != RIGHT_ARROW or tailBC != LEFT_ARROW:
            return False

        if not graph.has_edge(b, y):
            return False
        relDepBY = next(iter(_getRelDep(graph, b, y)))
        if (not relDepBY) or (relDepBY.tailMarkTo != RIGHT_ARROW):
            return False

    return True