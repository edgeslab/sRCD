import pdb
import logging
import itertools

import networkx as nx

from . import EdgeOrientation
from causality.graph.GraphUtil import _getRelDep, _isCirclePath, _isUncoveredPath, _isPotentiatllyDirectedPath, _isDiscriminatingPath

from causality.dseparation.AbstractGroundGraph import AbstractGroundGraph
from causality.model.RelationalDependency import RelationalVariable, RelationalDependency

logger = logging.getLogger(__name__)


EMPTY = RelationalDependency.TAIL_MARK_EMPTY
CIRCLE = RelationalDependency.TAIL_MARK_CIRCLE
LEFT_ARROW = RelationalDependency.TAIL_MARK_LEFT_ARROW
RIGHT_ARROW = RelationalDependency.TAIL_MARK_RIGHT_ARROW


#########################################################################################################################
####################################################### decorators ######################################################
#########################################################################################################################

def is_valid_rel_var_arg(num_vars):
    def is_valid_rel_var(func):
        def wrapper(*args, **kwargs):
            if not isinstance(args[1], RelationalVariable) or not isinstance(args[2], RelationalVariable):
                return False
            if num_vars == 3 and not isinstance(args[3], RelationalVariable):
                return False
            if num_vars == 4 and not isinstance(args[4], RelationalVariable):
                return False
            return func(*args, **kwargs)
        return wrapper
    return is_valid_rel_var

def is_valid_same_attr_arg(other):
    def is_valid_same_attr(func):
        def wrapper(*args, **kwargs):
            if args[1].attrName == args[other].attrName:
                return False
            return func(*args, **kwargs)
        return wrapper
    return is_valid_same_attr



#########################################################################################################################
####################################################### apply rules #####################################################
#########################################################################################################################


def applyColliderDetection(algo, record=None):
    newOrientationsFound = False
    for agg in algo.perspectiveToAgg.values():
        for relVar1, relVar2 in EdgeOrientation._findColliderDetectionCandidates(agg, algo.sepsets, _isValidCDCandidate, algo): 
            propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar1, relVar2), recurse=True, isTo=True, mark=RIGHT_ARROW)
            propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar2, relVar1), recurse=True, isTo=False, mark=LEFT_ARROW)
            if record:
                algo.recordEdgeOrientationUsage('CD')
            newOrientationsFound = True
    return newOrientationsFound


def applyRBO(algo):
    newOrientationsFound = False
    for agg in algo.perspectiveToAgg.values():
        for relVar1, relVar2 in _findRBORemovals(agg, algo):
            propagateEdgeRemoval(algo.perspectiveToAgg, _getRelDep(agg, relVar1, relVar2), recurse=True)
            propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar2, relVar1), recurse=True)
            propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar2, relVar1), recurse=True, isTo=False, mark=EMPTY)
            algo.recordEdgeOrientationUsage('RBO')
            newOrientationsFound = True
        for relVar1, relVar2 in _findRBOCandidates(agg, algo):
            propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar1, relVar2), recurse=True)
            propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar1, relVar2), recurse=True, isTo=False, mark=CIRCLE)
            propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar2, relVar1), recurse=True, mark=CIRCLE)
            propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar2, relVar1), recurse=True, isTo=False, mark=LEFT_ARROW)
            algo.recordEdgeOrientationUsage('RBO')
            newOrientationsFound = True
    return newOrientationsFound


def applyKnownNonColliders(algo):
    newOrientationsFound = False
    for agg in algo.perspectiveToAgg.values():
        for relVar1, relVar2 in _findKnownNonCollidersRemovals(agg):
            propagateEdgeRemoval(algo.perspectiveToAgg, _getRelDep(agg, relVar1, relVar2), recurse=True)
            propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar2, relVar1), recurse=True)
            propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar2, relVar1), recurse=True, isTo=False, mark=EMPTY)
            algo.recordEdgeOrientationUsage('KNC')
            logger.info("KNC Oriented edge: {node2}->{node3}".format(node2=relVar2, node3=relVar1))
            newOrientationsFound = True
    return newOrientationsFound


def applyCycleAvoidance(algo):
    newOrientationsFound = False
    for agg in algo.perspectiveToAgg.values():
        for relVar1, relVar2 in _findCycleAvoidanceCandidates(agg):
            if isinstance(relVar1, RelationalVariable) and isinstance(relVar2, RelationalVariable):
                propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar1, relVar2), recurse=True, isTo=True, mark=RIGHT_ARROW)
                propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar2, relVar1), recurse=True, isTo=False, mark=LEFT_ARROW)
                algo.recordEdgeOrientationUsage('CA')
                newOrientationsFound = True
    return newOrientationsFound


def applyMR3(algo):
    newOrientationsFound = False
    for agg in algo.perspectiveToAgg.values():
        for relVar1, relVar2 in _findMR3Candidates(agg):
            if isinstance(relVar1, RelationalVariable) and isinstance(relVar2, RelationalVariable):
                propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar1, relVar2), recurse=True, isTo=True, mark=RIGHT_ARROW)
                propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar2, relVar1), recurse=True, isTo=False, mark=LEFT_ARROW)
                algo.recordEdgeOrientationUsage('MR3')
                newOrientationsFound = True
    return newOrientationsFound


def applyFCIR4(algo):
    newOrientationsFound = False
    for agg in algo.perspectiveToAgg.values():
        for relVar1, relVar2 in _findFCIR4Removals(agg, algo):
            if isinstance(relVar1, RelationalVariable) and isinstance(relVar2, RelationalVariable):
                propagateEdgeRemoval(algo.perspectiveToAgg, _getRelDep(agg, relVar1, relVar2), recurse=True)
                propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar2, relVar1), recurse=True)
                propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar2, relVar1), recurse=True, isTo=False, mark=EMPTY)
                algo.recordEdgeOrientationUsage('FCIR4')
                newOrientationsFound = True
        # for relVar1, relVar2 in _findFCIR4Candidates(agg, algo):
        #     if isinstance(relVar1, RelationalVariable) and isinstance(relVar2, RelationalVariable):
        #         propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar1, relVar2), recurse=True)
        #         propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar2, relVar1), recurse=True)
        #         algo.recordEdgeOrientationUsage('FCIR4')
        #         newOrientationsFound = True
    return newOrientationsFound


def applyFCIR5(algo):
    newOrientationsFound = False
    for agg in algo.perspectiveToAgg.values():
        for relVar1, relVar2 in _findFCIR5Candidates(agg, algo):
            if isinstance(relVar1, RelationalVariable) and isinstance(relVar2, RelationalVariable):
                propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar1, relVar2), recurse=True, isTo=True, mark=EMPTY)
                propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar1, relVar2), recurse=True, isTo=False, mark=EMPTY)
                propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar2, relVar1), recurse=True, isTo=True, mark=EMPTY)
                propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar2, relVar1), recurse=True, isTo=False, mark=EMPTY)
                algo.recordEdgeOrientationUsage('FCIR5')
                newOrientationsFound = True
    return newOrientationsFound


def applyFCIR6(algo):
    newOrientationsFound = False
    for agg in algo.perspectiveToAgg.values():
        for relVar1, relVar2 in _findFCIR6Candidates(agg, algo):
            if isinstance(relVar1, RelationalVariable) and isinstance(relVar2, RelationalVariable):
                propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar1, relVar2), recurse=True, isTo=False, mark=EMPTY)
                propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar2, relVar1), recurse=True, isTo=True, mark=EMPTY)
                algo.recordEdgeOrientationUsage('FCIR6')
                newOrientationsFound = True
    return newOrientationsFound


def applyFCIR7(algo):
    newOrientationsFound = False
    for agg in algo.perspectiveToAgg.values():
        for relVar1, relVar2 in _findFCI7Candidates(agg):
            if isinstance(relVar1, RelationalVariable) and isinstance(relVar2, RelationalVariable):
                propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar1, relVar2), recurse=True, isTo=False, mark=EMPTY)
                propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar2, relVar1), recurse=True, isTo=True, mark=EMPTY)
                algo.recordEdgeOrientationUsage('FCIR7')
                newOrientationsFound = True
    return newOrientationsFound


def applyFCIR8(algo):
    newOrientationsFound = False
    for agg in algo.perspectiveToAgg.values():
        for relVar1, relVar2 in _findFCIR8Removals(agg, algo):
            if isinstance(relVar1, RelationalVariable) and isinstance(relVar2, RelationalVariable):
                propagateEdgeRemoval(algo.perspectiveToAgg, _getRelDep(agg, relVar1, relVar2), recurse=True)
                propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar2, relVar1), recurse=True)
                propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar2, relVar1), recurse=True, isTo=False, mark=EMPTY)
                algo.recordEdgeOrientationUsage('FCIR8')
                newOrientationsFound = True
    return newOrientationsFound


def applyFCIR9(algo):
    newOrientationsFound = False
    for agg in algo.perspectiveToAgg.values():
        for relVar1, relVar2 in _findFCIR9Removals(agg, algo):
            if isinstance(relVar1, RelationalVariable) and isinstance(relVar2, RelationalVariable):
                propagateEdgeRemoval(algo.perspectiveToAgg, _getRelDep(agg, relVar1, relVar2), recurse=True)
                propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar2, relVar1), recurse=True)
                propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar2, relVar1), recurse=True, isTo=False, mark=EMPTY)
                algo.recordEdgeOrientationUsage('FCIR9')
                newOrientationsFound = True
    return newOrientationsFound


def applyFCIR10(algo):
    newOrientationsFound = False
    for agg in algo.perspectiveToAgg.values():
        for relVar1, relVar2 in _findFCIR10Removals(agg, algo):
            if isinstance(relVar1, RelationalVariable) and isinstance(relVar2, RelationalVariable):
                propagateEdgeRemoval(algo.perspectiveToAgg, _getRelDep(agg, relVar1, relVar2), recurse=True)
                propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar2, relVar1), recurse=True)
                propagateEdgeOrientation(algo.perspectiveToAgg, _getRelDep(agg, relVar2, relVar1), recurse=True, isTo=False, mark=EMPTY)
                algo.recordEdgeOrientationUsage('FCIR9')
                newOrientationsFound = True
    return newOrientationsFound


#########################################################################################################################
################################################### candidate finding ###################################################
#########################################################################################################################


def _findKnownNonCollidersRemovals(graph):
    triples = _findKnownNonCollidersTriples(graph)
    for node1, node2, node3 in triples:
        if _isValidKnownNonCollider(graph, node1, node2, node3):
            if node2 in graph[node3] and node3 in graph[node2]: # 2-3 still undirected
                logger.debug("KNC candidate: %s, %s, %s", node1, node2, node3)
                yield node3, node2


def _findKnownNonCollidersTriples(graph):
    candidates = set() # set of node triples
    for node1 in graph.nodes():
        neighbors1 = set(graph.predecessors(node1) + graph.successors(node1))
        successors1 = graph.strictSuccessors(node1)
        for node2 in successors1:
            undirectedNeighbors2 = set(graph.predecessors(node2)) & set(graph.successors(node2))
            for node3 in undirectedNeighbors2:
                if node3 not in neighbors1:
                    candidates.add((node1, node2, node3))
    return candidates


def _findCycleAvoidanceCandidates(graph):
    triples = EdgeOrientation._findCycleAvoidanceTriples(graph)
    for node1, node2, node3 in triples:
        if _isValidCycleAvoidance(graph, node1, node2, node3):
            if node1 in graph[node3] and node3 in graph[node1]: # 1-3 still undirected
                yield node1, node3
                logger.info("CA Oriented edge: {node1}*->{node3}".format(node1=node1, node3=node3))


def _findMR3Candidates(graph):
    quartets = EdgeOrientation._findMR3Quartets(graph)
    for theta, beta, alpha, gamma in quartets:
        if _isValidMR3(graph, alpha, beta, gamma, theta):
            if beta in graph[theta] and theta in graph[beta]: # 1-2 still undirected
                logger.debug("MR3 candidate: {node1}, {node2}, {node3}, {node4}".format(
                    node1=alpha, node2=beta, node3=gamma, node4=theta))
                yield theta, beta
                logger.info("MR3 Oriented edge: {node1}*->{node2}".format(node1=theta, node2=beta))


def _findRBORemovals(graph, algo=None):
    for node1, node2, node3 in EdgeOrientation._findUnshieldedTriples(graph):
        args = [graph, node1, node2, node3, algo.sepsets]
        if algo:
            args.append(algo)
        if _isValidRBOCandidate(*args):
            if node2 in algo.sepsets[(node1, node3)] or \
                    any([node2.intersects(sepsetVar) for sepsetVar in algo.sepsets[(node1, node3)]]): # common cause
                if node3 in graph[node2] and node2 in graph[node3]: # 2-3 undirected
                    yield node3, node2
                    logger.info("RBO Oriented edge: {node2}->{node3}".format(node2=node2, node3=node3))


def _findRBOCandidates(graph, algo=None):
    for node1, node2, node3 in EdgeOrientation._findUnshieldedTriples(graph):
        args = [graph, node1, node2, node3, algo.sepsets]
        if algo:
            args.append(algo)
        if _isValidRBOCandidate(*args):
            if node2 in algo.sepsets[(node1, node3)] or \
                    any([node2.intersects(sepsetVar) for sepsetVar in algo.sepsets[(node1, node3)]]): # common cause
                pass
            else: # common effect
                if node3 in graph[node2] and node2 in graph[node3]: # 2-3 undirected
                    yield node3, node2
                    logger.info("RBO Oriented edge: {node3}->{node2}".format(node3=node3, node2=node2))


def _findFCIR4Quartets(graph):
    for beta in graph.nodes():
        undirectedNeighbors1 = set(graph.predecessors(beta)) & set(graph.successors(beta))
        if len(undirectedNeighbors1) > 1:
            
            paths = []
            for node in graph.nodes():
                if node == beta:
                    continue
                for p in nx.all_simple_paths(graph, beta, node):
                    if len(p) >= 3:
                        paths.append(p)

            for gamma in undirectedNeighbors1:
                relDep = next(iter(_getRelDep(graph, beta, gamma))) 
                if relDep.tailMarkFrom != CIRCLE:
                    continue
                for p in paths:
                    if p[1] == gamma or p[-1] == gamma or graph.has_edge(p[-1], gamma) or graph.has_edge(gamma, p[-1]):
                        continue
                    if not _isDiscriminatingPath(graph, p[::-1] + [gamma], beta):
                        continue
                    yield (p[-1], p[1], beta, gamma)


def _findFCIR4Removals(graph, algo):
    for theta, alpha, beta, gamma in _findFCIR4Quartets(graph):
        if _isValidFCIR4Candidate(graph, alpha, beta, gamma, algo.perspectiveToAgg, algo):
            if (theta, gamma) in algo.sepsets and beta in algo.sepsets[theta, gamma]:
                if beta in graph[gamma] and gamma in graph[beta]: # beta-gamma still undirected
                    logger.debug("FCIR4 candidate: {node1}, {node2}, {node3}, {node4}".format(
                        node1=theta, node2=alpha, node3=beta, node4=gamma))
                    yield gamma, beta
                    logger.info("FCIR4 Oriented edge: {node1}->{node2}".format(node1=beta, node2=gamma))
                #TODO: implement else case, check with arrows (confounder vs loop)


def _findFCIR4Candidates(graph, algo):
    for theta, alpha, beta, gamma in _findFCIR4Quartets(graph):
        if _isValidFCIR4Candidate(graph, alpha, beta, gamma, algo.perspectiveToAgg, algo):
            if ((theta, gamma) not in algo.sepsets or beta not in algo.sepsets[theta, gamma]) and\
                (beta not in algo.sepsets[gamma, theta] or beta not in algo.sepsets[gamma, theta]):
                logger.debug("FCIR4 candidate: {node1}, {node2}, {node3}, {node4}".format(
                        node1=theta, node2=alpha, node3=beta, node4=gamma))
                if alpha in graph[beta] and beta in graph[alpha]: # alpha-beta still undirected
                    yield alpha, beta
                    logger.info("FCIR4 Oriented edge: {node1}->{node2}".format(node1=beta, node2=gamma))
                if beta in graph[gamma] and gamma in graph[beta]: # beta-gamma still undirected
                    yield beta, gamma
                    logger.info("FCIR4 Oriented edge: {node1}->{node2}".format(node1=beta, node2=gamma))


def _findFCIR5Pairs(graph):
    for alpha, beta in graph.edges():
        relDep = next(iter(_getRelDep(graph, alpha, beta)))
        if relDep.tailMarkFrom != CIRCLE or relDep.tailMarkTo != CIRCLE:
            continue
        paths = nx.all_simple_paths(graph, alpha, beta)
        for p in paths:
            if len(p) < 4:
                continue
            if not _isCirclePath(graph, p):
                continue
            gamma, theta = p[1], p[-2]
            if graph.has_edge(alpha, theta) or graph.has_edge(theta, alpha) or \
                graph.has_edge(beta, gamma) or graph.has_edge(gamma, beta):
                continue
            for u,v in zip(p, p[1:]):
                yield u,v
            yield alpha, beta


def _findFCIR5Candidates(graph, algo):
    for node1, node2 in _findFCIR5Pairs(graph):
        if _isValidFCIR5Candidate(graph, node1, node2):
            logger.debug("FCIR5 candidate: {node1}, {node2}".format(node1=node1, node2=node2))
            if node1 in graph[node2] and node2 in graph[node1]: # node1-node2 still undirected
                yield node1, node2
                logger.info("FCIR5 Oriented edge: {node1}-{node2}".format(node1=node1, node2=node2))


def _findFCIR6Triples(graph):
    for node1 in graph.nodes():
        neighbors1 = set(graph.predecessors(node1) + graph.successors(node1))
        for node2 in neighbors1:
            neighbors2 = set(graph.predecessors(node2) + graph.successors(node2)) - {node1}
            for node3 in neighbors2:
                yield node1, node2, node3


def _findFCIR6Candidates(graph, algo):
    try:
        for alpha, beta, gamma in _findFCIR6Triples(graph):
            relDepAlphaBeta = next(iter(_getRelDep(graph, alpha, beta)))
            relDepBetaGamma = next(iter(_getRelDep(graph, beta, gamma)))
            if not relDepAlphaBeta or not relDepBetaGamma:
                continue
            if _isValidFCIR6Candidate(graph, alpha, beta, gamma):
                if beta in graph[gamma] and gamma in graph[beta]: # node1-node2 still undirected
                    yield beta, gamma
                    logger.info("FCIR6 Oriented edge: {node1}-*{node2}".format(node1=beta, node2=gamma))
    except:
        return


def _findFCI7Candidates(graph):
    for node1, node2, node3 in EdgeOrientation._findUnshieldedTriples(graph):
        if _isValidFCIR7Candidate(graph, node1, node2, node3):
            yield node2, node3
            logger.info("FCIR7 Oriented edge: {node2}-*{node3}".format(node3=node3, node2=node2))


def _findFCIR8Removals(graph, algo):
    for alpha, beta, gamma in _findFCIR6Triples(graph):
        if _isValidFCIR8Candidate(graph, alpha, beta, gamma):
            if alpha in graph[gamma] and gamma in graph[alpha]: # beta-gamma still undirected
                yield gamma, alpha
                logger.info("FCIR8 Oriented edge: {node1}->{node2}".format(node1=alpha, node2=gamma))


def _findFCIR9Pairs(graph):
    candidates = []
    for alpha in graph.nodes():
        paths = []
        for node in graph.nodes():
            if node == alpha:
                continue
            for p in nx.all_simple_paths(graph, alpha, node):
                if len(p) >= 4:
                    paths.append(p)

        for p in paths:
            gamma = p[-1]
            if not graph.has_edge(alpha, gamma) or not graph.has_edge(gamma, alpha):
                continue
            relDep = next(iter(_getRelDep(graph, alpha, gamma))) 
            if relDep.tailMarkFrom != CIRCLE or relDep.tailMarkTo != RIGHT_ARROW:
                continue
            if graph.has_edge(p[1], gamma) or graph.has_edge(gamma, p[1]):
                continue
            if not _isPotentiatllyDirectedPath(graph, p) or not _isUncoveredPath(graph, p):
                continue
            candidates.append((alpha, gamma))
    return candidates


def _findFCIR9Removals(graph, algo):
    for alpha, gamma in _findFCIR9Pairs(graph):
        if _isValidFCIR9Candidate(graph, alpha, gamma):
            logger.debug("FCIR9 candidate: {node1}, {node2}".format(node1=alpha, node2=gamma))
            yield gamma, alpha
            logger.info("FCIR9 Oriented edge: {node1}->{node2}".format(node1=alpha, node2=gamma))


def _findFCIR10Quartrets(graph):
    for beta, gamma, theta in EdgeOrientation._findUnshieldedTriples(graph):
        if beta in graph[gamma] or theta in graph[gamma]:   #already a collider
            continue
        for alpha in set(graph.predecessors(gamma)):
            if alpha == beta or alpha == theta:
                continue
            relDep = next(iter(_getRelDep(graph, alpha, gamma)))
            if relDep.tailMarkFrom == CIRCLE and relDep.tailMarkTo == RIGHT_ARROW:
                yield alpha, beta, gamma, theta


def _findFCIR10Removals(graph, algo):
    for alpha, beta, gamma, theta in _findFCIR10Quartrets(graph):
        for p1 in nx.all_simple_paths(graph, alpha, beta):
            if not _isPotentiatllyDirectedPath(graph, p1) or not _isUncoveredPath(graph, p1):
                continue
            for p2 in nx.all_simple_paths(graph, alpha, theta):
                if not _isPotentiatllyDirectedPath(graph, p1) or not _isUncoveredPath(graph, p1):
                    continue
                if p1[1] == p2[1]:
                    continue
                if graph.has_edge(p1[1], p2[1]) or graph.has_edge(p2[1], p1[1]):
                    continue
                if _isValidFCIR10Candidate(graph, alpha, beta, gamma, theta):
                    logger.debug("FCIR10 candidate: {node1}, {node2}".format(node1=alpha, node2=gamma))
                    yield gamma, alpha
                    logger.info("FCIR10 Oriented edge: {node1}->{node2}".format(node1=alpha, node2=gamma))


#########################################################################################################################
################################################## candidate validation #################################################
#########################################################################################################################


@is_valid_rel_var_arg(3)
@is_valid_same_attr_arg(3)
def _isValidCDCandidate(graph, relVar1, relVar2, relVar3, nodePairToSepset, algo=None):
    if len(relVar3.path) > 1:
        return False
    # Check if triple can still be oriented as a collider
    if not(relVar2 in graph[relVar1] and relVar2 in graph[relVar3] and
                (relVar1 in graph[relVar2] or relVar3 in graph[relVar2])):
        return False
    relDep1 = next(iter(_getRelDep(graph, relVar1, relVar2)))
    relDep2 = next(iter(_getRelDep(graph, relVar2, relVar3)))
    if relDep1.isFeedbackLoop() or relDep2.isFeedbackLoop():
        return False
    if relDep1.tailMarkTo == RIGHT_ARROW and relDep2.tailMarkFrom == LEFT_ARROW:
        return False
    logger.debug('CD candidate: %s, %s, %s', relVar1, relVar2, relVar3)
    sepset = algo.findRecordAndReturnSepset(relVar1, relVar3) if algo else nodePairToSepset
    return sepset is not None and relVar2 not in sepset and \
            all([not relVar2.intersects(sepsetVar) for sepsetVar in sepset])


@is_valid_rel_var_arg(3)
@is_valid_same_attr_arg(3)
def _isValidKnownNonCollider(graph, relVar1, relVar2, relVar3):
    if relVar1 in graph[relVar3] or relVar3 in graph[relVar1]:
        return False
    # Check if pair is already oriented
    if relVar2 not in graph[relVar3] or relVar3 not in graph[relVar2]:
        return False
    relDep1 = next(iter(_getRelDep(graph, relVar1, relVar2)))
    relDep2 = next(iter(_getRelDep(graph, relVar2, relVar3)))
    if relDep1.isFeedbackLoop() or relDep2.isFeedbackLoop():
        return False
    if relDep1.tailMarkTo != RIGHT_ARROW:
        return False
    if relDep2.tailMarkFrom != CIRCLE:
        return False
    return True


@is_valid_rel_var_arg(3)
@is_valid_same_attr_arg(3)
def _isValidCycleAvoidance(graph, relVar1, relVar2, relVar3):
    # Check if pair is already oriented
    if relVar1 not in graph[relVar3] or relVar3 not in graph[relVar1]:
        return False
    if relVar1 in graph[relVar2] and relVar2 in graph[relVar1] and \
        relVar2 in graph[relVar3] and relVar3 in graph[relVar2]:
        return False
    relDep1 = next(iter(_getRelDep(graph, relVar1, relVar2)))
    relDep2 = next(iter(_getRelDep(graph, relVar2, relVar3)))
    relDep3 = next(iter(_getRelDep(graph, relVar1, relVar3)))
    if relDep3.isFeedbackLoop() or relDep3.tailMarkTo != CIRCLE:
        return False
    if (relVar1 in graph[relVar2] and relDep2.tailMarkTo == RIGHT_ARROW) or \
        (relVar2 in graph[relVar3] and relDep1.tailMarkTo == RIGHT_ARROW):
        return True
    return False


@is_valid_rel_var_arg(4)
@is_valid_same_attr_arg(4)
def _isValidMR3(graph, alpha, beta, gamma, theta):
    # Check if pairs are already oriented
    if alpha not in graph[theta] or beta not in graph[theta] or gamma not in graph[theta]:
        return False
    alphaBeta = next(iter(_getRelDep(graph, alpha, beta)))
    betaGamma = next(iter(_getRelDep(graph, beta, gamma)))
    alphaTheta = next(iter(_getRelDep(graph, alpha, theta)))
    thetaGamma = next(iter(_getRelDep(graph, theta, gamma)))
    thetaBeta = next(iter(_getRelDep(graph, theta, beta)))
    if thetaBeta.isFeedbackLoop() or thetaBeta.tailMarkTo != CIRCLE:
        return False
    if alphaBeta.tailMarkTo != RIGHT_ARROW or betaGamma.tailMarkFrom != LEFT_ARROW:
        return False
    if alphaTheta.tailMarkTo != CIRCLE or thetaGamma.tailMarkFrom != CIRCLE:
        return False
    return True


@is_valid_rel_var_arg(3)
def _isValidRBOCandidate(graph, relVar1, relVar2, relVar3, ignoredSepset, algo=None):
    if len(relVar3.path) > 1:
        return False
    if relVar1.attrName != relVar3.attrName:
        return False
    # Check if triple is already oriented
    if relVar2 not in graph[relVar1] or relVar2 not in graph[relVar3] or \
                    relVar1 not in graph[relVar2] or relVar3 not in graph[relVar2]:
        return False
    relDep1 = next(iter(_getRelDep(graph, relVar1, relVar2)))
    relDep2 = next(iter(_getRelDep(graph, relVar2, relVar3)))
    if relDep1.tailMarkFrom != CIRCLE or relDep1.tailMarkTo != CIRCLE or \
        relDep2.tailMarkFrom != CIRCLE or relDep2.tailMarkTo != CIRCLE:
        return False
    logger.debug('RBO candidate: %s, %s, %s', relVar1, relVar2, relVar3)
    sepset = algo.findRecordAndReturnSepset(relVar1, relVar3) if algo else ignoredSepset
    return sepset is not None


@is_valid_rel_var_arg(3)
@is_valid_same_attr_arg(3)
def _isValidFCIR4Candidate(graph, relVar1, relVar2, relVar3, ignoredSepset, algo=None):
    # Check if triple is already oriented
    if relVar2 not in graph[relVar1] or relVar2 not in graph[relVar3] or \
                    relVar1 not in graph[relVar2] or relVar3 not in graph[relVar2]:
        return False
    relDep = next(iter(_getRelDep(graph, relVar2, relVar3)))
    if relDep.tailMarkFrom != CIRCLE:
        return False
    logger.debug('FCIR4 candidate: %s, %s, %s', relVar1, relVar2, relVar3)
    return True


@is_valid_rel_var_arg(2)
@is_valid_same_attr_arg(2)
def _isValidFCIR5Candidate(graph, relVar1, relVar2):
    # Check if pair is already oriented
    if relVar2 not in graph[relVar1] or relVar1 not in graph[relVar2]:
        return False
    relDep = next(iter(_getRelDep(graph, relVar1, relVar2)))
    if relDep.tailMarkFrom != CIRCLE or relDep.tailMarkTo != CIRCLE:
        return False
    logger.debug('FCIR5 candidate: %s, %s', relVar1, relVar2)
    return True


@is_valid_rel_var_arg(3)
@is_valid_same_attr_arg(3)
def _isValidFCIR6Candidate(graph, relVar1, relVar2, relVar3):
    # Check if pair is already oriented
    if relVar2 not in graph[relVar1] or relVar1 not in graph[relVar2]:
        return False
    relDep1 = next(iter(_getRelDep(graph, relVar1, relVar2)))
    if relDep1.tailMarkFrom != EMPTY or relDep1.tailMarkTo != EMPTY:
        return False
    relDep2 = next(iter(_getRelDep(graph, relVar2, relVar3)))
    if relDep2.tailMarkFrom != CIRCLE:
        return False
    logger.debug('FCIR6 candidate: %s, %s', relVar1, relVar2)
    return True


@is_valid_rel_var_arg(3)
@is_valid_same_attr_arg(3)
def _isValidFCIR7Candidate(graph, relVar1, relVar2, relVar3):
    relDep1 = next(iter(_getRelDep(graph, relVar1, relVar2)))
    relDep2 = next(iter(_getRelDep(graph, relVar2, relVar3)))
    if relDep1.isFeedbackLoop() or relDep2.isFeedbackLoop():
        return False
    if relDep1.tailMarkFrom != EMPTY or relDep1.tailMarkTo != CIRCLE:
        return False
    if relDep2.tailMarkFrom != CIRCLE:
        return False
    logger.debug('FCIR7 candidate: %s, %s', relVar1, relVar2)
    return True


@is_valid_rel_var_arg(3)
@is_valid_same_attr_arg(3)
def _isValidFCIR8Candidate(graph, relVar1, relVar2, relVar3):
    if relVar3 not in graph[relVar1] or relVar1 not in graph[relVar3]:
        return False
    if relVar2 in graph[relVar3] or relVar3 not in graph[relVar2]:
        return False
    relDep12 = next(iter(_getRelDep(graph, relVar1, relVar2)))
    relDep23 = next(iter(_getRelDep(graph, relVar2, relVar3)))
    relDep13 = next(iter(_getRelDep(graph, relVar1, relVar3)))
    if relDep12.isFeedbackLoop() or relDep23.isFeedbackLoop():
        return False
    if relDep12.tailMarkFrom != EMPTY or (relDep12.tailMarkTo != RIGHT_ARROW and relDep12.tailMarkTo != CIRCLE):
        return False
    if relDep23.tailMarkFrom != EMPTY or relDep23.tailMarkTo != RIGHT_ARROW:
        return False
    if relDep13.tailMarkFrom != CIRCLE or relDep13.tailMarkTo != RIGHT_ARROW:
        return False
    logger.debug('FCIR8 candidate: %s, %s', relVar1, relVar2)
    return True


@is_valid_rel_var_arg(2)
@is_valid_same_attr_arg(2)
def _isValidFCIR9Candidate(graph, relVar1, relVar2):
    # Check if pair is already oriented
    if relVar2 not in graph[relVar1] or relVar1 not in graph[relVar2]:
        return False
    relDep12 = next(iter(_getRelDep(graph, relVar1, relVar2)))
    if relDep12.tailMarkFrom != CIRCLE or relDep12.tailMarkTo != RIGHT_ARROW:
        return False
    logger.debug('FCIR9 candidate: %s, %s', relVar1, relVar2)
    return True


@is_valid_rel_var_arg(4)
@is_valid_same_attr_arg(4)
def _isValidFCIR10Candidate(graph, relVar1, relVar2, relVar3, relVar4):
    if relVar3 not in graph[relVar1] or relVar1 not in graph[relVar3]:
        return False
    if relVar2 in graph[relVar3] or relVar4 in graph[relVar3]:
        return False
    
    relDep13 = next(iter(_getRelDep(graph, relVar1, relVar3)))
    if relDep13.isFeedbackLoop():
        return False
    if relDep13.tailMarkFrom != CIRCLE or relDep13.tailMarkTo != RIGHT_ARROW:
        return False
    
    logger.debug('FCIR10 candidate: %s, %s', relVar1, relVar2)
    return True


#########################################################################################################################
################################################ orientation propagation ################################################
#########################################################################################################################


def propagateEdgeRemoval(perspectiveToAgg, underlyingRelDeps, recurse=False):
    underlyingRelDeps = set(underlyingRelDeps)
    for agg in perspectiveToAgg.values():
        for underlyingRelDep in underlyingRelDeps:
            otherUnderlyingRelDeps = agg.removeEdgesForDependency(underlyingRelDep)
            if recurse:
                propagateEdgeRemoval(perspectiveToAgg, otherUnderlyingRelDeps - underlyingRelDeps)


def propagateEdgeOrientation(perspectiveToAgg, underlyingRelDeps, recurse=False, isTo=True, mark=RIGHT_ARROW):
    underlyingRelDeps = set(underlyingRelDeps)
    for agg in perspectiveToAgg.values():
        for underlyingRelDep in underlyingRelDeps:
            otherUnderlyingRelDeps = agg.orientEdgesForDependency(underlyingRelDep, isTo, mark)
            if recurse:
                propagateEdgeOrientation(perspectiveToAgg, otherUnderlyingRelDeps - underlyingRelDeps)