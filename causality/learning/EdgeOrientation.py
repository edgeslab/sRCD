import logging
import itertools

from causality.dseparation.AbstractGroundGraph import AbstractGroundGraph
from causality.model.RelationalDependency import RelationalDependency

logger = logging.getLogger(__name__)


def applyColliderDetectionNaive(graph, nodePairToSepset):
    """
    Finds and orients unshielded colliders. Modifies the graph in place.
    Pattern: 1-2-3 and 1 is not a neighbor of 3
    """
    newOrientationsFound = False
    for edgeToRemove in _findColliderDetectionRemovals(graph, nodePairToSepset, _isValidCDCandidateNaive):
        graph.remove_edge(edgeToRemove[0], edgeToRemove[1])
        newOrientationsFound = True
    return newOrientationsFound


def _findColliderDetectionRemovals(graph, nodePairToSepset, isValidCDCandidate):
    for node1, node2, node3 in _findUnshieldedTriples(graph):#, nodePairToSepset):
        if isValidCDCandidate(graph, node1, node2, node3, nodePairToSepset):
            if node1 in graph[node2] and node2 in graph[node1]: # 1-2 undirected
                yield node2, node1
                logger.info("CD Oriented edge: {node1}->{node2}".format(node1=node1, node2=node2))
            if node3 in graph[node2] and node2 in graph[node3]: # 2-3 undirected
                yield node2, node3
                logger.info("CD Oriented edge: {node3}->{node2}".format(node3=node3, node2=node2))


def _findColliderDetectionCandidates(graph, nodePairToSepset, isValidCDCandidate, algo=None):
    for node1, node2, node3 in _findUnshieldedTriples(graph):#, nodePairToSepset):
        if isValidCDCandidate(graph, node1, node2, node3, nodePairToSepset, algo):
            if node1 in graph[node2] and node2 in graph[node1]: # 1-2 undirected
                yield node1, node2
                logger.info("CD Oriented edge: {node1}->{node2}".format(node1=node1, node2=node2))
            if node3 in graph[node2] and node2 in graph[node3]: # 2-3 undirected
                yield node3, node2
                logger.info("CD Oriented edge: {node3}->{node2}".format(node3=node3, node2=node2))


def _findUnshieldedTriples(graph):
    for node1 in graph.nodes():
        neighbors1 = set(graph.predecessors(node1) + graph.successors(node1))
        for node2 in neighbors1:
            neighbors2 = set(graph.predecessors(node2) + graph.successors(node2)) - {node1}
            for node3 in neighbors2:
                if node3 not in neighbors1:
                    yield node1, node2, node3


def _isValidCDCandidateNaive(graph, node1, node2, node3, nodePairToSepset):
    if not(node2 in graph[node1] and node2 in graph[node3] and (node1 in graph[node2] or node3 in graph[node2])):
        return False
    return (node1, node3) in nodePairToSepset and node2 not in nodePairToSepset[(node1, node3)]


def applyKnownNonColliders(graph):
    """
    Finds and orients known non-colliders. Modifies graph in place. Returns newOrientationsFound.
    NB: applyColliderDetection must be called first. If not, see below comment for changes.
    Pattern: 1->2-3 and 1 is not a neighbor of 3
    """
    newOrientationsFound = False
    for edgeToRemove in _findKnownNonCollidersRemovals(graph):
        graph.remove_edge(edgeToRemove[0], edgeToRemove[1])
        newOrientationsFound = True
    return newOrientationsFound


def _findKnownNonCollidersRemovals(graph):
    triples = _findKnownNonCollidersTriples(graph)
    for node1, node2, node3 in triples:
        if node2 in graph[node3] and node3 in graph[node2]: # 2-3 still undirected
            logger.debug("KNC candidate: %s, %s, %s", node1, node2, node3)
            yield node3, node2


def _findKnownNonCollidersCandidates(graph):
    triples = _findKnownNonCollidersTriples(graph)
    for node1, node2, node3 in triples:
        if node2 in graph[node3] and node3 in graph[node2]: # 2-3 still undirected
            logger.debug("KNC candidate: %s, %s, %s", node1, node2, node3)
            yield node2, node3


def _successors(graph, node):
    if isinstance(graph, AbstractGroundGraph):
        return graph.strictSuccessors(node)
    return set(graph.successors(node)) - set(graph.predecessors(node))


def _predecessors(graph, node):
    if isinstance(graph, AbstractGroundGraph):
        return graph.strictPredecessors(node)
    return set(set(graph.predecessors(node)) - set(graph.successors(node)))


def _findKnownNonCollidersTriples(graph):
    candidates = set() # set of node triples
    for node1 in graph.nodes():
        neighbors1 = set(graph.predecessors(node1) + graph.successors(node1))
        successors1 = sorted(_successors(graph, node1))
        for node2 in successors1:
            undirectedNeighbors2 = sorted(set(graph.predecessors(node2)) & set(graph.successors(node2)))
            for node3 in undirectedNeighbors2:
                if node3 not in neighbors1:
                    candidates.add((node1, node2, node3))
    return candidates


def applyCycleAvoidance(graph):
    """
    Finds and breaks potential cycles. Modifies graph in place. Returns newOrientationsFound.
    NB: applyColliderDetection must be called first.
    Pattern: 1->2->3, 1-3
    """
    newOrientationsFound = False
    for edgeToRemove in _findCycleAvoidanceRemovals(graph):
        graph.remove_edge(edgeToRemove[0], edgeToRemove[1])
        newOrientationsFound = True
    return newOrientationsFound


def _findCycleAvoidanceRemovals(graph):
    triples = _findCycleAvoidanceTriples(graph)
    for node1, node2, node3 in triples:
        logger.debug("CA candidate: {node1}, {node2}, {node3}".format(
                node1=node1, node2=node2, node3=node3))
        if node1 in graph[node3] and node3 in graph[node1]: # 1-3 still undirected
            yield node3, node1
            logger.info("CA Oriented edge: {node1}->{node3}".format(node1=node1, node3=node3))


def _findCycleAvoidanceCandidates(graph):
    triples = _findCycleAvoidanceTriples(graph)
    for node1, node2, node3 in triples:
        if node1 in graph[node3] and node3 in graph[node1]: # 1-3 still undirected
            yield node1, node3
            logger.info("CA Oriented edge: {node1}->{node3}".format(node1=node1, node3=node3))


def _findCycleAvoidanceTriples(graph):
    candidates = set() # set of node triples
    for node1 in graph.nodes():
        undirectedNeighbors1 = set(graph.predecessors(node1)) & set(graph.successors(node1))
        successors1 = _successors(graph, node1)
        for node2 in successors1:
            successors2 = _successors(graph, node2)
            for node3 in successors2:
                if node3 in undirectedNeighbors1:
                    candidates.add((node1, node2, node3))
    return candidates


def applyMR3(graph):
    """
    Modifies graph in place. Returns newOrientationsFound.
    NB: applyColliderDetection must be called first.
    Pattern: 1-3->2, 1-2, 1-4->2
    """
    newOrientationsFound = False
    for edgeToRemove in _findMR3Removals(graph):
        graph.remove_edge(edgeToRemove[0], edgeToRemove[1])
        newOrientationsFound = True
    return newOrientationsFound


def _findMR3Removals(graph):
    quartets = _findMR3Quartets(graph)
    for node1, node2, node3, node4 in quartets:
        if node1 in graph[node2] and node2 in graph[node1]: # 1-2 still undirected
            logger.debug("MR3 candidate: {node1}, {node2}, {node3}, {node4}".format(
                node1=node1, node2=node2, node3=node3, node4=node4))
            yield node2, node1
            logger.info("MR3 Oriented edge: {node1}->{node2}".format(node1=node1, node2=node2))


def _findMR3Candidates(graph):
    quartets = _findMR3Quartets(graph)
    for node1, node2, node3, node4 in quartets:
        if node1 in graph[node2] and node2 in graph[node1]: # 1-2 still undirected
            logger.debug("MR3 candidate: {node1}, {node2}, {node3}, {node4}".format(
                node1=node1, node2=node2, node3=node3, node4=node4))
            yield node1, node2
            logger.info("MR3 Oriented edge: {node1}->{node2}".format(node1=node1, node2=node2))


def _findMR3Quartets(graph):
    candidates = set() # set of node quartets
    for node1 in graph.nodes():
        undirectedNeighbors1 = set(graph.predecessors(node1)) & set(graph.successors(node1))
        if len(undirectedNeighbors1) >= 3:
            for node2 in undirectedNeighbors1:
                predecessors2 = _predecessors(graph, node2)
                predecessors2 = predecessors2 & undirectedNeighbors1
                for (node3, node4) in itertools.combinations(predecessors2, 2):
                    neighbors3 = set(graph.predecessors(node3) + graph.successors(node3))
                    if node4 not in neighbors3:
                        candidates.add((node1, node2, node3, node4))
                        break
    return candidates


def _findRBORemovals(graph, nodePairToSepset, isValidRBOCandidate, algo=None):
    for node1, node2, node3 in _findUnshieldedTriples(graph):
        args = [graph, node1, node2, node3, nodePairToSepset]
        if algo:
            args.append(algo)
        if isValidRBOCandidate(*args):
            if node2 in nodePairToSepset[(node1, node3)] or \
                    any([node2.intersects(sepsetVar) for sepsetVar in nodePairToSepset[(node1, node3)]]): # common cause
                if node3 in graph[node2] and node2 in graph[node3]: # 2-3 undirected
                    yield node3, node2
                    logger.info("RBO Oriented edge: {node2}->{node3}".format(node2=node2, node3=node3))
            else: # common effect
                if node3 in graph[node2] and node2 in graph[node3]: # 2-3 undirected
                    yield node2, node3
                    logger.info("RBO Oriented edge: {node3}->{node2}".format(node3=node3, node2=node2))


def _findRBOCandidates(graph, nodePairToSepset, isValidRBOCandidate, isPAG=False):
    for node1, node2, node3 in _findUnshieldedTriples(graph):
        if isValidRBOCandidate(graph, node1, node2, node3, nodePairToSepset):
            if node2 in nodePairToSepset[(node1, node3)] or \
                    any([node2.intersects(sepsetVar) for sepsetVar in nodePairToSepset[(node1, node3)]]): # common cause
                if node3 in graph[node2] and node2 in graph[node3]: # 2-3 undirected
                    yield node2, node3
                    logger.info("RBO Oriented edge: {node2}->{node3}".format(node2=node2, node3=node3))
            else: # common effect
                if not isPAG:
                    if node3 in graph[node2] and node2 in graph[node3]: # 2-3 undirected
                        yield node3, node2
                        logger.info("RBO Oriented edge: {node3}->{node2}".format(node3=node3, node2=node2))