import numbers
import logging
import itertools
import collections
from ..model.RelationalDependency import RelationalVariable
from ..learning import EdgeOrientation
from ..model import ParserUtil
from ..model import RelationalValidity
from ..dseparation.AbstractGroundGraph import AbstractGroundGraph
from ..modelspace import RelationalSpace
from ..dseparation.SigmaSeparation import SigmaSeparation

from causality.learning.RCD import RCD

logger = logging.getLogger(__name__)



class sRCD(RCD):

    def __init__(self, schema, citest, hopThreshold, depth=None):
        super(sRCD, self).__init__(schema, citest, hopThreshold, depth)
        self.CC = set()


    def orientDependencies(self, rboOrder='first'):
        """
        rboOrder can be one of {'normal', 'first', 'last'} which supports interleaving the RBO rule at different
        points during edge orientation.  Enables experiments to test unique contributions of RBO with respect to
        the other PC-like rules.
        """
        logger.info('Phase II: orienting dependencies')
        if not hasattr(self, 'undirectedDependencies') or self.undirectedDependencies is None:
            raise Exception("No undirected dependencies found. Try running Phase I first.")
        if not hasattr(self, 'sepsets') or self.sepsets is None:
            raise Exception("No sepsets found. Try running Phase I first.")

        # self.constructAggsFromDependencies(self.undirectedDependencies)

        if self.depth is None: # if it wasn't set in Phase I (e.g., manually set undirected dependencies)
            self.depth = max([len(agg.nodes()) - 2 for agg in self.perspectiveToAgg.values()])

        self.applyOrientationRules(rboOrder)

        learnedDependencies = set()
        self.orientedDependencies = set()
        for agg in self.perspectiveToAgg.values():
            for edge in agg.edges(data=True):
                for relDep in edge[2][AbstractGroundGraph.UNDERLYING_DEPENDENCIES]:
                    learnedDependencies.add(relDep)
        for relDep in learnedDependencies:
            if relDep.reverse() not in learnedDependencies:
                self.orientedDependencies.add(relDep)
        
        logger.info("Separating sets: %s", self.sepsets)
        logger.info("Oriented dependencies: %s", self.orientedDependencies)
        logger.info(self.ciRecord)
        logger.info(self.edgeOrientationRuleFrequency)
        

    def applyColliderDetection(self):
        newOrientationsFound = False
        for partiallyDirectedAgg in self.perspectiveToAgg.values():
            for relVar1, relVar2 in EdgeOrientation._findColliderDetectionRemovals(partiallyDirectedAgg,
                                                                                   self.sepsets,
                                                                                   self._isValidCDCandidate):
                if (relVar1, relVar2) not in self.CC:
                    self.propagateEdgeRemoval(partiallyDirectedAgg[relVar1][relVar2]
                        [AbstractGroundGraph.UNDERLYING_DEPENDENCIES], recurse=True)
                    self.recordEdgeOrientationUsage('CD')
                    newOrientationsFound = True
        return newOrientationsFound


    def applyKnownNonColliders(self):
        newOrientationsFound = False
        for partiallyDirectedAgg in self.perspectiveToAgg.values():
            for relVar1, relVar2 in EdgeOrientation._findKnownNonCollidersRemovals(partiallyDirectedAgg):
                if isinstance(relVar1, RelationalVariable) and isinstance(relVar2, RelationalVariable) and\
                    (relVar1, relVar2) not in self.CC:
                    self.propagateEdgeRemoval(partiallyDirectedAgg[relVar1][relVar2]
                        [AbstractGroundGraph.UNDERLYING_DEPENDENCIES], recurse=True)
                    self.recordEdgeOrientationUsage('KNC')
                    logger.info("KNC Oriented edge: {node2}->{node3}".format(node2=relVar2, node3=relVar1))
                    newOrientationsFound = True
        return newOrientationsFound

    
    def applyCycleAvoidance(self):
        newOrientationsFound = False
        for partiallyDirectedAgg in self.perspectiveToAgg.values():
            for relVar1, relVar2 in EdgeOrientation._findCycleAvoidanceRemovals(partiallyDirectedAgg):
                if isinstance(relVar1, RelationalVariable) and isinstance(relVar2, RelationalVariable) and\
                    (relVar1, relVar2) not in self.CC:
                    self.propagateEdgeRemoval(partiallyDirectedAgg[relVar1][relVar2]
                        [AbstractGroundGraph.UNDERLYING_DEPENDENCIES], recurse=True)
                    self.recordEdgeOrientationUsage('CA')
                    newOrientationsFound = True
        return newOrientationsFound


    def applyMR3(self):
        newOrientationsFound = False
        for partiallyDirectedAgg in self.perspectiveToAgg.values():
            for relVar1, relVar2 in EdgeOrientation._findMR3Removals(partiallyDirectedAgg):
                if isinstance(relVar1, RelationalVariable) and isinstance(relVar2, RelationalVariable) and\
                    (relVar1, relVar2) not in self.CC:
                    self.propagateEdgeRemoval(partiallyDirectedAgg[relVar1][relVar2]
                        [AbstractGroundGraph.UNDERLYING_DEPENDENCIES], recurse=True)
                    self.recordEdgeOrientationUsage('MR3')
                    newOrientationsFound = True
        return newOrientationsFound


    def _isValidRBOCandidate(self, graph, relVar1, relVar2, relVar3, ignoredSepset):
        if not isinstance(relVar1, RelationalVariable) or not isinstance(relVar2, RelationalVariable) or \
                not isinstance(relVar3, RelationalVariable):
            return False
        if relVar1.attrName != relVar3.attrName:
            return False
        if len(relVar3.path) > 1:
            return False
        # Check if triple is already oriented
        if relVar2 not in graph[relVar1] or relVar2 not in graph[relVar3] or \
                        relVar1 not in graph[relVar2] or relVar3 not in graph[relVar2]:
            return False
        logger.debug('RBO candidate: %s, %s, %s', relVar1, relVar2, relVar3)
        sepset = self.findRecordAndReturnSepset(relVar1, relVar3)
        if sepset is None:
            self.CC.add((relVar1, relVar2))
            self.CC.add((relVar2, relVar1))
            self.CC.add((relVar3, relVar2))
            self.CC.add((relVar2, relVar3))
            # import pdb
            # pdb.set_trace()
        return sepset is not None


    def runsRCD(schema, citest, hopThreshold, depth=None):
        rcd = sRCD(schema, citest, hopThreshold, depth)
        rcd.identifyUndirectedDependencies()
        rcd.orientDependencies()
        return rcd