import pdb
import numbers
import logging
import itertools
import collections

from queue import Queue
from datetime import datetime

import networkx as nx

from ..model.RelationalDependency import RelationalVariable, RelationalDependency
from ..learning import EdgeOrientation, FCIEdgeOrientation
from ..model import ParserUtil
from ..model import RelationalValidity
from ..dseparation.AbstractGroundGraph import AbstractGroundGraph
from ..modelspace import RelationalSpace


logger = logging.getLogger(__name__)

class SchemaDependencyWrapper:

    def __init__(self, schema, dependencies):
        self.schema = schema
        self.dependencies = dependencies


class RelFCI(object):

    def __init__(self, schema, citest, hopThreshold, depth=None):
        if not isinstance(hopThreshold, numbers.Integral) or hopThreshold < 0:
            raise Exception("Hop threshold must be a non-negative integer: found {}".format(hopThreshold))
        if depth is not None and (not isinstance(depth, numbers.Integral) or depth < 0):
            raise Exception("Depth must be a non-negative integer or None: found {}".format(depth))

        self.schema = schema
        self.citest = citest
        self.hopThreshold = hopThreshold
        self.depth = depth
        self.perspectiveToAgg = None
        self.potentialDependencySorter = lambda l: l # no sorting by default
        self.potentialDependencySorterPDS = lambda l: sorted(l, key=lambda x: x.relVar1)
        self.generateSepsetCombinations = itertools.combinations
        self.undirectedDependencies = None
        self.orientedDependencies = None
        self.ciTestCache = {}
        self.pdsCache = {}
        self.ciRecord = {'Phase I': 0, 'Phase II': 0, 'Phase III': 0, 'total': 0}
        self.resetEdgeOrientationUsage()


    def updateSkeleton(self, potentialDeps, orderIndependentSkeleton=False, isPossibleDSep=False):
        self.maxDepthReached = -1
        if self.depth is None:
            self.depth = max([len(agg.nodes()) - 2 for agg in self.perspectiveToAgg.values()])
        logger.info("Number of potentialDeps %d", len(potentialDeps))
        remainingDeps = potentialDeps[:]
        currentDepthDependenciesToRemove = []
        # Check for independencies
        for conditioningSetSize in range(self.depth+1):
            self.maxDepthReached = conditioningSetSize
            testedAtCurrentSize = False
            logger.info("Conditioning set size %d", conditioningSetSize)
            logger.debug("remaining dependencies %s", remainingDeps)
            for potentialDep in potentialDeps:
                logger.debug("potential dependency %s", potentialDep)
                if potentialDep not in remainingDeps:
                    continue
                relVar1, relVar2 = potentialDep.relVar1, potentialDep.relVar2
                
                pds = self.findPossibleDsep(relVar1) if isPossibleDSep else None
                sepset, curTestedAtCurrentSize = self.findSepset(relVar1, relVar2, conditioningSetSize, PDS=pds, phaseForRecording='Phase I')
                testedAtCurrentSize = testedAtCurrentSize or curTestedAtCurrentSize
                
                if sepset is not None:
                    logger.debug("removing edge %s -- %s", relVar1, relVar2)
                    self.sepsets[relVar1, relVar2] = set(sepset)
                    self.sepsets[relVar2, relVar1] = set(sepset)
                    remainingDeps.remove(potentialDep)
                    potentialDepReverse = potentialDep.reverse()
                    remainingDeps.remove(potentialDepReverse)
                    if not orderIndependentSkeleton:
                        self.removeDependency(potentialDep)
                    else: # delay removal in underlying AGGs until after current depth
                        currentDepthDependenciesToRemove.append(potentialDep)
            if orderIndependentSkeleton:
                for potentialDep in currentDepthDependenciesToRemove:
                    self.removeDependency(potentialDep)
                currentDepthDependenciesToRemove = []
            if not testedAtCurrentSize: # exit early, no possible sepsets of a larger size
                break
            potentialDeps = remainingDeps[:]

        self.undirectedDependencies = remainingDeps
        logger.info("Undirected dependencies: %s", self.undirectedDependencies)
        logger.info(self.ciRecord)


    def identifyUndirectedDependencies(self, orderIndependentSkeleton=False):
        logger.info('Phase I: identifying undirected dependencies')
        # Create fully connected undirected AGG
        potentialDeps = RelationalSpace.getRelationalDependencies(self.schema, self.hopThreshold, includeExistence=False, isPAG=True)
        potentialDeps = self.potentialDependencySorter(potentialDeps)
        self.constructAggsFromDependencies(potentialDeps)

        # Keep track of separating sets
        self.sepsets = {}
        self.updateSkeleton(potentialDeps, orderIndependentSkeleton)
        FCIEdgeOrientation.applyColliderDetection(self)
        self.updateOrientedDependencies()
        logger.info("Oriented dependencies: %s", self.orientedDependencies)


    def finalizeUndirectedDependencies(self, orderIndependentSkeleton=False):
        logger.info('Phase II: finalizing undirected dependencies')

        potentialDeps = self.undirectedDependencies
        potentialDeps = self.potentialDependencySorterPDS(potentialDeps)

        self.updateSkeleton(potentialDeps, orderIndependentSkeleton, isPossibleDSep=True)
        self.resetOrientedDependencies()
        self.updateOrientedDependencies()
        FCIEdgeOrientation.applyColliderDetection(self)
        self.updateOrientedDependencies()
        logger.info("Oriented dependencies: %s", self.orientedDependencies)


    def findSepset(self, relVar1, relVar2, conditioningSetSize, PDS=None, phaseI=True, phaseForRecording='Phase I'):
        agg = self.perspectiveToAgg[relVar2.getBaseItemName()]
        neighborsMix2 = PDS
        if not neighborsMix2:
            neighborsMix2 = set(agg.predecessors(relVar2) + agg.successors(relVar2))
        neighbors2 = set()
        for neighbor in neighborsMix2:
            if isinstance(neighbor, RelationalVariable):
                if phaseI and len(neighbor.path) <= (self.hopThreshold+1):
                    neighbors2.add(neighbor)
                elif not phaseI:
                    neighbors2.add(neighbor)
                else:
                    continue
            else: # relational variable intersection, take both relational variable sources
                if phaseI and len(neighbor.relVar1.path) <= (self.hopThreshold+1) and \
                    len(neighbor.relVar2.path) <= (self.hopThreshold+1):
                    neighbors2.add(neighbor.relVar1)
                    neighbors2.add(neighbor.relVar2)
                elif not phaseI:
                    neighbors2.add(neighbor.relVar1)
                    neighbors2.add(neighbor.relVar2)
                else:
                    continue
        logger.debug("neighbors2 %s", neighbors2)
        if relVar1 in neighbors2:
            neighbors2.remove(relVar1)
        if relVar2 in neighbors2:
            neighbors2.remove(relVar2)
        testedAtCurrentSize = False
        if conditioningSetSize <= len(neighbors2):
            for candidateSepSet in self.generateSepsetCombinations(neighbors2, conditioningSetSize):
                logger.debug("checking %s _||_ %s | { %s }", relVar1, relVar2, candidateSepSet)
                testedAtCurrentSize = True
                ciTestKey = (relVar1, relVar2, tuple(sorted(list(candidateSepSet))))
                if ciTestKey not in self.ciTestCache:
                    self.ciRecord[phaseForRecording] += 1
                    depthStr = 'depth {}'.format(len(candidateSepSet))
                    if phaseI:
                        self.ciRecord.setdefault(depthStr, 0)
                        self.ciRecord[depthStr] += 1
                    self.ciRecord['total'] += 1
                    isCondInd = self.citest.isConditionallyIndependent(relVar1, relVar2, candidateSepSet)
                    self.ciTestCache[ciTestKey] = isCondInd
                else:
                    logger.debug("found result in CI cache")
                if self.ciTestCache[ciTestKey]:
                    return set(candidateSepSet), testedAtCurrentSize
        return None, testedAtCurrentSize


    def findPossibleDsep(self, node):
        agg = self.perspectiveToAgg[node.getBaseItemName()]
        graph = nx.Graph(agg)

        pdsKey = (node.getBaseItemName(), node)
        if pdsKey in self.pdsCache:
            logger.debug("found pdsKey in pds cache")
            return self.pdsCache[pdsKey]

        #TODO: try caching
        def isTriangle(graph, u, v, w):
            return graph.has_edge(u, v) and graph.has_edge(v, w) and graph.has_edge(u, w)

        def isCollider(agg, u, v, w):
            relDepUV = next(iter(agg[u][v][AbstractGroundGraph.UNDERLYING_DEPENDENCIES]))
            relDepVW = next(iter(agg[v][w][AbstractGroundGraph.UNDERLYING_DEPENDENCIES]))
            tailUV = relDepUV.tailMarkTo
            tailWV = relDepVW.tailMarkFrom
            return tailUV == RelationalDependency.TAIL_MARK_RIGHT_ARROW and tailWV == RelationalDependency.TAIL_MARK_LEFT_ARROW

        pds = set()

        for target in graph.nodes():
            if node == target:
                continue
            paths = list(nx.all_simple_paths(graph, source=node, target=target))
            for path in paths:
                if len(path) < 3:
                    continue
                ok = True
                for u,v,w in zip(path, path[1:], path[2:]):
                    if isTriangle(graph, u, v, w) or isCollider(agg, u, v, w):
                        continue
                    ok = False
                    break
                if ok:
                    pds.add(path[-1])
                    break
        self.pdsCache[pdsKey] = pds
        return pds


    def removeDependency(self, dependency):
        depReverse = dependency.reverse()
        FCIEdgeOrientation.propagateEdgeRemoval(self.perspectiveToAgg, [dependency, depReverse])


    def updateOrientedDependencies(self):
        self.orientedDependencies = set()
        for agg in self.perspectiveToAgg.values():
            for edge in agg.edges(data=True):
                for relDep in edge[2][AbstractGroundGraph.UNDERLYING_DEPENDENCIES]:
                    if relDep.tailMarkFrom != RelationalDependency.TAIL_MARK_CIRCLE or relDep.tailMarkTo != RelationalDependency.TAIL_MARK_CIRCLE:
                        self.orientedDependencies.add(relDep)


    def resetOrientedDependencies(self):
        for agg in self.perspectiveToAgg.values():
            for edge in agg.edges(data=True):
                for relDep in edge[2][AbstractGroundGraph.UNDERLYING_DEPENDENCIES]:
                    relDep.tailMarkFrom = RelationalDependency.TAIL_MARK_CIRCLE
                    relDep.tailMarkTo = RelationalDependency.TAIL_MARK_CIRCLE
                if (edge[1], edge[0]) not in agg.edges():
                    agg.add_edge(edge[1], edge[0])
                    self[edge[1]][edge[0]].setdefault(AbstractGroundGraph.UNDERLYING_DEPENDENCIES, edge[2][AbstractGroundGraph.UNDERLYING_DEPENDENCIES])
        

    def orientDependencies(self, rboOrder='normal'):
        """
        rboOrder can be one of {'normal', 'first', 'last'} which supports interleaving the RBO rule at different
        points during edge orientation.  Enables experiments to test unique contributions of RBO with respect to
        the other PC-like rules.
        """
        logger.info('Phase III: orienting dependencies')
        if not hasattr(self, 'undirectedDependencies') or self.undirectedDependencies is None:
            raise Exception("No undirected dependencies found. Try running Phase I first.")
        if not hasattr(self, 'sepsets') or self.sepsets is None:
            raise Exception("No sepsets found. Try running Phase I first.")

        # self.constructAggsFromDependencies(self.undirectedDependencies)

        if self.depth is None: # if it wasn't set in Phase I (e.g., manually set undirected dependencies)
            self.depth = max([len(agg.nodes()) - 2 for agg in self.perspectiveToAgg.values()])

        self.applyOrientationRules(rboOrder)
        self.updateOrientedDependencies()
        
        logger.info("Separating sets: %s", self.sepsets)
        logger.info("Oriented dependencies: %s", self.orientedDependencies)
        logger.info(self.ciRecord)
        logger.info(self.edgeOrientationRuleFrequency)


    def applyOrientationRules(self, rboOrder):
        if rboOrder == 'normal':
            FCIEdgeOrientation.applyColliderDetection(self, record=True)
            FCIEdgeOrientation.applyRBO(self)
            self.applySepsetFreeOrientationRules()
            self.applyFCIOrientationRules()
        elif rboOrder == 'first':
            FCIEdgeOrientation.applyRBO(self)
            FCIEdgeOrientation.applyColliderDetection(self, record=True)
            self.applySepsetFreeOrientationRules()
            self.applyFCIOrientationRules()
        elif rboOrder == 'last':
            FCIEdgeOrientation.applyColliderDetection(self, record=True)
            self.applySepsetFreeOrientationRules()
            FCIEdgeOrientation.applyRBO(self)
            self.applySepsetFreeOrientationRules()
            self.applyFCIOrientationRules()
        else:
            raise Exception("rboOrder must be one of 'normal', 'first', or 'last': found {!r}".format(rboOrder))


    def applySepsetFreeOrientationRules(self):
        newOrientationsFound = True
        while newOrientationsFound:
            newOrientationsFound = FCIEdgeOrientation.applyKnownNonColliders(self) or \
                                   FCIEdgeOrientation.applyCycleAvoidance(self) or \
                                   FCIEdgeOrientation.applyMR3(self)


    def applyFCIOrientationRules(self):
        newOrientationsFound = True
        while newOrientationsFound:
            newOrientationsFound =  FCIEdgeOrientation.applyFCIR4(self) or \
                                    FCIEdgeOrientation.applyFCIR8(self) or \
                                    FCIEdgeOrientation.applyFCIR9(self) or \
                                    FCIEdgeOrientation.applyFCIR9(self)
                                    # FCIEdgeOrientation.applyFCIR5(self) or \
                                    # FCIEdgeOrientation.applyFCIR6(self) or \
                                    # FCIEdgeOrientation.applyFCIR7(self) or \


    def findRecordAndReturnSepset(self, relVar1, relVar2):
        if (relVar1, relVar2) in self.sepsets:
            return self.sepsets[(relVar1, relVar2)]
        else:
            sepset = None
            logger.debug("findRecordAndReturnSepset for %s and %s", relVar1, relVar2)
            for conditioningSetSize in range(self.depth+1):
                sepset, testedAtCurrentSize = self.findSepset(relVar1, relVar2, conditioningSetSize,
                                                              phaseI=False, phaseForRecording='Phase III')
                if sepset is not None:
                    logger.debug("recording sepset %s", sepset)
                    self.sepsets[(relVar1, relVar2)] = sepset
                    self.sepsets[(relVar2, relVar1)] = sepset
                    break
                if not testedAtCurrentSize: # exit early, no other candidate sepsets to check
                    break
            return sepset


    def setUndirectedDependencies(self, undirectedDependencyStrs, dependencyChecker=RelationalValidity.checkRelationalDependencyValidity):
        if not isinstance(undirectedDependencyStrs, collections.Iterable):
            raise Exception("Undirected dependencies must be an iterable sequence of parseable RelationalDependency "
                            "strings: found {}".format(undirectedDependencyStrs))

        undirectedDependencies = [ParserUtil.parseRelDep(depStr) for depStr in undirectedDependencyStrs]
        # check each undirected dependency for consistency against the schema
        self.undirectedDependencies = []
        for undirectedDependency in undirectedDependencies:
            dependencyChecker(self.schema, undirectedDependency)
            undirectedDependency.tailMarkFrom = RelationalDependency.TAIL_MARK_CIRCLE
            undirectedDependency.tailMarkTo = RelationalDependency.TAIL_MARK_CIRCLE
            self.undirectedDependencies.append(undirectedDependency)
            self.undirectedDependencies.append(undirectedDependency.mirror())
        # self.undirectedDependencies = undirectedDependencies
        self.constructAggsFromDependencies(self.undirectedDependencies)


    def setSepsets(self, sepsets, relationalVariableSetChecker=RelationalValidity.checkValidityOfRelationalVariableSet):
        """
        Sets the sepsets internally.  Accepts string representation of the relational variables in the sepsets.
        """
        if not isinstance(sepsets, dict):
            raise Exception("Sepsets must be a dictionary: found {}".format(sepsets))

        self.sepsets = {(ParserUtil.parseRelVar(relVar1Str), ParserUtil.parseRelVar(relVar2Str)):
                         {ParserUtil.parseRelVar(condVarStr) for condVarStr in sepsetStr}
                         for (relVar1Str, relVar2Str), sepsetStr in sepsets.items()}

        for (relVar1, relVar2), condRelVars in self.sepsets.items():
            relationalVariableSetChecker(self.schema, self.hopThreshold, {relVar1, relVar2} | condRelVars)


    def constructAggsFromDependencies(self, dependencies):
        schemaDepWrapper = SchemaDependencyWrapper(self.schema, dependencies)
        perspectives = [si.name for si in self.schema.getSchemaItems()]
        self.perspectiveToAgg = {perspective: AbstractGroundGraph(schemaDepWrapper, perspective, 2*self.hopThreshold, isPAG=True)
                                      for perspective in perspectives}

    def constructTrueAggsFromDependencies(schema, dependencies, hopThreshold):
        schemaDepWrapper = SchemaDependencyWrapper(schema, dependencies)
        perspectives = [si.name for si in schema.getSchemaItems()]
        perspectiveToAgg = {perspective: AbstractGroundGraph(schemaDepWrapper, perspective, 2*hopThreshold, isPAG=True)
                                      for perspective in perspectives}
        return perspectiveToAgg


    def recordEdgeOrientationUsage(self, edgeOrientationName):
        self.edgeOrientationRuleFrequency[edgeOrientationName] += 1


    def resetEdgeOrientationUsage(self):
        self.edgeOrientationRuleFrequency = {'CD': 0, 'KNC': 0, 'CA': 0, 'MR3': 0, 'RBO': 0, \
            'FCIR4': 0, 'FCIR5': 0, 'FCIR6': 0, 'FCIR7': 0, 'FCIR8': 0, 'FCIR9': 0, 'FCIR10': 0}


    def report(self):
        return self.ciRecord, self.edgeOrientationRuleFrequency


    @staticmethod
    def runRelFCI(schema, citest, hopThreshold, depth=None):
        relfci = RelFCI(schema, citest, hopThreshold, depth)
        relfci.identifyUndirectedDependencies()
        relfci.finalizeUndirectedDependencies()
        relfci.orientDependencies()
        return relfci
