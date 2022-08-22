import pdb
import numbers
import logging
import itertools
import collections

from ..learning import EdgeOrientation
from ..model import ParserUtil
from ..model import RelationalValidity
from ..modelspace import RelationalSpace
from ..model.RelationalDependency import RelationalVariable
from ..dseparation.AbstractGroundGraph import AbstractGroundGraph
from causality.dseparation.sigma_helper import SCC


logger = logging.getLogger(__name__)

class SchemaDependencyWrapper:

    def __init__(self, schema, dependencies):
        self.schema = schema
        self.dependencies = dependencies


class RCD(object):

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
        self.generateSepsetCombinations = itertools.combinations
        self.undirectedDependencies = None
        self.orientedDependencies = None
        self.ciTestCache = {}
        self.ciRecord = {'Phase I': 0, 'Phase II': 0, 'total': 0}
        self.resetEdgeOrientationUsage()


    def identifyUndirectedDependencies(self, orderIndependentSkeleton=False):
        logger.info('Phase I: identifying undirected dependencies')
        # Create fully connected undirected AGG
        potentialDeps = RelationalSpace.getRelationalDependencies(self.schema, self.hopThreshold, includeExistence=False)
        potentialDeps = self.potentialDependencySorter(potentialDeps)
        self.constructAggsFromDependencies(potentialDeps)

        # Keep track of separating sets
        self.sepsets = collections.OrderedDict()

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
                sepset, curTestedAtCurrentSize = self.findSepset(relVar1, relVar2, conditioningSetSize,
                                                                 phaseForRecording='Phase I')
                testedAtCurrentSize = testedAtCurrentSize or curTestedAtCurrentSize
                if sepset is not None:
                    logger.debug("removing edge %s -- %s", relVar1, relVar2)
                    self.sepsets[relVar1, relVar2] = set(sepset)
                    self.sepsets[relVar2, relVar1] = set(sepset)
                    remainingDeps.remove(potentialDep)
                    potentialDepReverse = potentialDep.reverse()
                    if potentialDep != potentialDepReverse and potentialDepReverse in remainingDeps:
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
        # logger.info("EDGES")
        # for edge in self.perspectiveToAgg['B'].edges():
        #     # if not isinstance(edge[0], RelationalVariableIntersection) and not isinstance(edge[1], RelationalVariableIntersection):
        #     logger.info(edge)


    def findSepset(self, relVar1, relVar2, conditioningSetSize, phaseI=True, phaseForRecording='Phase I'):
        agg = self.perspectiveToAgg[relVar2.getBaseItemName()]
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


    def removeDependency(self, dependency):
        depReverse = dependency.reverse()
        self.propagateEdgeRemoval([dependency, depReverse], force=True)


    def orientDependencies(self, rboOrder='normal'):
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

        self.orientedDependencies = set()
        for agg in self.perspectiveToAgg.values():
            for edge in agg.edges(data=True):
                if agg.has_edge(edge[1], edge[0]):
                    continue
                for relDep in edge[2][AbstractGroundGraph.UNDERLYING_DEPENDENCIES]:
                    self.orientedDependencies.add(relDep)
        
        logger.info("Separating sets: %s", self.sepsets)
        logger.info("Oriented dependencies: %s", self.orientedDependencies)
        logger.info(self.ciRecord)
        logger.info(self.edgeOrientationRuleFrequency)


    def applyOrientationRules(self, rboOrder):
        if rboOrder == 'normal':
            self.applyColliderDetection()
            self.applyRBO()
            self.applySepsetFreeOrientationRules()
        elif rboOrder == 'first':
            self.applyRBO()
            self.applyColliderDetection()
            self.applySepsetFreeOrientationRules()
        elif rboOrder == 'last':
            self.applyColliderDetection()
            self.applySepsetFreeOrientationRules()
            self.applyRBO()
            self.applySepsetFreeOrientationRules()
        else:
            raise Exception("rboOrder must be one of 'normal', 'first', or 'last': found {!r}".format(rboOrder))


    def applySepsetFreeOrientationRules(self):
        newOrientationsFound = True
        while newOrientationsFound:
            newOrientationsFound = self.applyKnownNonColliders() or \
                                   self.applyCycleAvoidance() or \
                                   self.applyMR3()


    def applyColliderDetection(self):
        newOrientationsFound = False
        for partiallyDirectedAgg in self.perspectiveToAgg.values():
            for relVar1, relVar2 in EdgeOrientation._findColliderDetectionRemovals(partiallyDirectedAgg,
                                                                                   self.sepsets,
                                                                                   self._isValidCDCandidate):
                self.propagateEdgeRemoval(partiallyDirectedAgg[relVar1][relVar2]
                    [AbstractGroundGraph.UNDERLYING_DEPENDENCIES], recurse=True)
                self.recordEdgeOrientationUsage('CD')
                newOrientationsFound = True
        return newOrientationsFound


    def applyRBO(self):
        newOrientationsFound = False
        for partiallyDirectedAgg in self.perspectiveToAgg.values():
            for relVar1, relVar2 in EdgeOrientation._findRBORemovals(partiallyDirectedAgg,
                                                                                   self.sepsets,
                                                                                   self._isValidRBOCandidate):
                self.propagateEdgeRemoval(partiallyDirectedAgg[relVar1][relVar2]
                    [AbstractGroundGraph.UNDERLYING_DEPENDENCIES], recurse=True)
                self.recordEdgeOrientationUsage('RBO')
                newOrientationsFound = True
        return newOrientationsFound


    def applyKnownNonColliders(self):
        newOrientationsFound = False
        for partiallyDirectedAgg in self.perspectiveToAgg.values():
            for relVar1, relVar2 in EdgeOrientation._findKnownNonCollidersRemovals(partiallyDirectedAgg):
                if isinstance(relVar1, RelationalVariable) and isinstance(relVar2, RelationalVariable):
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
                if isinstance(relVar1, RelationalVariable) and isinstance(relVar2, RelationalVariable):
                    self.propagateEdgeRemoval(partiallyDirectedAgg[relVar1][relVar2]
                        [AbstractGroundGraph.UNDERLYING_DEPENDENCIES], recurse=True)
                    self.recordEdgeOrientationUsage('CA')
                    newOrientationsFound = True
        return newOrientationsFound


    def applyMR3(self):
        newOrientationsFound = False
        for partiallyDirectedAgg in self.perspectiveToAgg.values():
            for relVar1, relVar2 in EdgeOrientation._findMR3Removals(partiallyDirectedAgg):
                if isinstance(relVar1, RelationalVariable) and isinstance(relVar2, RelationalVariable):
                    self.propagateEdgeRemoval(partiallyDirectedAgg[relVar1][relVar2]
                        [AbstractGroundGraph.UNDERLYING_DEPENDENCIES], recurse=True)
                    self.recordEdgeOrientationUsage('MR3')
                    newOrientationsFound = True
        return newOrientationsFound


    def _isValidCDCandidate(self, graph, relVar1, relVar2, relVar3, ignoredSepset):
        if not isinstance(relVar1, RelationalVariable) or not isinstance(relVar2, RelationalVariable) or \
                not isinstance(relVar3, RelationalVariable):
            return False
        if relVar1.attrName == relVar3.attrName:
            return False
        if len(relVar3.path) > 1:
            return False
        # Check if triple can still be oriented as a collider
        if not(relVar2 in graph[relVar1] and relVar2 in graph[relVar3] and
                   (relVar1 in graph[relVar2] or relVar3 in graph[relVar2])):
            return False
        logger.debug('CD candidate: %s, %s, %s', relVar1, relVar2, relVar3)
        sepset = self.findRecordAndReturnSepset(relVar1, relVar3)
        return sepset is not None and relVar2 not in sepset and \
               all([not relVar2.intersects(sepsetVar) for sepsetVar in sepset])


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
        return sepset is not None


    def findRecordAndReturnSepset(self, relVar1, relVar2):
        if (relVar1, relVar2) in self.sepsets:
            return self.sepsets[(relVar1, relVar2)]
        else:
            sepset = None
            logger.debug("findRecordAndReturnSepset for %s and %s", relVar1, relVar2)
            for conditioningSetSize in range(self.depth+1):
                sepset, testedAtCurrentSize = self.findSepset(relVar1, relVar2, conditioningSetSize,
                                                              phaseI=False, phaseForRecording='Phase II')
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
        for undirectedDependency in undirectedDependencies:
            dependencyChecker(self.schema, undirectedDependency)
        self.undirectedDependencies = undirectedDependencies
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
        self.perspectiveToAgg = collections.OrderedDict([(perspective, AbstractGroundGraph(schemaDepWrapper, perspective, 2*self.hopThreshold)) for perspective in perspectives])

    def constructTrueAggsFromDependencies(schema, dependencies, hopThreshold):
        schemaDepWrapper = SchemaDependencyWrapper(schema, dependencies)
        perspectives = [si.name for si in schema.getSchemaItems()]
        perspectiveToAgg = {perspective: AbstractGroundGraph(schemaDepWrapper, perspective, 2*hopThreshold, isTruth=True)
                                      for perspective in perspectives}
        return perspectiveToAgg


    def recordEdgeOrientationUsage(self, edgeOrientationName):
        self.edgeOrientationRuleFrequency[edgeOrientationName] += 1


    def resetEdgeOrientationUsage(self):
        self.edgeOrientationRuleFrequency = {'CD': 0, 'KNC': 0, 'CA': 0, 'MR3': 0, 'RBO': 0}


    def propagateEdgeRemoval(self, underlyingRelDeps, recurse=False, force=False):
        underlyingRelDeps = set(underlyingRelDeps)
        for agg in self.perspectiveToAgg.values():
            for underlyingRelDep in underlyingRelDeps:
                otherUnderlyingRelDeps = agg.removeEdgesForDependency(underlyingRelDep, force=force)
                if recurse:
                    self.propagateEdgeRemoval(otherUnderlyingRelDeps - underlyingRelDeps)


    def report(self):
        return self.ciRecord, self.edgeOrientationRuleFrequency


    def runRCD(schema, citest, hopThreshold, depth=None):
        rcd = RCD(schema, citest, hopThreshold, depth)
        rcd.identifyUndirectedDependencies()
        rcd.orientDependencies()
        return rcd