import itertools
import logging
import numbers
import networkx as nx
from ..model import RelationalValidity
from ..model.RelationalDependency import RelationalDependency, RelationalVariable
from ..model.RelationalDependency import RelationalVariableIntersection
from ..modelspace import RelationalSpace

logger = logging.getLogger(__name__)

class AbstractGroundGraph(nx.DiGraph):

    UNDERLYING_DEPENDENCIES = 'underlying_dependencies'

    def __init__(self, model, perspective, hopThreshold, isPAG=False, isTruth=False):
        """
        Builds an Abstract Ground Graph from the dependencies in the model for relational variables with base items
        of perspective limited by the hopThreshold length
        """
        super().__init__()
        if not model.schema.hasSchemaItem(perspective):
            raise Exception("Perspective must be a valid schema item name")
        self.perspective = perspective

        self.isPAG = isPAG

        if not isinstance(hopThreshold, numbers.Integral) or hopThreshold < 0:
            raise Exception("hopThreshold must be a non-negative integer")
        self.hopThreshold = hopThreshold

        self.add_nodes_from([relVar for relVar in RelationalSpace.getRelationalVariables(model.schema, self.hopThreshold,
            includeExistence=False) if relVar.getBaseItemName() == self.perspective])

        # map from relVar -> relVars + relVarInts that it subsumes
        self.subsumedVariableDict = {relVar: {relVar} for relVar in self.nodes()}

        # Cache ancestors for efficiency
        self.ancestorsCache = {}

        attrRelVarToDeps = {} # canonical, singleton outcomes to the dependencies that involve them
                              # e.g., [A].X -> [ [A, AB, B].Y -> [A].X ]
        for dependency in model.dependencies:
            attrRelVarToDeps.setdefault(dependency.relVar2, []).append(dependency)
        # For every node (relational variable), connect its causes using the extend method from the RDS paper
        for node in self.nodes():
            attrRelVar = RelationalVariable([node.getTerminalItemName()], node.attrName)
            if attrRelVar in attrRelVarToDeps:
                for dependency in attrRelVarToDeps[attrRelVar]:
                    newPaths = extendPath(model.schema, node.path, dependency.relVar1.path)
                    for newPath in newPaths:
                        causeRelVar = RelationalVariable(newPath, dependency.relVar1.attrName)
                        if causeRelVar in self.nodes():
                            if causeRelVar == node:
                                continue
                            if dependency == dependency.reverse() and isTruth:
                                if len(causeRelVar.path) < len(node.path):
                                    continue
                            self.add_edge(causeRelVar, node)
                            self[causeRelVar][node].setdefault(AbstractGroundGraph.UNDERLYING_DEPENDENCIES, set()).\
                                add(dependency)

        # Construct relational variable intersection nodes
        # Two relational variables have an intersection if they have the same base and terminal items,
        # the same attribute, and they diverge at some point (neither relational path is a prefix of the other)
        intersectionNodes = []
        for relVar1, relVar2 in itertools.combinations(self.nodes(), 2):
            if relVar1.intersects(relVar2):
                intersectionNode = RelationalVariableIntersection(relVar1, relVar2)
                intersectionNodes.append(intersectionNode)
                self.subsumedVariableDict[relVar1].add(intersectionNode)
                self.subsumedVariableDict[relVar2].add(intersectionNode)
        self.add_nodes_from(intersectionNodes)

        # Inherit dependencies from the sources of relational variable intersection nodes
        #  and pool together underlying dependencies
        intersectionEdges = {} # (from, to) -> attrDict
        for intersectionNode in intersectionNodes:
            for causeRelVar in self.predecessors(intersectionNode.relVar1):
                if (causeRelVar, intersectionNode) in intersectionEdges:
                    intersectionEdges[(causeRelVar, intersectionNode)][AbstractGroundGraph.UNDERLYING_DEPENDENCIES] |= \
                        self[causeRelVar][intersectionNode.relVar1][AbstractGroundGraph.UNDERLYING_DEPENDENCIES]
                else:
                    intersectionEdges[(causeRelVar, intersectionNode)] = {}
                    intersectionEdges[(causeRelVar, intersectionNode)][AbstractGroundGraph.UNDERLYING_DEPENDENCIES] = \
                        set(self[causeRelVar][intersectionNode.relVar1][AbstractGroundGraph.UNDERLYING_DEPENDENCIES])
            for effectRelVar in self.successors(intersectionNode.relVar1):
                if (intersectionNode, effectRelVar) in intersectionEdges:
                    intersectionEdges[(intersectionNode, effectRelVar)][AbstractGroundGraph.UNDERLYING_DEPENDENCIES] |= \
                        self[intersectionNode.relVar1][effectRelVar][AbstractGroundGraph.UNDERLYING_DEPENDENCIES]
                else:
                    intersectionEdges[(intersectionNode, effectRelVar)] = {}
                    intersectionEdges[(intersectionNode, effectRelVar)][AbstractGroundGraph.UNDERLYING_DEPENDENCIES] = \
                        set(self[intersectionNode.relVar1][effectRelVar][AbstractGroundGraph.UNDERLYING_DEPENDENCIES])
            for causeRelVar in self.predecessors(intersectionNode.relVar2):
                if (causeRelVar, intersectionNode) in intersectionEdges:
                    intersectionEdges[(causeRelVar, intersectionNode)][AbstractGroundGraph.UNDERLYING_DEPENDENCIES] |= \
                        self[causeRelVar][intersectionNode.relVar2][AbstractGroundGraph.UNDERLYING_DEPENDENCIES]
                else:
                    intersectionEdges[(causeRelVar, intersectionNode)] = {}
                    intersectionEdges[(causeRelVar, intersectionNode)][AbstractGroundGraph.UNDERLYING_DEPENDENCIES] = \
                        set(self[causeRelVar][intersectionNode.relVar2][AbstractGroundGraph.UNDERLYING_DEPENDENCIES])
            for effectRelVar in self.successors(intersectionNode.relVar2):
                if (intersectionNode, effectRelVar) in intersectionEdges:
                    intersectionEdges[(intersectionNode, effectRelVar)][AbstractGroundGraph.UNDERLYING_DEPENDENCIES] |= \
                        self[intersectionNode.relVar2][effectRelVar][AbstractGroundGraph.UNDERLYING_DEPENDENCIES]
                else:
                    intersectionEdges[(intersectionNode, effectRelVar)] = {}
                    intersectionEdges[(intersectionNode, effectRelVar)][AbstractGroundGraph.UNDERLYING_DEPENDENCIES] = \
                        set(self[intersectionNode.relVar2][effectRelVar][AbstractGroundGraph.UNDERLYING_DEPENDENCIES])

        for (fromRelVar, toRelVar), data in intersectionEdges.items():
            self.add_edge(fromRelVar, toRelVar, data)


    def getRelationalVariableNodes(self):
        return [node for node in self.nodes() if isinstance(node, RelationalVariable)]


    def getRelationalVariableIntersectionNodes(self):
        return [node for node in self.nodes() if isinstance(node, RelationalVariableIntersection)]


    def getSubsumedVariables(self, relVar):
        if not isinstance(relVar, RelationalVariable):
            raise Exception("relVar must be a RelationalVariable: found {!r}".format(str(relVar)))
        if relVar not in self.nodes():
            raise Exception("relVar {!r} is not a node in the abstract ground graph".format(str(relVar)))

        return self.subsumedVariableDict[relVar]


    def getNodesByName(self, varNames):
        nodes = self.getRelationalVariableNodes()
        out = []
        for name in varNames:
            for n in nodes:
                if str(n) == name:
                    out.append(n)
        return out


    def getAncestors(self, node):
        if node in self.ancestorsCache:
            return self.ancestorsCache[node]
        else:
            mark = {}
            ancestors = self._getAncestorsRecursive(node, mark)
            self.ancestorsCache[node] = ancestors
            return ancestors


    def _getAncestorsRecursive(self, node, mark):
        ancestors = {node}
        mark[node] = True
        for parent in self.predecessors(node):
            if parent not in mark:
                ancestors |= self._getAncestorsRecursive(parent, mark)
        return ancestors


    def removeEdgesForDependency(self, relDep, force=False):
        otherUnderlyingRelDeps = set()
        for edge in self.edges(data=True):
            if relDep in edge[2][AbstractGroundGraph.UNDERLYING_DEPENDENCIES]:
                if not force and relDep == relDep.reverse() and len(edge[0].path) > len(edge[1].path):
                    continue
                edge[2][AbstractGroundGraph.UNDERLYING_DEPENDENCIES] = \
                    edge[2][AbstractGroundGraph.UNDERLYING_DEPENDENCIES] - {relDep}
                for otherRelDep in edge[2][AbstractGroundGraph.UNDERLYING_DEPENDENCIES]:
                    if otherRelDep != relDep:
                        otherUnderlyingRelDeps.add(otherRelDep)
                if not edge[2][AbstractGroundGraph.UNDERLYING_DEPENDENCIES]:
                    self.remove_edge(edge[0], edge[1])
        return otherUnderlyingRelDeps


    def orientEdgesForDependency(self, relDep, isTo=True, mark=RelationalDependency.TAIL_MARK_RIGHT_ARROW):
        otherUnderlyingRelDeps = set()
        for edge in self.edges(data=True):
            if relDep in edge[2][AbstractGroundGraph.UNDERLYING_DEPENDENCIES]:
                edge[2][AbstractGroundGraph.UNDERLYING_DEPENDENCIES] = \
                    edge[2][AbstractGroundGraph.UNDERLYING_DEPENDENCIES] - {relDep}
                for otherRelDep in edge[2][AbstractGroundGraph.UNDERLYING_DEPENDENCIES]:
                    if otherRelDep != relDep:
                        otherUnderlyingRelDeps.add(otherRelDep)
                markFrom = relDep.tailMarkFrom if isTo else mark
                markTo = mark if isTo else relDep.tailMarkTo
                orientedRelDep = RelationalDependency(relDep.relVar1, relDep.relVar2, markFrom, markTo)
                edge[2][AbstractGroundGraph.UNDERLYING_DEPENDENCIES].add(orientedRelDep)
        return otherUnderlyingRelDeps


    def strictPredecessors(self, node):
        if not self.isPAG:
            return set(self.predecessors(node)) - set(self.successors(node))
        candidates = self.predecessors(node)
        predecessors = set()
        for c in candidates:
            relDep = list(self[c][node][AbstractGroundGraph.UNDERLYING_DEPENDENCIES])[0]
            if relDep.tailMarkTo == RelationalDependency.TAIL_MARK_RIGHT_ARROW:
                predecessors.add(c)
        return predecessors


    def strictSuccessors(self, node):
        if not self.isPAG:
            return set(self.successors(node)) - set(self.predecessors(node))
        candidates = self.successors(node)
        successors = set()
        for c in candidates:
            relDep = list(self[node][c][AbstractGroundGraph.UNDERLYING_DEPENDENCIES])[0]
            if relDep.tailMarkTo == RelationalDependency.TAIL_MARK_RIGHT_ARROW:
                successors.add(c)
        return successors


def extendPath(schema, pathOrig, pathExt):
    reversePathOrig = pathOrig[::-1]
    pivots = findPivots(reversePathOrig, pathExt)
    newPaths = []
    for pivot in pivots:
        newPath = pathOrig[0:len(pathOrig)-pivot+1] + pathExt[pivot:len(pathExt)]
        try:
            RelationalValidity.checkRelationalPathValidity(schema, newPath)
            newPaths.append(newPath)
        except:
            continue
    return newPaths


def findPivots(path1, path2):
    pivots = []
    for idx, (item1, item2) in enumerate(zip(path1, path2)):
        if item1 != item2:
            return pivots
        pivots.append(idx+1)
    return pivots