import pdb
import random

import numpy as np
import networkx as nx

from itertools import product

from ..model import ParserUtil
from ..model.Model import Model
from ..modelspace import RelationalSpace
from ..dseparation.AbstractGroundGraph import AbstractGroundGraph
from ..model.RelationalDependency import RelationalDependency

from ..dseparation.sigma_helper import SCC
from ..graph.GraphUtil import isPossibleAncestor, isNoAncestor


def generateModel(schema, hopThreshold, numDependencies, numFeedbackLoops, maxNumParents=None, dependencies=None, randomPicker=random.sample):
    """
    dependencies is an optional set of potential dependencies that the model chooses its dependencies from.
    If dependencies=None, then the potential dependencies is the full space given by the schema and hopThreshold.
    If dependencies is specified, then hopThreshold is ignored.
    dependencies is only for testing so consistency with schema isn't checked.
    dependencies will be modified in place as they get selected without replacement.
    """
    if not dependencies:
        dependencies = RelationalSpace.getRelationalDependencies(schema, hopThreshold)
    else:
        dependencies = [ParserUtil.parseRelDep(relDepStr) for relDepStr in dependencies]
    if len(dependencies) < numDependencies:
        raise Exception("Could not generate a model: not enough dependencies to draw from")

    if not numDependencies:
        return Model(schema, [])

    attrToNumParents = {}

    model = None
    modelDeps = []
    while len(dependencies) > 0 and len(modelDeps) < (numDependencies - numFeedbackLoops):
        dependency = randomPicker(dependencies, 1)[0]
        dependencies.remove(dependency)
        modelDeps.append(dependency)
        
        try:
            model = Model(schema, modelDeps, allowCycles=False)
            attrToNumParents.setdefault(dependency.relVar2, 0)
            attrToNumParents[dependency.relVar2] += 1
            if maxNumParents and attrToNumParents[dependency.relVar2] > maxNumParents:
                raise Exception
        except Exception:
            model = None
            modelDeps.remove(dependency)

    if not model or len(model.dependencies) < (numDependencies - numFeedbackLoops):
        raise Exception("Could not generate a model: failed to find a model with {} dependenc[y|ies]".format(numDependencies))

    effects = {}
    for relDep in modelDeps:
        ck = relDep.relVar2.attrName
        if ck in effects:
            effects[ck] += 1
        else:
            effects[ck] = 1

    modelReverseDeps = []
    i = 0
    while i < len(modelDeps) and len(modelReverseDeps) < numFeedbackLoops:
        # dependency = randomPicker(modelDeps, 1)[0]
        dependency = modelDeps[i]
        if len(schema.entityNamesToEntities.keys()) > 1 and \
            (dependency.relVar1.attrName in effects or effects[dependency.relVar2.attrName] > 1):
            i += 1
            continue
        reverseDependency = dependency.reverse()
        # if dependency.relVar2.path[0] == reverseDependency.relVar2.path[0]:
        modelReverseDeps.append(reverseDependency)
        i += 1

    if numFeedbackLoops > 0:
        model = Model(schema, modelDeps + modelReverseDeps, allowCycles=True)


    if not model or len(modelReverseDeps) < numFeedbackLoops:
        raise Exception("Could not generate a model: failed to find a model with {} feedback loops".format(numFeedbackLoops))

    # if not connectedAggs(schema, model, hopThreshold):
    #     print('not connected!')
    #     raise Exception("Could not generate a model: AGG not connected")
    
    if not validAggSize(schema, model, hopThreshold):
        raise Exception("Could not generate a model: max AGG size exceeds limit {},{}".format(15, 30))

    if not hasValidRelAcyclifications(schema, model):
        raise Exception("Could not generate a model: cannot have valid relational acyclifications")

    return model


def connectedAggs(schema, model, hopThreshold):
    perspectives = [si.name for si in schema.getSchemaItems()]
    for perspective in perspectives:
        agg = AbstractGroundGraph(model, perspective, 2*hopThreshold)
        # agg.remove_nodes_from(nx.isolates(agg))
        if nx.number_connected_components(agg.to_undirected()) > 1:
            return False
    return True


def validAggSize(schema, model, hopThreshold):
    perspectives = [si.name for si in schema.getSchemaItems()]
    max_nodes, max_edges = 0, 0
    for perspective in perspectives:
        agg = AbstractGroundGraph(model, perspective, 2*hopThreshold)
        max_nodes = max(len(agg), max_nodes)
        max_edges = max(len(agg.edges()), max_edges)
    return max_nodes <= 15 and max_edges <= 30
        

def hasValidRelAcyclifications(schema, model):
    relationships = list(schema.relNamesToRelationships.keys())

    attrAdjs = {}
    floops = {}
    for relDep in model.dependencies:
        attr1 = relDep.relVar1.attrName
        attr2 = relDep.relVar2.attrName
        attrAdjs.setdefault(attr1, []).append(attr2)
        if attr1 in attrAdjs.get(attr2, []):
            if relDep.relVar1.getTerminalItemName() != relDep.relVar2.getTerminalItemName():
                floops[relDep.relVar1.getTerminalItemName()] = relDep.relVar2.getTerminalItemName()
                floops[relDep.relVar2.getTerminalItemName()] = relDep.relVar1.getTerminalItemName()

    for relDep in model.dependencies:
        if relDep.relVar2.getTerminalItemName() not in floops:
            continue
        Iu = relDep.relVar1.getTerminalItemName()
        Iv = floops[relDep.relVar2.getTerminalItemName()]

        if Iu != Iv and not str(Iu + Iv) in relationships and not str(Iv + Iu) in relationships:
            return False
    return True


def hasValidAcyclifications(schema, model, hopThreshold):
    perspectives = [si.name for si in schema.getSchemaItems()]
    for perspective in perspectives:
        agg = AbstractGroundGraph(model, perspective, 2*hopThreshold)
        acyAGGs = genAcyclifications(agg)
        if not isValidAcyclifications(agg, acyAGGs):
            return False
    return True


def genAcyclifications(g):
    """
    Generates all possible acyclifications of DCG g
    """
    sccs = {}
    for node in g:
        sccs[node] = SCC(g, node)

    ag = nx.DiGraph()
    ag.add_nodes_from(g.nodes())
    edge_goups = []

    # any incoming edge to a component should be replicated for all members of the component
    # for any pair (i, j) in same SCC, enumerate all possible orientations
    mark = set()
    for u,v in g.edges():
        if sccs[u] == sccs[v]:
            ek = tuple(sorted((u,v)))
            if ek not in mark:
                edge_goups.append([(u,v), (v,u)])
                mark.add(ek)
        else:
            for _v in sccs[v]:
                if not ag.has_edge(u, _v):
                    ag.add_edge(u, _v)

    for u in g:
        for v in g:
            if u == v:
                continue
            if sccs[u] != sccs[v]:
                continue
            if ag.has_edge(u, v) or ag.has_edge(v, u):
                continue
            edge_goups.append([(u,v), (v,u)])

    ags = []
    for edges in product(*edge_goups):
        _ag = ag.copy()
        _ag.add_edges_from(edges)
        if nx.is_directed_acyclic_graph(_ag):
            ags.append(_ag)

    return ags


# def isValidAcyclifications(trueAGG, acyAGGs):
#     tA = isPossibleAncestor(trueAGG)
#     for acyAGG in acyAGGs:
#         aA = isPossibleAncestor(acyAGG)
#         if np.array_equal(tA, aA):
#             return True
#     return False


def isValidAcyclifications(trueAGG, acyAGGs):
    tA = isPossibleAncestor(trueAGG)
    aA = np.zeros(tA.shape, dtype=int)
    for acyAGG in acyAGGs:
        aA |= isPossibleAncestor(acyAGG)
    if not np.array_equal(tA, tA & aA):
        return False
    if not np.array_equal(tA, tA | aA):
        return False
    return True