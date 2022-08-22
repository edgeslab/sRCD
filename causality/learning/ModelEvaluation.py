import pdb
import copy
import collections

import numpy as np
import networkx as nx

from ..model import ParserUtil
from ..model.RelationalDependency import RelationalDependency
from causality.dseparation.AbstractGroundGraph import AbstractGroundGraph
from . import FCIEdgeOrientation

def precision(trueValues, learnedValues):
    trueValueSet = set(trueValues)
    learnedValueSet = set(learnedValues)
    if not learnedValueSet:
        return 1.0
    else:
        return len(learnedValueSet & trueValueSet) / len(learnedValueSet)


def precisionTailMarks(trueDeps, learnedDeps):
    trueDepSet = set(trueDeps)
    learnedDepSet = set(learnedDeps)
    
    if not learnedDepSet:
        return 1.0

    count = 0
    discount = 0
    for relDep in learnedDepSet:
        if (relDep not in trueDepSet) and (relDep.mirror() not in trueDepSet):
            continue
        for trueDep in trueDepSet:
            if relDep == trueDep or relDep.mirror() == trueDep:
                if relDep.mirror() == trueDep:
                    relDep = relDep.mirror()
                if relDep.tailMarkFrom == RelationalDependency.TAIL_MARK_CIRCLE:
                    discount += 1
                if relDep.tailMarkTo == RelationalDependency.TAIL_MARK_CIRCLE:
                    discount += 1
                if relDep.tailMarkFrom == trueDep.tailMarkFrom:
                    count += 1
                if relDep.tailMarkTo == trueDep.tailMarkTo:
                    count += 1

    if (2 * len(learnedDepSet) - discount) == 0:
        return 1.0

    return count / (2 * len(learnedDepSet) - discount)


def recall(trueValues, learnedValues):
    trueValueSet = set(trueValues)
    learnedValueSet = set(learnedValues)
    if not trueValueSet:
        return 1.0
    else:
        return len(learnedValueSet & trueValueSet) / len(trueValueSet)


def recallTailMarks(trueDeps, learnedDeps):
    trueDepSet = set(trueDeps)
    learnedDepSet = set(learnedDeps)
    
    if not trueDepSet:
        return 1.0
    
    count = 0
    discount = 0
    for trueDep in trueDepSet:
        notfound = (trueDep not in learnedDepSet) and (trueDep.mirror() not in learnedDepSet)
        if notfound and trueDep.isFeedbackLoop():
            notfound = trueDep.reverse() not in learnedDepSet
        if notfound:
            continue
        
        for relDep in learnedDepSet:
            if relDep == trueDep or relDep.mirror() == trueDep:
                if relDep.mirror() == trueDep:
                    relDep = relDep.mirror()
                if relDep.tailMarkFrom == RelationalDependency.TAIL_MARK_CIRCLE:
                    discount += 1
                if relDep.tailMarkTo == RelationalDependency.TAIL_MARK_CIRCLE:
                    discount += 1
                if relDep.tailMarkFrom == trueDep.tailMarkFrom:
                    count += 1
                if relDep.tailMarkTo == trueDep.tailMarkTo:
                    count += 1

    if (2 * len(learnedDepSet) - discount) == 0:
        return 0.0

    return count / (2 * len(trueDepSet) - discount)


def skeletonPrecision(model, learnedDependencies):
    checkInput(learnedDependencies)
    # Only counting edges (ignoring orientation), so add all reverses in true and learned and take the set
    learnedDependencies = [ParserUtil.parseRelDep(relDepStr) for relDepStr in learnedDependencies]
    learnedUndirectedDependencies = learnedDependencies + [dependency.reverse() for dependency in learnedDependencies]
    trueUndirectedDependencies = model.dependencies + [dependency.reverse() for dependency in model.dependencies]
    return precision(trueUndirectedDependencies, learnedUndirectedDependencies)


def skeletonRecall(model, learnedDependencies):
    checkInput(learnedDependencies)
    # Only counting edges (ignoring orientation), so add all reverses in true and learned and take the set
    learnedDependencies = [ParserUtil.parseRelDep(relDepStr) for relDepStr in learnedDependencies]
    learnedUndirectedDependencies = learnedDependencies + [dependency.reverse() for dependency in learnedDependencies]
    trueUndirectedDependencies = model.dependencies + [dependency.reverse() for dependency in model.dependencies]
    return recall(trueUndirectedDependencies, learnedUndirectedDependencies)


def orientedPrecision(model, learnedDependencies, isPAG=False):
    checkInput(learnedDependencies)
    # Only counting oriented edges (ignoring unoriented), so remove dependencies that include their reverses
    learnedDependencies = {ParserUtil.parseRelDep(relDepStr) for relDepStr in learnedDependencies}
    learnedOrientedDependencies = set()
    for learnedDependency in learnedDependencies:
        if learnedDependency.reverse() not in learnedDependencies:
            learnedOrientedDependencies.add(learnedDependency)
    if isPAG:
        for learnedDependency in learnedDependencies:
            if learnedDependency.mirror() not in learnedOrientedDependencies:
                learnedOrientedDependencies.add(learnedDependency)
        return precisionTailMarks(depsPAG(model), learnedOrientedDependencies)
    else:
        for learnedDependency in learnedDependencies:
            if learnedDependency.reverse() not in learnedDependencies:
                learnedOrientedDependencies.add(learnedDependency)
        return precision(model.dependencies, learnedOrientedDependencies)


def depsPAG(model):
    modelDepsPAG = set()
    for trueDep in model.dependencies:
        trueDepCopy = copy.deepcopy(trueDep)
        if trueDep.reverse() in model.dependencies:
            trueDepCopy.tailMarkFrom = RelationalDependency.TAIL_MARK_LEFT_ARROW
            trueDepCopy.tailMarkTo = RelationalDependency.TAIL_MARK_RIGHT_ARROW
        if trueDepCopy.reverse() not in modelDepsPAG:
            modelDepsPAG.add(trueDepCopy)
    return modelDepsPAG


def orientedRecall(model, learnedDependencies, isPAG=False):
    checkInput(learnedDependencies)
    # Only counting oriented edges (ignoring unoriented), so remove dependencies that include their reverses
    learnedDependencies = {ParserUtil.parseRelDep(relDepStr) for relDepStr in learnedDependencies}
    learnedOrientedDependencies = set()
    if isPAG:
        for learnedDependency in learnedDependencies:
            if learnedDependency.mirror() not in learnedOrientedDependencies:
                learnedOrientedDependencies.add(learnedDependency)
        return recallTailMarks(depsPAG(model), learnedOrientedDependencies)
    else:
        for learnedDependency in learnedDependencies:
            if learnedDependency.reverse() not in learnedDependencies:
                learnedOrientedDependencies.add(learnedDependency)
        return recall(model.dependencies, learnedOrientedDependencies)



def checkInput(learnedDependencies):
    if not isinstance(learnedDependencies, collections.Iterable) or isinstance(learnedDependencies, str):
        raise Exception("learnedDependencies must be a list of RelationalDependencies "
                        "or parseable RelationalDependency strings")


def parentalQuery(trueAggs, learnedAggs, queryFunc):
    tpr, fpr, fnr = 0, 0, 0
    for k,v in trueAggs.items():
        trueA = queryFunc(trueAggs[k])
        learnedA = queryFunc(learnedAggs[k], trueAggs[k])
        tpr += ((learnedA == 1) & (learnedA == trueA)).sum() / (trueA == 1).sum()
        fpr += ((learnedA == 1) & (learnedA != trueA)).sum() / (trueA == 0).sum()
        fnr += ((learnedA == 0) & (learnedA != trueA)).sum() / (trueA == 1).sum()
    tpr = tpr/len(trueAggs)
    fpr = fpr/len(trueAggs)
    fnr = fnr/len(trueAggs)

    precision = tpr / (tpr + fpr)
    recall = tpr / (tpr + fnr)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


def getTrueLoops(aggToPerspectives):
    trueLoops = set()
    for p in aggToPerspectives.keys():
        for e in aggToPerspectives[p].edges():
            if aggToPerspectives[p].has_edge(e[1], e[0]):
                trueLoops.add(tuple(sorted((e[0], e[1]))))
    return trueLoops


def getLearnedLoopCandidates(aggToPerspectives):
    learnedCandidates = set()
    for p in aggToPerspectives.keys():
        agg = aggToPerspectives[p]
        for e in agg.edges():
            if agg.has_edge(e[1], e[0]):
                parent0 = set(agg.neighbors(e[0])) & (set(agg.predecessors(e[0])) - set(agg.successors(e[0]))) - set([e[1]])
                parent1 = set(agg.neighbors(e[1])) & (set(agg.predecessors(e[1])) - set(agg.successors(e[1]))) - set([e[0]])
                if len(parent0) == len(parent0 & parent1):
                    learnedCandidates.add(tuple(sorted((e[0], e[1]))))
    return learnedCandidates


def feedbackPrecision(trueAggs, learnedAggs):
    trueLoops = getTrueLoops(trueAggs)
    learnedCandidates = getLearnedLoopCandidates(learnedAggs)

    if len(learnedCandidates) == 0:
        return 1.0
    
    return len(learnedCandidates & trueLoops) / len(learnedCandidates)


def feedbackRecall(trueAggs, learnedAggs):
    trueLoops = getTrueLoops(trueAggs)
    learnedCandidates = getLearnedLoopCandidates(learnedAggs)
    
    if len(trueLoops) == 0:
        return 1.0
    return len(learnedCandidates & trueLoops) / len(trueLoops)
    