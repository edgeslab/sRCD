import numbers
import random
import itertools
import string
import networkx as nx
from ..model.Distribution import MarginalDistribution
from ..model.Distribution import DiscreteMarginalDistribution
from ..model.Distribution import PoissonMarginalDistribution
from ..model.Schema import Schema

def generateSchema(numEntities, numRelationships,
                   entityAttrDistribution=PoissonMarginalDistribution(2.4),
                   relationshipAttrDistribution=PoissonMarginalDistribution(1),
                   entityPairPicker=random.sample,
                   cardinalityDistribution=DiscreteMarginalDistribution({Schema.ONE: 0.0, Schema.MANY: 1.0}),
                   allowCycles=True, oneRelationshipPerPair=False):
    """
    entityAttrDistribution and relationshipAttrDistribution are marginal distributions to be sampled for determining
        the number of attributes for each entity or relationship, respectively.
    entityPairPicker is a function that selects two entities from a sequence of all entities in the schema for a relationship.
    cardinalityDistribution is a marginal distribution whose domain is a subset of {Schema.ONE, Schema.MANY}.
    allowCycles is a flag that restricts whether cycles can exist in the schema.  Setting allowCycles=False overrides the
        default behavior of oneRelationshipPerPair (equivalent to setting it to True)
    oneRelationshipPerPair is a flag limiting the number of relationships per pair of entities.
    """
    if not isinstance(numEntities, numbers.Integral) or numEntities < 0:
        raise Exception("numEntities must be a non-negative integer")
    if not isinstance(numRelationships, numbers.Integral) or numRelationships < 0:
        raise Exception("numRelationships must be a non-negative integer")
    if numRelationships > 0 and numEntities < 2:
        raise Exception("must have at least 2 entities to support a relationship: found {}".format(numEntities))
    if not allowCycles and numRelationships > numEntities-1:
        raise Exception("Too many relationships requested: asked for {}, at most {} possible".format(
            numRelationships, numEntities-1))
    if oneRelationshipPerPair and numRelationships > numEntities*(numEntities-1)/2: # number of unique entity pairs
        raise Exception("Too many relationships requested: asked for {}, at most {} possible".format(
            numRelationships, int(numEntities*(numEntities-1)/2)))
    if not isinstance(entityAttrDistribution, MarginalDistribution):
        raise Exception("entityAttrDistribution must be a MarginalDistribution")
    if not isinstance(relationshipAttrDistribution, MarginalDistribution):
        raise Exception("relationshipAttrDistribution must be a MarginalDistribution")
    if not isinstance(cardinalityDistribution, MarginalDistribution):
        raise Exception("cardinalityDistribution must be a MarginalDistribution")

    entityNames = string.ascii_uppercase[:6]
    entityNamesToAttrNamePrefix = {'A': 'X', 'B': 'Y', 'C': 'Z', 'D': 'W', 'E': 'V', 'F': 'U'}
    relNamesToCount = {}

    schema = Schema(relItems=False)
    for entIdx in range(numEntities):
        entName = entityNames[entIdx]
        schema.addEntity(entName)

        numEntityAttrs = 1 + entityAttrDistribution.sample(None)
        if not isinstance(numEntityAttrs, numbers.Integral) or numEntityAttrs < 0:
            raise Exception("entityAttrDistribution must have a domain of non-negative integers: returned {}".
                format(numEntityAttrs))

        for attrIdx in range(numEntityAttrs):
            schema.addAttribute(entName, '{attrNamePrefix}{attrIdx}'.format(
                attrNamePrefix=entityNamesToAttrNamePrefix[entName], attrIdx=attrIdx+1))

    entityGraph = nx.Graph()
    entityGraph.add_nodes_from([entity.name for entity in schema.getEntities()])
    candidateEntityPairs = list(itertools.combinations(list(schema.getEntities()), 2))
    random.shuffle(candidateEntityPairs)

    # relNames = set()
    for relIdx in range(numRelationships):
        # relName = 'Rel_{relIdx}'.format(relIdx=relIdx)
        ent1, ent2 = chooseEntityPair(entityPairPicker, candidateEntityPairs, oneRelationshipPerPair, allowCycles, entityGraph)
        relNameBase = ent1.name + ent2.name if ent1.name < ent2.name else ent2.name + ent1.name
        relNamesToCount.setdefault(relNameBase, 0)
        relNamesToCount[relNameBase] += 1
        if relNamesToCount[relNameBase] == 1:
            relName = relNameBase
        else:
            relName = relNameBase + str(relNamesToCount[relNameBase])

        ent1Card = cardinalityDistribution.sample(None)
        ent2Card = cardinalityDistribution.sample(None)
        if not all([card == Schema.ONE or card == Schema.MANY for card in [ent1Card, ent2Card]]):
            raise Exception("cardinalityDistribution must return either Schema.ONE or Schema.MANY: returned {} and {}".
                format(ent1Card, ent2Card))
        schema.addRelationship(relName, (ent1.name, ent1Card), (ent2.name, ent2Card))

        # numRelationshipAttrs = relationshipAttrDistribution.sample(None)
        numRelationshipAttrs = 0 #TODO: check
        if not isinstance(numRelationshipAttrs, numbers.Integral) or numRelationshipAttrs < 0:
            raise Exception("relationshipAttrDistribution must have a domain of non-negative integers: returned {}".
                format(numRelationshipAttrs))

        attrNamePrefix = entityNamesToAttrNamePrefix[ent1.name] + entityNamesToAttrNamePrefix[ent2.name] if \
                            ent1.name < ent2.name else \
                            entityNamesToAttrNamePrefix[ent2.name] + entityNamesToAttrNamePrefix[ent1.name]
        if relNamesToCount[relNameBase] > 1:
            attrNamePrefix += str(relNamesToCount[relNameBase]) + '_'
        for attrIdx in range(numRelationshipAttrs):
            schema.addAttribute(relName, '{attrNamePrefix}{attrIdx}'.format(attrNamePrefix=attrNamePrefix,
                                                                             attrIdx=attrIdx+1))

    return schema


def chooseEntityPair(entityPairPicker, candidateEntityPairs, oneRelationshipPerPair, allowCycles, entityGraph):
    entityPair = None
    if allowCycles:
        entityPair = entityPairPicker(candidateEntityPairs, 1)[0]
        if oneRelationshipPerPair: # sample without replacement
            candidateEntityPairs.remove(entityPair)
    else:
        foundNonCycleRel = False
        while not foundNonCycleRel:
            entityPair = entityPairPicker(candidateEntityPairs, 1)[0]
            if not doesPathExist(entityGraph, entityPair[0].name, entityPair[1].name):
                foundNonCycleRel = True
                entityGraph.add_edge(entityPair[0].name, entityPair[1].name)
            candidateEntityPairs.remove(entityPair)
    return entityPair


def doesPathExist(graph, fromNode, toNode):
    """
    Is there an path in graph from fromNode to toNode? graph is undirected.
    """
    neighbors = set(graph.neighbors(fromNode))
    if toNode in neighbors:
        return True
    visited = {fromNode} | neighbors
    frontier = neighbors
    while frontier:
        nextFrontier = set()
        for node in frontier:
            nextFrontier |= {neighbor for neighbor in graph.neighbors(node) if neighbor not in visited}
            if toNode in nextFrontier:
                return True
        frontier = nextFrontier
        visited |= frontier
    return False