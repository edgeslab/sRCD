import numbers
from ..model.RelationalDependency import RelationalVariable
from ..model.RelationalDependency import RelationalDependency
from ..model.Schema import Schema
from ..model.Schema import SchemaItem

def getRelationalPaths(schema, hopThreshold):
    if not isinstance(hopThreshold, numbers.Number):
        raise Exception("Hop threshold must be a number: found {!r}".format(str(hopThreshold)))
    if hopThreshold < 0:
        raise Exception("Hop threshold must be >= 0: found {}".format(hopThreshold))

    schemaItems = schema.getSchemaItems()
    relPaths = [[item.name] for item in schemaItems]
    frontier = relPaths
    for hop in range(hopThreshold):
        newRelPaths = extendRelationalPaths(schema, frontier)
        relPaths.extend(newRelPaths)
        frontier = newRelPaths
    return relPaths


def extendRelationalPaths(schema, relPaths):
    extendedRelPaths = []
    for relPath in relPaths:
        terminalItem = relPath[-1]
        if schema.hasRelationship(terminalItem):
            relationship = schema.getRelationship(terminalItem)
            if len(relPath) > 1:
                if relationship.entity1Name != relPath[-2]:
                    extendedRelPaths.append(relPath[:] + [relationship.entity1Name])
                if relationship.entity2Name != relPath[-2]:
                    extendedRelPaths.append(relPath[:] + [relationship.entity2Name])
            else:
                extendedRelPaths.append(relPath[:] + [relationship.entity1Name])
                extendedRelPaths.append(relPath[:] + [relationship.entity2Name])
        else: # terminalItem is an entity
            memberRelationships = schema.getRelationshipsForEntity(terminalItem)
            for relationship in memberRelationships:
                if len(relPath) > 1:
                    if relationship.name != relPath[-2]:
                        extendedRelPaths.append(relPath[:] + [relationship.name])
                    else: # just visited this relationship, so check forward cardinality
                        if terminalItem == relationship.entity1Name:
                            if relationship.entity1Card == Schema.MANY:
                                extendedRelPaths.append(relPath[:] + [relationship.name])
                        else: # terminalItem is entity2
                            if relationship.entity2Card == Schema.MANY:
                                extendedRelPaths.append(relPath[:] + [relationship.name])
                else:
                    extendedRelPaths.append(relPath[:] + [relationship.name])
    return extendedRelPaths


def getRelationalVariables(schema, hopThreshold, includeExistence=False):
    relPaths = getRelationalPaths(schema, hopThreshold)
    relVars = []
    for relPath in relPaths:
        terminalItemName = relPath[-1]
        terminalItemAttrs = schema.getSchemaItem(terminalItemName).attributes
        for terminalItemAttr in terminalItemAttrs:
            relVars.append(RelationalVariable(relPath, terminalItemAttr.name))
        if includeExistence:
            relVars.append(RelationalVariable(relPath, SchemaItem.EXISTS_ATTR_NAME))
    return relVars


def getRelationalDependencies(schema, hopThreshold, includeExistence=False, isPAG=False):
    """
    Valid relational dependencies (relVar1 -> relVar2) must pass the following conditions:
        (1) relVar1 and relVar2 have the same perspective (base item)
        (2) relVar2 has path length 1 (canonical form)
        (3) relVar1 attribute cannot be the same as relVar2 attribute
        (4) if relVar2 is relationship existence, the relationship cannot appear in
            the path of relVar1 more than the initial base item
        (5) attribute of an item cannot cause its own existence
        (6) existence of an item cannot cause its own attributes
        (7) currently excluding entity existence as cause or effect
    """
    relVars = getRelationalVariables(schema, hopThreshold, includeExistence=includeExistence)
    relDeps = []
    for relVar1 in relVars:
        if not includeExistence and relVar1.attrName == SchemaItem.EXISTS_ATTR_NAME:
            continue
        if relVar1.isExistence() and schema.hasEntity(relVar1.getTerminalItemName()):
            continue
        for relVar2 in relVars:
            if not includeExistence and relVar2.attrName == SchemaItem.EXISTS_ATTR_NAME:
                continue
            if relVar2.isExistence() and schema.hasEntity(relVar2.getTerminalItemName()):
                continue
            if len(relVar2.path) > 1:
                continue
            if relVar1.getBaseItemName() != relVar2.getBaseItemName():
                continue
            if len(relVar1.path) == 1 and (relVar1.isExistence() or relVar2.isExistence()):
                continue
            if relVar2.isExistence() and relVar1.path.count(relVar2.getBaseItemName()) > 1:
                continue
            # if relVar1.attrName == relVar2.attrName and relVar1.attrName != SchemaItem.EXISTS_ATTR_NAME:
            #     continue
            if relVar1 == relVar2:
                continue
            if isPAG:
                dependency = RelationalDependency(relVar1, relVar2, RelationalDependency.TAIL_MARK_CIRCLE, RelationalDependency.TAIL_MARK_CIRCLE)
            else:        
                dependency = RelationalDependency(relVar1, relVar2)
            relDeps.append(dependency)
    return relDeps