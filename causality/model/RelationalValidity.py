from ..model.Schema import Schema

def checkRelationalPathValidity(schema, relPath):
    # A relational path is valid if it is an alternating sequence of entities and relationships,
    # where each entity participates in the relationships that precede or succeed it.
    # No triple ERE can occur and if RER occurs then card(R, E) = MANY
    for itemName in relPath:
        if not schema.hasSchemaItem(itemName):
            raise Exception("Schema has no item {!r} in relationalPath {!r}".format(itemName, relPath))

    for item1Name, item2Name in zip(relPath[:-1], relPath[1:]):
        if schema.hasEntity(item1Name) and schema.hasRelationship(item2Name):
            if not schema.getRelationship(item2Name).hasEntity(item1Name):
                raise Exception("Entity {!r} does not participate in relationship {!r}".format(item1Name, item2Name))
        elif schema.hasRelationship(item1Name) and schema.hasEntity(item2Name):
            if not schema.getRelationship(item1Name).hasEntity(item2Name):
                raise Exception("Entity {!r} does not participate in relationship {!r}".format(item2Name, item1Name))
        else: # successive entity-entity or relationship-relationship pair
            raise Exception("Invalid item1Name {!r} and item2Name {!r} in relationalPath: types must alternate between entities " \
                            "and relationships".format(item1Name, item2Name))

    for item1Name, item2Name, item3 in zip(relPath[:-2], relPath[1:-1], relPath[2:]):
        if item1Name == item3 and schema.hasRelationship(item2Name):
            raise Exception("Found ERE pattern in relationalPath")
        if item1Name == item3 and schema.hasEntity(item2Name):
            rel = schema.getRelationship(item1Name)
            if (rel.entity1Name == item2Name and rel.getCardinality(rel.entity1Name) == Schema.ONE) or \
                    (rel.entity2Name == item2Name and rel.getCardinality(rel.entity2Name) == Schema.ONE):
                raise Exception("Found RER pattern in relationalPath with card(R, E) = ONE")


def checkRelationalVariableValidity(schema, relVar, relationalPathChecker=checkRelationalPathValidity):
    relationalPathChecker(schema, relVar.path)
    if not relVar.isExistence():
        if relVar.attrName not in [attr.name for attr in schema.getSchemaItem(relVar.getTerminalItemName()).attributes]:
            raise Exception("Schema item {!r} has no attribute {!r} in relationalVariable {!r}"
            .format(relVar.getTerminalItemName(), relVar.attrName, str(relVar)))


def checkRelationalDependencyValidity(schema, relDep, relationalVariableChecker=checkRelationalVariableValidity):
    # Check that dependency is canonical (child has singleton path)
    
    #TODO: double check whether it is ok to ignore this
    # if len(relDep.relVar2.path) > 1:
    #     raise Exception("Dependency {!r} is not canonical".format(str(relDep)))

    # Then check that dependency has consistent base items and valid relational paths
    if relDep.relVar1.getBaseItemName() != relDep.relVar2.getBaseItemName():
        raise Exception("Dependency {!r} has inconsistent base items".format(str(relDep)))

    # check that both parent and child relational variables are valid
    relationalVariableChecker(schema, relDep.relVar2)
    relationalVariableChecker(schema, relDep.relVar1)


def checkValidityOfRelationalVariableSet(schema, hopThreshold, relVars, relationalVariableChecker=checkRelationalVariableValidity):
    perspective = list(relVars)[0].getBaseItemName()
    if any([relVar.getBaseItemName() != perspective for relVar in relVars]):
        raise Exception("Perspective is not consistent across all relational variables")

    for relVar in relVars:
        if len(relVar.path) > hopThreshold+1:
            raise Exception("Relational variable {!r} is longer than the hop threshold".format(str(relVar)))
        relationalVariableChecker(schema, relVar)


