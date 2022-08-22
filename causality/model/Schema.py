import json

class Schema(object):
    """
    Manages entities and relationships. Note: Naming of those is global, and therefore must have unique item (E or R)
    names.

    Enum for cardinalities:
    """
    ONE = 'one'
    MANY = 'many'


    def __init__(self, relItems=False):
        self.entityNamesToEntities = {}
        self.relNamesToRelationships = {}
        self.relationAsItems = relItems


    def __key(self):
        return tuple(sorted({entName: ent._SchemaItem__key() for entName, ent in self.entityNamesToEntities.items()}.items())), \
               tuple(sorted({relName: rel._Relationship__key() for relName, rel in self.relNamesToRelationships.items()}.items())) # prepending attribute __key with class name to fix namespace issue


    def __eq__(self, other):
        return isinstance(other, Schema) and self.__key() == other.__key()


    def __hash__(self):
        return hash(self.__key())


    def __repr__(self):
        return "<{} {!r}: {}>".format(self.__class__.__name__, list(self.entityNamesToEntities.values()),
            list(self.relNamesToRelationships.values()))


    def addEntity(self, entityName):
        if entityName in self.entityNamesToEntities:
            raise Exception("Schema already has entity named {!r}".format(entityName))
        if entityName in self.relNamesToRelationships:
            raise Exception("Schema already has relationship named {!r}".format(entityName))
        self.entityNamesToEntities[entityName] = Entity(entityName)


    def getEntities(self):
        return self.entityNamesToEntities.values()


    def getEntity(self, entityName):
        if entityName not in self.entityNamesToEntities:
            raise Exception('Entity {!r} does not exist'.format(entityName))
        return self.entityNamesToEntities[entityName]


    def hasEntity(self, entityName):
        return entityName in self.entityNamesToEntities


    def addRelationship(self, relationshipName, entity1NameCard, entity2NameCard):
        """
        entity1NameCard and entity2NameCard are (entityName, cardinality) pairs where cardinality is one of Schema.ONE or Schema.MANY
        """
        if relationshipName in self.relNamesToRelationships:
            raise Exception("Schema already has relationship named {!r}".format(relationshipName))
        if relationshipName in self.entityNamesToEntities:
            raise Exception("Schema already has entity named {!r}".format(relationshipName))
        if relationshipName in self.relNamesToRelationships:
            raise Exception("Schema already has relationship named {!r}".format(relationshipName))
        entity1Name, entity1Card = entity1NameCard
        entity2Name, entity2Card = entity2NameCard
        if (entity1Name not in self.entityNamesToEntities) or (entity2Name not in self.entityNamesToEntities):
            raise Exception("One of entity1Name or entity2Name " \
                            "not listed in entityNames: {!r}, {!r}".format(entity1Name, entity2Name))
        if entity1Card not in [Schema.ONE, Schema.MANY] or entity2Card not in [Schema.ONE, Schema.MANY]:
            raise Exception("Bad cardinality for entity1 or entity2: {} {}. Should be either Schema.ONE or Schema.MANY".format(entity1Card, entity2Card))

        self.relNamesToRelationships[relationshipName] = Relationship(relationshipName, entity1NameCard, entity2NameCard)


    def getRelationships(self):
        return self.relNamesToRelationships.values()


    def getRelationship(self, relationshipName):
        if relationshipName not in self.relNamesToRelationships:
            raise Exception('Relationship {!r} does not exist'.format(relationshipName))
        return self.relNamesToRelationships[relationshipName]


    def hasRelationship(self, relationshipName):
        return relationshipName in self.relNamesToRelationships


    def getSchemaItems(self):
        schemaItems = list(self.entityNamesToEntities.values())
        if self.relationAsItems:
            schemaItems.extend(list(self.relNamesToRelationships.values()))
        return schemaItems


    def getSchemaItem(self, schemaItemName):
        if schemaItemName in self.entityNamesToEntities:
            return self.entityNamesToEntities[schemaItemName]
        elif schemaItemName in self.relNamesToRelationships:
            return self.relNamesToRelationships[schemaItemName]
        else:
            raise Exception('Schema item {!r} does not exist'.format(schemaItemName))


    def hasSchemaItem(self, schemaItemName):
        return schemaItemName in self.relNamesToRelationships or schemaItemName in self.entityNamesToEntities


    def addAttribute(self, schemaItemName, attrName, dataType=None):
        self.getSchemaItem(schemaItemName).addAttribute(attrName, dataType)


    def addAttributes(self, schemaItemName, attrNames):
        item = self.getSchemaItem(schemaItemName)
        for attName in attrNames:
            item.addAttribute(attName, None)


    def getRelationshipsForEntity(self, entityName):
        self.getEntity(entityName) # raises exception if entityName isn't an entity
        relationships = []
        for relationship in self.relNamesToRelationships.values():
            if relationship.hasEntity(entityName):
                relationships.append(relationship)
        return relationships


    def getRelationshipsBetweenEntities(self, entity1Name, entity2Name):
        self.getEntity(entity1Name) # raises exception if entity1Name isn't an entity
        self.getEntity(entity2Name) # raises exception if entity2Name isn't an entity
        relationships = []
        for relationship in self.relNamesToRelationships.values():
            if relationship.hasEntity(entity1Name) and relationship.hasEntity(entity2Name):
                relationships.append(relationship)
        return relationships


    def toJsonDict(self):
        """
        Returns a json-compatible dictionary that serializes this schema.
        """
        return {'entities': [ent.toJsonDict() for ent in self.entityNamesToEntities.values()],
                'relationships': [rel.toJsonDict() for rel in self.relNamesToRelationships.values()]}


    def toFile(self, fileName):
        with open(fileName, 'w') as file:
            json.dump(self.toJsonDict(), file, indent='\t')


    @classmethod
    def fromFile(cls, fileName):
        schema = Schema()
        with open(fileName, 'r') as file:
            jsonDict = json.load(file)
            for entDict in jsonDict['entities']:
                entName = entDict['name']
                schema.addEntity(entName)
                for attrList in entDict['attributes']:
                    schema.addAttribute(entName, attrList['name'], attrList['dataType'])
            for relDict in jsonDict['relationships']:
                relName = relDict['name']
                schema.addRelationship(relName, (relDict['cardinalities']['entity1Name'], relDict['cardinalities']['entity1Card']),
                                                (relDict['cardinalities']['entity2Name'], relDict['cardinalities']['entity2Card']))
                for attrList in relDict['attributes']:
                    schema.addAttribute(relName, attrList['name'], attrList['dataType'])
        return schema


class SchemaItem(object):
    """
    Every SchemaItem has an implicit existence attribute used by Model.  This attribute has a special case-sensitive
    name ('exists'), but it is not explicitly represented in the list of attributes.  Dependencies can refer to it
    in the usual syntax of relational variables and dependencies. E.g., [A].exists or [A, AB].exists
    The 'exists' attribute name is unique and we enforce it.
    """
    EXISTS_ATTR_NAME = 'exists'

    def __init__(self, itemName):
        self.name = itemName
        self.attributes = []


    def __key(self):
        return self.name, tuple(sorted(self.attributes, key=lambda attr: attr.name))


    def __eq__(self, other):
        return isinstance(other, SchemaItem) and type(self) == type(other) and self.__key() == other.__key()


    def __hash__(self):
        return hash(self.__key())

    
    def __repr__(self):
        return "<{} {!r}: {}>".format(self.__class__.__name__, self.name, self.attributes)


    def addAttribute(self, attrName, dataType):
        if attrName == SchemaItem.EXISTS_ATTR_NAME:
            raise Exception("Cannot add attribute with reserved name {!r}".format(SchemaItem.EXISTS_ATTR_NAME))
        if attrName in [attr.name for attr in self.attributes]:
            raise Exception("Schema already has attribute named {!r} for item {!r}".format(attrName, self.name))
        self.attributes.append(Attribute(attrName, dataType))


    def toJsonDict(self):
        # return {'name': self.name, 'attributes': [attr.name for attr in self.attributes]}
        attrDicts = []
        for attr in self.attributes:
            attrDicts.append({'name': attr.name, 'dataType': attr.dataType})
        return {'name': self.name, 'attributes': attrDicts}


class Entity(SchemaItem):

    def __init__(self, entityName):
        super().__init__(entityName)


class Relationship(SchemaItem):

    def __init__(self, relationshipName, entity1NameCard, entity2NameCard):
        super().__init__(relationshipName)
        self.cardinalities = {}
        self.entity1Name, self.entity1Card = entity1NameCard
        self.cardinalities[self.entity1Name] = self.entity1Card
        self.entity2Name, self.entity2Card = entity2NameCard
        self.cardinalities[self.entity2Name] = self.entity2Card


    def __key(self):
        return self.name, tuple(sorted(self.attributes, key=lambda attr: attr.name)), tuple(sorted(self.cardinalities.items()))


    def __eq__(self, other):
        return super().__eq__(other) and self.__key() == other.__key()


    def __hash__(self):
        return hash(self.__key())


    def __repr__(self):
        return "<{} {!r}: {} {}>".format(self.__class__.__name__, self.name, self.cardinalities, self.attributes)


    def getCardinality(self, entityName):
        if entityName not in self.cardinalities:
            raise Exception('Entity {!r} does not exist for relationship {!r}'.format(entityName, self.name))
        return self.cardinalities[entityName]


    def hasEntity(self, entityName):
        return entityName in [self.entity1Name, self.entity2Name]


    def toJsonDict(self):
        jsonDict = super().toJsonDict()
        jsonDict['cardinalities'] = {'entity1Name': self.entity1Name, 'entity1Card': self.entity1Card,
                                     'entity2Name': self.entity2Name, 'entity2Card': self.entity2Card}
        return jsonDict


class Attribute(object):

    # all supported types of data
    DATA_TYPES = {'STRING': 0, 'INTEGER': 1, 'FLOAT': 2}
    STRING = DATA_TYPES['STRING']
    INTEGER = DATA_TYPES['INTEGER']
    FLOAT = DATA_TYPES['FLOAT']

    def __init__(self, attrName, dataType=None):
        """
        Pass None if use is dataType-agnostic
        """
        if dataType and dataType not in self.DATA_TYPES.values():
            raise Exception("unknown dataType found: {}".format(dataType))
        self.name = attrName
        self.dataType = dataType


    def __key(self):
        return self.name, self.dataType


    def __eq__(self, other):
        return isinstance(other, Attribute) and self.__key() == other.__key()


    def __hash__(self):
        return hash(self.__key())


    def __repr__(self):
        return "<{} {!r} {!r}>".format(self.__class__.__name__, self.name, self.dataType)
