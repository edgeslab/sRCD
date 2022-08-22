import unittest
from causality.test import TestUtil
from causality.model.Distribution import ConstantDistribution
from causality.model.Schema import Schema
from causality.modelspace import SchemaGenerator

class TestSchemaGenerator(unittest.TestCase):

    def testGenerateEntities(self):
        schema = SchemaGenerator.generateSchema(0, 0)
        expectedSchema = Schema()
        self.assertEqual(schema, expectedSchema)

        schema = SchemaGenerator.generateSchema(1, 0, entityAttrDistribution=ConstantDistribution(0))
        expectedSchema.addEntity('A')
        self.assertEqual(schema, expectedSchema)

        schema = SchemaGenerator.generateSchema(2, 0, entityAttrDistribution=ConstantDistribution(0))
        expectedSchema.addEntity('B')
        self.assertEqual(schema, expectedSchema)


    def testGenerateEntityAttributes(self):
        schema = SchemaGenerator.generateSchema(1, 0, entityAttrDistribution=ConstantDistribution(1))
        expectedSchema = Schema()
        expectedSchema.addEntity('A')
        expectedSchema.addAttribute('A', 'X1')
        self.assertEqual(schema, expectedSchema)

        schema = SchemaGenerator.generateSchema(2, 0, entityAttrDistribution=ConstantDistribution(2))
        expectedSchema.addEntity('B')
        expectedSchema.addAttribute('A', 'X2')
        expectedSchema.addAttribute('B', 'Y1')
        expectedSchema.addAttribute('B', 'Y2')
        self.assertEqual(schema, expectedSchema)
        
        
    def testGenerateRelationships(self):
        schema = SchemaGenerator.generateSchema(2, 1, entityAttrDistribution=ConstantDistribution(0),
            relationshipAttrDistribution=ConstantDistribution(0), cardinalityDistribution=ConstantDistribution(Schema.ONE))
        expectedSchema = Schema()
        expectedSchema.addEntity('A')
        expectedSchema.addEntity('B')
        expectedSchema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        self.assertEqual(schema, expectedSchema)

        schema = SchemaGenerator.generateSchema(2, 2, entityAttrDistribution=ConstantDistribution(0),
            relationshipAttrDistribution=ConstantDistribution(0), cardinalityDistribution=ConstantDistribution(Schema.ONE))
        expectedSchema.addRelationship('AB2', ('B', Schema.ONE), ('A', Schema.ONE))
        self.assertEqual(schema, expectedSchema)

        picker = lambda entityPairs, constOne: [sorted(entityPairs, key=lambda entPair: entPair[0].name+entPair[1].name
            if entPair[0].name < entPair[1].name else entPair[1].name+entPair[0].name)[0]]
        schema = SchemaGenerator.generateSchema(3, 2, entityAttrDistribution=ConstantDistribution(0),
            relationshipAttrDistribution=ConstantDistribution(0), entityPairPicker=picker,
            cardinalityDistribution=ConstantDistribution(Schema.ONE))
        expectedSchema = Schema()
        expectedSchema.addEntity('A')
        expectedSchema.addEntity('B')
        expectedSchema.addEntity('C')
        expectedSchema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        expectedSchema.addRelationship('AB2', ('A', Schema.ONE), ('B', Schema.ONE))
        self.assertEqual(schema, expectedSchema)

        picker = lambda entityPairs, constOne: [sorted(entityPairs, key=lambda entPair: entPair[0].name+entPair[1].name
            if entPair[0].name < entPair[1].name else entPair[1].name+entPair[0].name)[-1]]
        schema = SchemaGenerator.generateSchema(3, 2, entityAttrDistribution=ConstantDistribution(0),
            relationshipAttrDistribution=ConstantDistribution(0), entityPairPicker=picker,
            cardinalityDistribution=ConstantDistribution(Schema.ONE))
        expectedSchema = Schema()
        expectedSchema.addEntity('A')
        expectedSchema.addEntity('B')
        expectedSchema.addEntity('C')
        expectedSchema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.ONE))
        expectedSchema.addRelationship('BC2', ('B', Schema.ONE), ('C', Schema.ONE))
        self.assertEqual(schema, expectedSchema)

        schema = SchemaGenerator.generateSchema(3, 5)
        self.assertEqual(5, len(schema.getRelationships()))


    def testGenerateRelationshipAttributes(self):
        schema = SchemaGenerator.generateSchema(2, 1, entityAttrDistribution=ConstantDistribution(0),
            relationshipAttrDistribution=ConstantDistribution(1), cardinalityDistribution=ConstantDistribution(Schema.ONE))
        expectedSchema = Schema()
        expectedSchema.addEntity('A')
        expectedSchema.addEntity('B')
        expectedSchema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        expectedSchema.addAttribute('AB', 'XY1')
        self.assertEqual(schema, expectedSchema)

        schema = SchemaGenerator.generateSchema(2, 2, entityAttrDistribution=ConstantDistribution(0),
            relationshipAttrDistribution=ConstantDistribution(2), cardinalityDistribution=ConstantDistribution(Schema.ONE))
        expectedSchema.addRelationship('AB2', ('A', Schema.ONE), ('B', Schema.ONE))
        expectedSchema.addAttribute('AB', 'XY2')
        expectedSchema.addAttribute('AB2', 'XY2_1')
        expectedSchema.addAttribute('AB2', 'XY2_2')
        self.assertEqual(schema, expectedSchema)


    def testCardinalities(self):
        schema = SchemaGenerator.generateSchema(2, 1, entityAttrDistribution=ConstantDistribution(0),
            relationshipAttrDistribution=ConstantDistribution(0), cardinalityDistribution=ConstantDistribution(Schema.ONE))
        expectedSchema = Schema()
        expectedSchema.addEntity('A')
        expectedSchema.addEntity('B')
        expectedSchema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        self.assertEqual(schema, expectedSchema)

        schema = SchemaGenerator.generateSchema(2, 1, entityAttrDistribution=ConstantDistribution(0),
            relationshipAttrDistribution=ConstantDistribution(0), cardinalityDistribution=ConstantDistribution(Schema.MANY))
        expectedSchema = Schema()
        expectedSchema.addEntity('A')
        expectedSchema.addEntity('B')
        expectedSchema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.MANY))
        self.assertEqual(schema, expectedSchema)


    def testBadNumEntitiesOrRelationships(self):
        TestUtil.assertRaisesMessage(self, Exception, "numEntities must be a non-negative integer",
            SchemaGenerator.generateSchema, None, 0)
        TestUtil.assertRaisesMessage(self, Exception, "numEntities must be a non-negative integer",
            SchemaGenerator.generateSchema, 1.5, 0)
        TestUtil.assertRaisesMessage(self, Exception, "numEntities must be a non-negative integer",
            SchemaGenerator.generateSchema, -1, 0)

        TestUtil.assertRaisesMessage(self, Exception, "numRelationships must be a non-negative integer",
            SchemaGenerator.generateSchema, 0, None)
        TestUtil.assertRaisesMessage(self, Exception, "numRelationships must be a non-negative integer",
            SchemaGenerator.generateSchema, 0, 1.5)
        TestUtil.assertRaisesMessage(self, Exception, "numRelationships must be a non-negative integer",
            SchemaGenerator.generateSchema, 0, -1)

        #  must have 2+ entities to have relationships
        TestUtil.assertRaisesMessage(self, Exception, "must have at least 2 entities to support a relationship: found 0",
            SchemaGenerator.generateSchema, 0, 1)
        TestUtil.assertRaisesMessage(self, Exception, "must have at least 2 entities to support a relationship: found 1",
            SchemaGenerator.generateSchema, 1, 1)


    def testBadDistributions(self):
        TestUtil.assertRaisesMessage(self, Exception, "entityAttrDistribution must be a MarginalDistribution",
            SchemaGenerator.generateSchema, 0, 0, entityAttrDistribution=None)

        TestUtil.assertRaisesMessage(self, Exception, "entityAttrDistribution must have a domain of non-negative "
                                                      "integers: returned None",
            SchemaGenerator.generateSchema, 1, 0, entityAttrDistribution=ConstantDistribution(None))

        TestUtil.assertRaisesMessage(self, Exception, "entityAttrDistribution must have a domain of non-negative "
                                                      "integers: returned 1.5",
            SchemaGenerator.generateSchema, 1, 0, entityAttrDistribution=ConstantDistribution(1.5))

        TestUtil.assertRaisesMessage(self, Exception, "entityAttrDistribution must have a domain of non-negative "
                                                      "integers: returned -1",
            SchemaGenerator.generateSchema, 1, 0, entityAttrDistribution=ConstantDistribution(-1))

        TestUtil.assertRaisesMessage(self, Exception, "relationshipAttrDistribution must be a MarginalDistribution",
            SchemaGenerator.generateSchema, 0, 0, relationshipAttrDistribution=None)

        TestUtil.assertRaisesMessage(self, Exception, "relationshipAttrDistribution must have a domain of non-negative "
                                                      "integers: returned None",
            SchemaGenerator.generateSchema, 2, 1, relationshipAttrDistribution=ConstantDistribution(None))

        TestUtil.assertRaisesMessage(self, Exception, "relationshipAttrDistribution must have a domain of non-negative "
                                                      "integers: returned 1.5",
            SchemaGenerator.generateSchema, 2, 1, relationshipAttrDistribution=ConstantDistribution(1.5))

        TestUtil.assertRaisesMessage(self, Exception, "relationshipAttrDistribution must have a domain of non-negative "
                                                      "integers: returned -1",
            SchemaGenerator.generateSchema, 2, 1, relationshipAttrDistribution=ConstantDistribution(-1))

        TestUtil.assertRaisesMessage(self, Exception, "cardinalityDistribution must be a MarginalDistribution",
            SchemaGenerator.generateSchema, 2, 1, cardinalityDistribution=None)

        TestUtil.assertRaisesMessage(self, Exception, "cardinalityDistribution must return either "
                                                      "Schema.ONE or Schema.MANY: returned None and None",
            SchemaGenerator.generateSchema, 2, 1, cardinalityDistribution=ConstantDistribution(None))


    def testOneRelationshipPerPairFlag(self):
        schema = SchemaGenerator.generateSchema(2, 1, relationshipAttrDistribution=ConstantDistribution(0),
            cardinalityDistribution=ConstantDistribution(Schema.ONE), oneRelationshipPerPair=True)
        expectedSchema = Schema()
        expectedSchema.addEntity('A')
        expectedSchema.addEntity('B')
        expectedSchema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        self.assertEqual([expectedSchema.getRelationship('AB')], schema.getRelationshipsBetweenEntities('A', 'B'))

        schema = SchemaGenerator.generateSchema(3, 2, relationshipAttrDistribution=ConstantDistribution(0),
            entityPairPicker=makePairPickerWithList(
                        [('A', 'B'), ('A', 'B'), ('A', 'C')]),
            cardinalityDistribution=ConstantDistribution(Schema.ONE), oneRelationshipPerPair=True)
        expectedSchema.addEntity('C')
        expectedSchema.addRelationship('AC', ('A', Schema.ONE), ('C', Schema.ONE))
        self.assertEqual([expectedSchema.getRelationship('AB')], schema.getRelationshipsBetweenEntities('A', 'B'))
        self.assertEqual([expectedSchema.getRelationship('AC')], schema.getRelationshipsBetweenEntities('A', 'C'))

        schema = SchemaGenerator.generateSchema(4, 4, relationshipAttrDistribution=ConstantDistribution(0),
            entityPairPicker=makePairPickerWithList(
                [('A', 'B'), ('A', 'B'), ('A', 'C'), ('A', 'D'),
                 ('A', 'D'), ('B', 'C')]),
            cardinalityDistribution=ConstantDistribution(Schema.ONE), oneRelationshipPerPair=True)
        expectedSchema.addEntity('D')
        expectedSchema.addRelationship('AD', ('A', Schema.ONE), ('D', Schema.ONE))
        expectedSchema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.ONE))
        self.assertEqual([expectedSchema.getRelationship('AB')], schema.getRelationshipsBetweenEntities('A', 'B'))
        self.assertEqual([expectedSchema.getRelationship('AC')], schema.getRelationshipsBetweenEntities('A', 'C'))
        self.assertEqual([expectedSchema.getRelationship('AD')], schema.getRelationshipsBetweenEntities('A', 'D'))
        self.assertEqual([expectedSchema.getRelationship('BC')], schema.getRelationshipsBetweenEntities('B', 'C'))


    def testTooManyRelationships(self):
        # Error condition interacting with oneRelationshipPerPair flag
        TestUtil.assertRaisesMessage(self, Exception, "Too many relationships requested: asked for 2, at most 1 possible",
            SchemaGenerator.generateSchema, 2, 2, oneRelationshipPerPair=True)
        TestUtil.assertRaisesMessage(self, Exception, "Too many relationships requested: asked for 7, at most 6 possible",
            SchemaGenerator.generateSchema, 4, 7, oneRelationshipPerPair=True)

        # Error condition interacting with allowCycles flag
        TestUtil.assertRaisesMessage(self, Exception, "Too many relationships requested: asked for 2, at most 1 possible",
            SchemaGenerator.generateSchema, 2, 2, allowCycles=False)
        TestUtil.assertRaisesMessage(self, Exception, "Too many relationships requested: asked for 4, at most 3 possible",
            SchemaGenerator.generateSchema, 4, 4, allowCycles=False)


    def testAllowCyclesFalse(self):
        schema = SchemaGenerator.generateSchema(4, 3, entityPairPicker=makePairPickerWithList(
            [('A', 'B'), ('B', 'C'), ('A', 'C'), ('C', 'D')]),
            oneRelationshipPerPair=True, allowCycles=False)
        self.assertEqual(3, len(schema.getRelationships()))
        self.assertEqual(1, len(schema.getRelationshipsBetweenEntities('A', 'B')))
        self.assertEqual(1, len(schema.getRelationshipsBetweenEntities('B', 'C')))
        self.assertEqual(1, len(schema.getRelationshipsBetweenEntities('C', 'D')))

        schema = SchemaGenerator.generateSchema(4, 3, entityPairPicker=makePairPickerWithList(
            [('C', 'D'), ('B', 'C'), ('B', 'D'), ('D', 'B'), ('A', 'D')]),
            oneRelationshipPerPair=True, allowCycles=False)
        self.assertEqual(3, len(schema.getRelationships()))
        self.assertEqual(1, len(schema.getRelationshipsBetweenEntities('D', 'C')))
        self.assertEqual(1, len(schema.getRelationshipsBetweenEntities('C', 'B')))
        self.assertEqual(1, len(schema.getRelationshipsBetweenEntities('D', 'A')))


def makePairPickerWithList(entityNamePairs):
    idx = 0
    def pairPicker(candidateEntityPairs, ignoredConstOne):
        nonlocal idx
        while idx < len(entityNamePairs):
            for ent1, ent2 in candidateEntityPairs:
                if (ent1.name == entityNamePairs[idx][0] and ent2.name == entityNamePairs[idx][1]) \
                    or (ent2.name == entityNamePairs[idx][0] and ent1.name == entityNamePairs[idx][1]):
                    idx += 1
                    return [(ent1, ent2)]
            idx += 1
    return pairPicker


if __name__ == '__main__':
    unittest.main()
