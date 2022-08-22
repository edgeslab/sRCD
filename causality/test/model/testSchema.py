import unittest
from causality.test import TestUtil
from causality.model.Schema import Attribute
from causality.model.Schema import Entity
from causality.model.Schema import Relationship
from causality.model.Schema import Schema

class TestSchema(unittest.TestCase):

    def testSchemaEntities(self):
        schemaWithEntityA = Schema()
        schemaWithEntityA.addEntity('A')
        actualEntities = schemaWithEntityA.getEntities()
        TestUtil.assertUnorderedListEqual(self, ['A'], [ent.name for ent in actualEntities])

        schemaWithEntityA.addEntity('B')
        actualEntities = schemaWithEntityA.getEntities()
        TestUtil.assertUnorderedListEqual(self, ['A', 'B'], [ent.name for ent in actualEntities])


    def testInvalidAttrNames(self):
        a = Entity('A')
        TestUtil.assertRaisesMessage(self, Exception, "Cannot add attribute with reserved name 'exists'",
            a.addAttribute, 'exists', Attribute.INTEGER)

        ab = Relationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        TestUtil.assertRaisesMessage(self, Exception, "Cannot add attribute with reserved name 'exists'",
            ab.addAttribute, 'exists', Attribute.INTEGER)


    def testAttributeDataTypes(self):
        # valid data types cause no exceptions
        for dataType in Attribute.DATA_TYPES.values():
            Attribute('name', dataType)

        TestUtil.assertRaisesMessage(self, Exception, "unknown dataType found: -1",
             Attribute, 'name', -1)


    def testDuplicateERNames(self):
        schemaWithEntityA = Schema()
        schemaWithEntityA.addEntity('A')
        TestUtil.assertRaisesMessage(self, Exception, "Schema already has entity named 'A'",
            schemaWithEntityA.addEntity, 'A')

        schemaWithEntityA.addEntity('B')
        schemaWithEntityA.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        TestUtil.assertRaisesMessage(self, Exception, "Schema already has relationship named 'AB'",
            schemaWithEntityA.addRelationship, 'AB', ('A', Schema.ONE), ('B', Schema.ONE))

        TestUtil.assertRaisesMessage(self, Exception, "Schema already has entity named 'A'",
            schemaWithEntityA.addRelationship, 'A', ('A', Schema.ONE), ('B', Schema.ONE))
        TestUtil.assertRaisesMessage(self, Exception, "Schema already has relationship named 'AB'",
            schemaWithEntityA.addEntity, 'AB')


    def testDuplicateAttrNames(self):
        schemaWithEntityA = Schema()
        schemaWithEntityA.addEntity('A')
        schemaWithEntityA.addAttribute('A', 'X', Attribute.INTEGER)
        TestUtil.assertRaisesMessage(self, Exception, "Schema already has attribute named 'X' for item 'A'",
            schemaWithEntityA.addAttribute, 'A', 'X', Attribute.INTEGER)

        schemaWithEntityA.addEntity('B')
        schemaWithEntityA.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        schemaWithEntityA.addAttribute('AB', 'XY', Attribute.INTEGER)
        TestUtil.assertRaisesMessage(self, Exception, "Schema already has attribute named 'XY' for item 'AB'",
            schemaWithEntityA.addAttribute, 'AB', 'XY', Attribute.INTEGER)

        a = schemaWithEntityA.getEntity('A')
        TestUtil.assertRaisesMessage(self, Exception, "Schema already has attribute named 'X' for item 'A'",
            a.addAttribute, 'X', Attribute.INTEGER)

        ab = schemaWithEntityA.getRelationship('AB')
        TestUtil.assertRaisesMessage(self, Exception, "Schema already has attribute named 'XY' for item 'AB'",
            ab.addAttribute, 'XY', Attribute.INTEGER)



    def testCardinalities(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        TestUtil.assertRaisesMessage(self, Exception, "Bad cardinality for entity1 or entity2: xxxx one. Should be either Schema.ONE or Schema.MANY",
            schema.addRelationship, 'AB', ('A', 'xxxx'), ('B', Schema.ONE))
        TestUtil.assertRaisesMessage(self, Exception, "Bad cardinality for entity1 or entity2: many xxxx. Should be either Schema.ONE or Schema.MANY",
            schema.addRelationship, 'AB', ('A', Schema.MANY), ('B', 'xxxx'))
        TestUtil.assertRaisesMessage(self, Exception, "Bad cardinality for entity1 or entity2: xxxx many. Should be either Schema.ONE or Schema.MANY",
            schema.addRelationship, 'AB', ('A', 'xxxx'), ('B', Schema.MANY))
        TestUtil.assertRaisesMessage(self, Exception, "Bad cardinality for entity1 or entity2: one xxxx. Should be either Schema.ONE or Schema.MANY",
            schema.addRelationship, 'AB', ('A', Schema.ONE), ('B', 'xxxx'))

        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        ab = schema.getRelationship('AB')
        self.assertEqual(Schema.ONE, ab.getCardinality('A'))
        self.assertEqual(Schema.ONE, ab.getCardinality('B'))

        schema.addEntity('C')
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.MANY))
        bc = schema.getRelationship('BC')
        self.assertEqual(Schema.ONE, bc.getCardinality('B'))
        self.assertEqual(Schema.MANY, bc.getCardinality('C'))
        TestUtil.assertRaisesMessage(self, Exception, "Entity 'A' does not exist for relationship 'BC'",
            bc.getCardinality, 'A')


    def testSchemaRelationships(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        actualRelationships = schema.getRelationships()
        actualNames = [rel.name for rel in actualRelationships]
        TestUtil.assertUnorderedListEqual(self, ['AB'], actualNames)

        schema.addEntity('C')
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.ONE))
        actualRelationships = schema.getRelationships()
        actualNames = [rel.name for rel in actualRelationships]
        TestUtil.assertUnorderedListEqual(self, ['AB', 'BC'], actualNames)

        # verify that entity names for relationships actually exist
        TestUtil.assertRaisesMessage(self, Exception, "One of entity1Name or entity2Name not listed in entityNames: 'C', 'D'",
            schema.addRelationship, 'CD', ('C', Schema.ONE), ('D', Schema.ONE))


    def testGetRelationship(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        actualRelationship = schema.getRelationship('AB')
        self.assertEqual('AB', actualRelationship.name)
        self.assertTrue(schema.hasRelationship('AB'))

        schema.addEntity('C')
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.ONE))
        actualRelationship = schema.getRelationship('BC')
        self.assertEqual('BC', actualRelationship.name)
        self.assertTrue(schema.hasRelationship('BC'))

        TestUtil.assertRaisesMessage(self, Exception, "Relationship 'XX' does not exist",
            schema.getRelationship, 'XX')
        self.assertFalse(schema.hasRelationship('XX'))


    def testERMembership(self):
        ab = Relationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        self.assertTrue(ab.hasEntity('A'))
        self.assertTrue(ab.hasEntity('B'))
        self.assertFalse(ab.hasEntity('C'))


    def testGetEntity(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        actualEntity = schema.getEntity('A')
        self.assertEqual('A', actualEntity.name)
        self.assertTrue(schema.hasEntity('A'))

        actualEntity = schema.getEntity('B')
        self.assertEqual('B', actualEntity.name)
        self.assertTrue(schema.hasEntity('B'))

        TestUtil.assertRaisesMessage(self, Exception, "Entity 'XX' does not exist",
            schema.getEntity, 'XX')
        self.assertFalse(schema.hasEntity('XX'))


    def testAddEntityAttribute(self):
        a = Entity('A')
        a.addAttribute('X', Attribute.INTEGER)
        TestUtil.assertUnorderedListEqual(self, ['X'], [attr.name for attr in a.attributes])

        a.addAttribute('Y', Attribute.INTEGER)
        TestUtil.assertUnorderedListEqual(self, ['X', 'Y'], [attr.name for attr in a.attributes])

        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X', Attribute.INTEGER)
        TestUtil.assertUnorderedListEqual(self, ['X'], [attr.name for attr in schema.getEntity('A').attributes])

        schema.addAttribute('A', 'Y', Attribute.INTEGER)
        TestUtil.assertUnorderedListEqual(self, ['X', 'Y'], [attr.name for attr in schema.getEntity('A').attributes])


    def testAddRelationshipAttribute(self):
        ab = Relationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        ab.addAttribute('XY1', Attribute.INTEGER)
        TestUtil.assertUnorderedListEqual(self, ['XY1'], [attr.name for attr in ab.attributes])

        ab.addAttribute('XY2', Attribute.INTEGER)
        TestUtil.assertUnorderedListEqual(self, ['XY1', 'XY2'], [attr.name for attr in ab.attributes])

        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        schema.addAttribute('AB', 'XY1', Attribute.INTEGER)
        TestUtil.assertUnorderedListEqual(self, ['XY1'], [attr.name for attr in schema.getRelationship('AB').attributes])

        schema.addAttribute('AB', 'XY2', Attribute.INTEGER)
        TestUtil.assertUnorderedListEqual(self, ['XY1', 'XY2'], [attr.name for attr in schema.getRelationship('AB').attributes])


    def testGetSchemaItems(self):
        schema = Schema()
        schema.addEntity('A')
        actualItems = schema.getSchemaItems()
        TestUtil.assertUnorderedListEqual(self, ['A'], [item.name for item in actualItems])
        actualItem = schema.getSchemaItem('A')
        self.assertEqual('A', actualItem.name)
        self.assertTrue(schema.hasSchemaItem('A'))

        schema.addEntity('B')
        actualItems = schema.getSchemaItems()
        TestUtil.assertUnorderedListEqual(self, ['A', 'B'], [item.name for item in actualItems])
        actualItem = schema.getSchemaItem('B')
        self.assertEqual('B', actualItem.name)
        self.assertTrue(schema.hasSchemaItem('B'))

        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        actualItems = schema.getSchemaItems()
        TestUtil.assertUnorderedListEqual(self, ['A', 'B', 'AB'], [item.name for item in actualItems])
        actualItem = schema.getSchemaItem('AB')
        self.assertEqual('AB', actualItem.name)
        self.assertTrue(schema.hasSchemaItem('AB'))

        TestUtil.assertRaisesMessage(self, Exception, "Schema item 'XX' does not exist",
            schema.getSchemaItem, 'XX')
        self.assertFalse(schema.hasSchemaItem('XX'))


    def testGetRelationshipsForEntity(self):
        schema = Schema()
        schema.addEntity('A')
        TestUtil.assertUnorderedListEqual(self, [], schema.getRelationshipsForEntity('A'))

        schema.addEntity('B')
        TestUtil.assertUnorderedListEqual(self, [], schema.getRelationshipsForEntity('B'))

        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        ab = schema.getRelationship('AB')
        TestUtil.assertUnorderedListEqual(self, [ab], schema.getRelationshipsForEntity('A'))
        TestUtil.assertUnorderedListEqual(self, [ab], schema.getRelationshipsForEntity('B'))

        schema.addEntity('C')
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.ONE))
        bc = schema.getRelationship('BC')
        TestUtil.assertUnorderedListEqual(self, [ab], schema.getRelationshipsForEntity('A'))
        TestUtil.assertUnorderedListEqual(self, [ab, bc], schema.getRelationshipsForEntity('B'))
        TestUtil.assertUnorderedListEqual(self, [bc], schema.getRelationshipsForEntity('C'))

        # test for bad entity names
        TestUtil.assertRaisesMessage(self, Exception, "Entity 'XX' does not exist",
            schema.getRelationshipsForEntity, 'XX')


    def testGetRelationshipsBetweenEntities(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        TestUtil.assertUnorderedListEqual(self, [], schema.getRelationshipsBetweenEntities('A', 'B'))
        TestUtil.assertUnorderedListEqual(self, [], schema.getRelationshipsBetweenEntities('B', 'A'))

        schema.addEntity('C')
        TestUtil.assertUnorderedListEqual(self, [], schema.getRelationshipsBetweenEntities('B', 'C'))
        TestUtil.assertUnorderedListEqual(self, [], schema.getRelationshipsBetweenEntities('C', 'B'))

        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        ab = schema.getRelationship('AB')
        TestUtil.assertUnorderedListEqual(self, [ab], schema.getRelationshipsBetweenEntities('A', 'B'))
        TestUtil.assertUnorderedListEqual(self, [ab], schema.getRelationshipsBetweenEntities('B', 'A'))

        schema.addRelationship('AB2', ('A', Schema.ONE), ('B', Schema.ONE))
        ab2 = schema.getRelationship('AB2')
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.ONE))
        bc = schema.getRelationship('BC')
        TestUtil.assertUnorderedListEqual(self, [ab, ab2], schema.getRelationshipsBetweenEntities('A', 'B'))
        TestUtil.assertUnorderedListEqual(self, [ab, ab2], schema.getRelationshipsBetweenEntities('B', 'A'))
        TestUtil.assertUnorderedListEqual(self, [bc], schema.getRelationshipsBetweenEntities('B', 'C'))

        # test for bad entity names
        TestUtil.assertRaisesMessage(self, Exception, "Entity 'XX' does not exist",
            schema.getRelationshipsBetweenEntities, 'XX', 'A')
        TestUtil.assertRaisesMessage(self, Exception, "Entity 'XX' does not exist",
            schema.getRelationshipsBetweenEntities, 'A', 'XX')


    def testEntityEq(self):
        a1 = Entity('A')
        a2 = Entity('A')
        self.assertEqual(a1, a2)

        a1.addAttribute('X1', Attribute.INTEGER)
        a2.addAttribute('X1', Attribute.INTEGER)
        self.assertEqual(a1, a2)

        b1 = Entity('B')
        b2 = Entity('B')
        b1.addAttribute('Y1', Attribute.INTEGER)
        b1.addAttribute('Y2', Attribute.INTEGER)
        b2.addAttribute('Y2', Attribute.INTEGER)
        b2.addAttribute('Y1', Attribute.INTEGER)
        self.assertEqual(b1, b2)

        b2.addAttribute('Y3', Attribute.INTEGER)
        self.assertNotEqual(b1, b2)


    def testRelationshipEq(self):
        ab1 = Relationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        ab2 = Relationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        self.assertEqual(ab1, ab2)

        ab1 = Relationship('AB', ('A', Schema.ONE), ('B', Schema.MANY))
        ab2 = Relationship('AB', ('A', Schema.ONE), ('B', Schema.MANY))
        self.assertEqual(ab1, ab2)

        ab1.addAttribute('XY1', Attribute.INTEGER)
        ab1.addAttribute('XY2', Attribute.INTEGER)
        ab2.addAttribute('XY2', Attribute.INTEGER)
        ab2.addAttribute('XY1', Attribute.INTEGER)
        self.assertEqual(ab1, ab2)

        ab1 = Relationship('AB', ('B', Schema.ONE), ('A', Schema.ONE))
        ab2 = Relationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        self.assertEqual(ab1, ab2)


    def testSchemaEq(self):
        schema1 = Schema()
        schema2 = Schema()
        self.assertEqual(schema1, schema2)

        schema1.addEntity('A')
        schema2.addEntity('A')
        self.assertEqual(schema1, schema2)

        schema1.addEntity('B')
        schema2.addEntity('B')
        self.assertEqual(schema1, schema2)

        schema1.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        schema2.addRelationship('AB', ('B', Schema.ONE), ('A', Schema.ONE))
        self.assertEqual(schema1, schema2)

        schema1.addEntity('C')
        schema2.addEntity('C')
        schema1.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.MANY))
        schema2.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.MANY))
        self.assertEqual(schema1, schema2)

        schema1.addAttribute('A', 'X', Attribute.INTEGER)
        schema1.addAttribute('B', 'Y', Attribute.INTEGER)
        schema1.addAttribute('C', 'Z', Attribute.INTEGER)
        schema2.addAttribute('A', 'X', Attribute.INTEGER)
        schema2.addAttribute('B', 'Y', Attribute.INTEGER)
        schema2.addAttribute('C', 'Z', Attribute.INTEGER)
        self.assertEqual(schema1, schema2)

        schema1.addAttribute('AB', 'XY1', Attribute.INTEGER)
        schema1.addAttribute('AB', 'XY2', Attribute.INTEGER)
        schema1.addAttribute('BC', 'YZ', Attribute.INTEGER)
        schema2.addAttribute('AB', 'XY1', Attribute.INTEGER)
        schema2.addAttribute('AB', 'XY2', Attribute.INTEGER)
        schema2.addAttribute('BC', 'YZ', Attribute.INTEGER)
        self.assertEqual(schema1, schema2)

        schema1.addAttribute('A', 'X2', Attribute.INTEGER)
        schema1.addAttribute('A', 'X3', Attribute.INTEGER)
        schema2.addAttribute('A', 'X3', Attribute.INTEGER)
        schema2.addAttribute('A', 'X2', Attribute.INTEGER)
        self.assertEqual(schema1, schema2)

        schema2.addAttribute('B', 'Y2', Attribute.INTEGER)
        self.assertNotEqual(schema1, schema2)


if __name__ == '__main__':
    unittest.main()
