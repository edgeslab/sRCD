import unittest
from causality.test import TestUtil
from causality.model.Schema import Schema
from causality.modelspace import RelationalSpace

class TestRelationalPathSpace(unittest.TestCase):

    def testOneEntity(self):
        schema = Schema()
        schema.addEntity('A')
        relPaths = RelationalSpace.getRelationalPaths(schema, 0)
        TestUtil.assertUnorderedListEqual(self, [['A']], relPaths)

        schema = Schema()
        schema.addEntity('B')
        relPaths = RelationalSpace.getRelationalPaths(schema, 0)
        TestUtil.assertUnorderedListEqual(self, [['B']], relPaths)

        schema.addEntity('A')
        relPaths = RelationalSpace.getRelationalPaths(schema, 0)
        TestUtil.assertUnorderedListEqual(self, [['A'], ['B']], relPaths)


    def testOneRelationshipManyToMany(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.MANY))

        relPaths = RelationalSpace.getRelationalPaths(schema, 0)
        hop0 = [['A'], ['B'], ['AB']]
        TestUtil.assertUnorderedListEqual(self, hop0, relPaths)

        relPaths = RelationalSpace.getRelationalPaths(schema, 1)
        hop1 = [['A', 'AB'], ['AB', 'A'], ['AB', 'B'], ['B', 'AB']]
        TestUtil.assertUnorderedListEqual(self, hop0 + hop1, relPaths)

        relPaths = RelationalSpace.getRelationalPaths(schema, 2)
        hop2 = [['A', 'AB', 'B'], ['AB', 'A', 'AB'], ['AB', 'B', 'AB'], ['B', 'AB', 'A']]
        TestUtil.assertUnorderedListEqual(self, hop0 + hop1 + hop2, relPaths)

        relPaths = RelationalSpace.getRelationalPaths(schema, 3)
        hop3 = [['A', 'AB', 'B', 'AB'], ['AB', 'A', 'AB', 'B'], ['AB', 'B', 'AB', 'A'], ['B', 'AB', 'A', 'AB']]
        TestUtil.assertUnorderedListEqual(self, hop0 + hop1 + hop2 + hop3, relPaths)


    def testOneRelationshipOneToMany(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.ONE))

        relPaths = RelationalSpace.getRelationalPaths(schema, 0)
        hop0 = [['A'], ['B'], ['AB']]
        TestUtil.assertUnorderedListEqual(self, hop0, relPaths)

        relPaths = RelationalSpace.getRelationalPaths(schema, 1)
        hop1 = [['A', 'AB'], ['AB', 'A'], ['AB', 'B'], ['B', 'AB']]
        TestUtil.assertUnorderedListEqual(self, hop0 + hop1 , relPaths)

        relPaths = RelationalSpace.getRelationalPaths(schema, 2)
        hop2 = [['A', 'AB', 'B'], ['AB', 'A', 'AB'], ['B', 'AB', 'A']]
        TestUtil.assertUnorderedListEqual(self, hop0 + hop1 + hop2, relPaths)

        relPaths = RelationalSpace.getRelationalPaths(schema, 3)
        hop3 = [['AB', 'A', 'AB', 'B'], ['B', 'AB', 'A', 'AB']]
        TestUtil.assertUnorderedListEqual(self, hop0 + hop1 + hop2 + hop3, relPaths)

        relPaths = RelationalSpace.getRelationalPaths(schema, 4)
        hop4 = [['B', 'AB', 'A', 'AB', 'B']]
        TestUtil.assertUnorderedListEqual(self, hop0 + hop1 + hop2 + hop3 + hop4, relPaths)

        relPaths = RelationalSpace.getRelationalPaths(schema, 5)
        hop5 = []
        TestUtil.assertUnorderedListEqual(self, hop0 + hop1 + hop2 + hop3 + hop4 + hop5, relPaths)


    def testOneRelationshipManyToOne(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.MANY))

        relPaths = RelationalSpace.getRelationalPaths(schema, 3)
        hopUpTo3 = [['A'], ['B'], ['AB'], ['A', 'AB'], ['AB', 'A'], ['AB', 'B'], ['B', 'AB'],
                    ['A', 'AB', 'B'], ['AB', 'B', 'AB'], ['B', 'AB', 'A'],
                    ['A', 'AB', 'B', 'AB'], ['AB', 'B', 'AB', 'A']]
        TestUtil.assertUnorderedListEqual(self, hopUpTo3, relPaths)


    def testOneRelationshipOneToOne(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))

        relPaths = RelationalSpace.getRelationalPaths(schema, 0)
        hop0 = [['A'], ['B'], ['AB']]
        TestUtil.assertUnorderedListEqual(self, hop0, relPaths)

        relPaths = RelationalSpace.getRelationalPaths(schema, 1)
        hop1 = [['A', 'AB'], ['AB', 'A'], ['AB', 'B'], ['B', 'AB']]
        TestUtil.assertUnorderedListEqual(self, hop0 + hop1, relPaths)

        schema = Schema()
        schema.addEntity('B')
        schema.addEntity('C')
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.ONE))

        relPaths = RelationalSpace.getRelationalPaths(schema, 0)
        hop0 = [['B'], ['C'], ['BC']]
        TestUtil.assertUnorderedListEqual(self, hop0, relPaths)

        relPaths = RelationalSpace.getRelationalPaths(schema, 1)
        hop1 = [['B', 'BC'], ['BC', 'B'], ['BC', 'C'], ['C', 'BC']]
        TestUtil.assertUnorderedListEqual(self, hop0 + hop1, relPaths)

        relPaths = RelationalSpace.getRelationalPaths(schema, 2)
        hop2 = [['B', 'BC', 'C'], ['C', 'BC', 'B']]
        TestUtil.assertUnorderedListEqual(self, hop0 + hop1 + hop2, relPaths)

        relPaths = RelationalSpace.getRelationalPaths(schema, 3)
        hop3 = []
        TestUtil.assertUnorderedListEqual(self, hop0 + hop1 + hop2 + hop3, relPaths)


    def testTwoRelationships(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addEntity('C')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.ONE))
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.MANY))

        relPaths = RelationalSpace.getRelationalPaths(schema, 0)
        hop0 = [['A'], ['B'], ['C'], ['AB'], ['BC']]
        TestUtil.assertUnorderedListEqual(self, hop0, relPaths)

        relPaths = RelationalSpace.getRelationalPaths(schema, 1)
        hop1 = [['A', 'AB'], ['B', 'AB'], ['B', 'BC'], ['C', 'BC'], ['AB', 'A'], ['AB', 'B'], ['BC', 'C'], ['BC', 'B']]
        TestUtil.assertUnorderedListEqual(self, hop0 + hop1, relPaths)

        relPaths = RelationalSpace.getRelationalPaths(schema, 2)
        hop2 = [['A', 'AB', 'B'], ['B', 'AB', 'A'], ['B', 'BC', 'C'], ['C', 'BC', 'B'], ['AB', 'A', 'AB'],
                ['AB', 'B', 'BC'], ['BC', 'C', 'BC'], ['BC', 'B', 'AB']]
        TestUtil.assertUnorderedListEqual(self, hop0 + hop1 + hop2, relPaths)

        relPaths = RelationalSpace.getRelationalPaths(schema, 3)
        hop3 = [['A', 'AB', 'B', 'BC'], ['B', 'AB', 'A', 'AB'], ['B', 'BC', 'C', 'BC'], ['C', 'BC', 'B', 'AB'],
                ['AB', 'A', 'AB', 'B'], ['AB', 'B', 'BC', 'C'], ['BC', 'C', 'BC', 'B'], ['BC', 'B', 'AB', 'A']]
        TestUtil.assertUnorderedListEqual(self, hop0 + hop1 + hop2 + hop3, relPaths)


    def testTwoRelationshipsTwoEntities(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB1', ('A', Schema.MANY), ('B', Schema.MANY))
        schema.addRelationship('AB2', ('A', Schema.MANY), ('B', Schema.ONE))

        relPaths = RelationalSpace.getRelationalPaths(schema, 0)
        hop0 = [['A'], ['B'], ['AB1'], ['AB2']]
        TestUtil.assertUnorderedListEqual(self, hop0, relPaths)

        relPaths = RelationalSpace.getRelationalPaths(schema, 1)
        hop1 = [['A', 'AB1'], ['AB1', 'A'], ['AB1', 'B'], ['B', 'AB1'], ['A', 'AB2'], ['AB2', 'A'], ['AB2', 'B'], ['B', 'AB2']]
        TestUtil.assertUnorderedListEqual(self, hop0 + hop1, relPaths)

        relPaths = RelationalSpace.getRelationalPaths(schema, 2)
        hop2 = [['A', 'AB1', 'B'], ['AB1', 'A', 'AB1'], ['AB1', 'B', 'AB1'], ['B', 'AB1', 'A'],
                ['A', 'AB2', 'B'], ['AB2', 'A', 'AB2'], ['B', 'AB2', 'A'],
                ['AB1', 'A', 'AB2'], ['AB1', 'B', 'AB2'], ['AB2', 'A', 'AB1'], ['AB2', 'B', 'AB1']]
        TestUtil.assertUnorderedListEqual(self, hop0 + hop1 + hop2, relPaths)


    def testBadHopThreshold(self):
        schema = Schema()
        TestUtil.assertRaisesMessage(self, Exception, "Hop threshold must be a number: found 'XX'",
            RelationalSpace.getRelationalPaths, schema, 'XX')

        TestUtil.assertRaisesMessage(self, Exception, "Hop threshold must be >= 0: found -1",
            RelationalSpace.getRelationalPaths, schema, -1)


if __name__ == '__main__':
    unittest.main()
