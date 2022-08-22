import unittest
from causality.model.RelationalDependency import RelationalDependency
from causality.test import TestUtil
from causality.model.Schema import Attribute
from causality.model.Schema import Schema
from causality.modelspace import RelationalSpace

class TestRelationalDependencySpace(unittest.TestCase):

    def testOneEntity(self):
        schema = Schema()
        schema.addEntity('A')
        relDeps = RelationalSpace.getRelationalDependencies(schema, 0)
        self.assertTrue(all([isinstance(relDep, RelationalDependency) for relDep in relDeps]))
        TestUtil.assertUnorderedListEqual(self, [], [str(relDep) for relDep in relDeps])

        schema.addAttribute('A', 'X1')
        relDeps = RelationalSpace.getRelationalDependencies(schema, 0)
        self.assertTrue(all([isinstance(relDep, RelationalDependency) for relDep in relDeps]))
        TestUtil.assertUnorderedListEqual(self, [], [str(relDep) for relDep in relDeps])

        schema.addAttribute('A', 'X2')
        relDeps = RelationalSpace.getRelationalDependencies(schema, 0)
        self.assertTrue(all([isinstance(relDep, RelationalDependency) for relDep in relDeps]))
        TestUtil.assertUnorderedListEqual(self, ['[A].X1 -> [A].X2', '[A].X2 -> [A].X1'], [str(relDep) for relDep in relDeps])

        schema.addAttribute('A', 'X3')
        relDeps = RelationalSpace.getRelationalDependencies(schema, 0)
        self.assertTrue(all([isinstance(relDep, RelationalDependency) for relDep in relDeps]))
        TestUtil.assertUnorderedListEqual(self, ['[A].X1 -> [A].X2', '[A].X2 -> [A].X1',
                                                 '[A].X1 -> [A].X3', '[A].X3 -> [A].X1',
                                                 '[A].X2 -> [A].X3', '[A].X3 -> [A].X2'], [str(relDep) for relDep in relDeps])

        relDeps = RelationalSpace.getRelationalDependencies(schema, 2)
        self.assertTrue(all([isinstance(relDep, RelationalDependency) for relDep in relDeps]))
        TestUtil.assertUnorderedListEqual(self, ['[A].X1 -> [A].X2', '[A].X2 -> [A].X1',
                                                 '[A].X1 -> [A].X3', '[A].X3 -> [A].X1',
                                                 '[A].X2 -> [A].X3', '[A].X3 -> [A].X2'], [str(relDep) for relDep in relDeps])

        schema.addEntity('B')
        relDeps = RelationalSpace.getRelationalDependencies(schema, 0)
        self.assertTrue(all([isinstance(relDep, RelationalDependency) for relDep in relDeps]))
        TestUtil.assertUnorderedListEqual(self, ['[A].X1 -> [A].X2', '[A].X2 -> [A].X1',
                                                 '[A].X1 -> [A].X3', '[A].X3 -> [A].X1',
                                                 '[A].X2 -> [A].X3', '[A].X3 -> [A].X2'], [str(relDep) for relDep in relDeps])

        schema.addAttribute('B', 'Y1')
        schema.addAttribute('B', 'Y2')
        relDeps = RelationalSpace.getRelationalDependencies(schema, 0)
        self.assertTrue(all([isinstance(relDep, RelationalDependency) for relDep in relDeps]))
        TestUtil.assertUnorderedListEqual(self, ['[A].X1 -> [A].X2', '[A].X2 -> [A].X1',
                                                 '[A].X1 -> [A].X3', '[A].X3 -> [A].X1',
                                                 '[A].X2 -> [A].X3', '[A].X3 -> [A].X2',
                                                 '[B].Y1 -> [B].Y2', '[B].Y2 -> [B].Y1',], [str(relDep) for relDep in relDeps])


    def testOneRelationship(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.MANY))

        schema.addAttribute('A', 'X')
        schema.addAttribute('B', 'Y')
        schema.addAttribute('AB', 'XY')

        relDeps = RelationalSpace.getRelationalDependencies(schema, 0, includeExistence=True)
        hop0 = []
        self.assertTrue(all([isinstance(relDep, RelationalDependency) for relDep in relDeps]))
        TestUtil.assertUnorderedListEqual(self, hop0, [str(relDep) for relDep in relDeps])

        relDeps = RelationalSpace.getRelationalDependencies(schema, 1, includeExistence=True)
        hop1 = ['[A, AB].XY -> [A].X', '[A, AB].exists -> [A].X', '[AB, A].X -> [AB].XY', '[AB, A].X -> [AB].exists',
                '[AB, B].Y -> [AB].XY', '[AB, B].Y -> [AB].exists', '[B, AB].XY -> [B].Y', '[B, AB].exists -> [B].Y']
        self.assertTrue(all([isinstance(relDep, RelationalDependency) for relDep in relDeps]))
        TestUtil.assertUnorderedListEqual(self, hop0 + hop1, [str(relDep) for relDep in relDeps])

        relDeps = RelationalSpace.getRelationalDependencies(schema, 2, includeExistence=True)
        hop2 = ['[A, AB, B].Y -> [A].X', '[AB, B, AB].exists -> [AB].XY', '[B, AB, A].X -> [B].Y']
        self.assertTrue(all([isinstance(relDep, RelationalDependency) for relDep in relDeps]))
        TestUtil.assertUnorderedListEqual(self, hop0 + hop1 + hop2, [str(relDep) for relDep in relDeps])

        relDeps = RelationalSpace.getRelationalDependencies(schema, 3, includeExistence=True)
        hop3 = ['[A, AB, B, AB].XY -> [A].X', '[A, AB, B, AB].exists -> [A].X', '[AB, B, AB, A].X -> [AB].XY']
        self.assertTrue(all([isinstance(relDep, RelationalDependency) for relDep in relDeps]))
        TestUtil.assertUnorderedListEqual(self, hop0 + hop1 + hop2 + hop3, [str(relDep) for relDep in relDeps])


    def testTwoRelationships(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addEntity('C')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.ONE))
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.MANY))
        schema.addAttribute('A', 'X')
        schema.addAttribute('B', 'Y')
        schema.addAttribute('C', 'Z')
        schema.addAttribute('AB', 'XY')
        schema.addAttribute('BC', 'YZ')

        hop0 = []

        hop1 = ['[A, AB].exists -> [A].X', '[B, AB].exists -> [B].Y', '[B, BC].exists -> [B].Y', '[C, BC].exists -> [C].Z',
                '[A, AB].XY -> [A].X', '[B, AB].XY -> [B].Y', '[B, BC].YZ -> [B].Y', '[C, BC].YZ -> [C].Z',
                '[AB, A].X -> [AB].exists', '[AB, A].X -> [AB].XY', '[AB, B].Y -> [AB].exists', '[AB, B].Y -> [AB].XY',
                '[BC, C].Z -> [BC].exists', '[BC, C].Z -> [BC].YZ', '[BC, B].Y -> [BC].exists', '[BC, B].Y -> [BC].YZ']

        hop2 = ['[AB, A, AB].exists -> [AB].XY', '[AB, B, BC].exists -> [AB].exists', '[AB, B, BC].exists -> [AB].XY',
                '[BC, C, BC].exists -> [BC].YZ', '[BC, B, AB].exists -> [BC].exists', '[BC, B, AB].exists -> [BC].YZ',
                '[A, AB, B].Y -> [A].X', '[B, AB, A].X -> [B].Y', '[B, BC, C].Z -> [B].Y', '[C, BC, B].Y -> [C].Z',
                '[AB, B, BC].YZ -> [AB].exists', '[AB, B, BC].YZ -> [AB].XY',
                '[BC, B, AB].XY -> [BC].exists', '[BC, B, AB].XY -> [BC].YZ']

        hop3 = ['[A, AB, B, BC].exists -> [A].X', '[B, AB, A, AB].exists -> [B].Y', '[B, BC, C, BC].exists -> [B].Y',
                '[C, BC, B, AB].exists -> [C].Z', '[A, AB, B, BC].YZ -> [A].X', '[B, AB, A, AB].XY -> [B].Y',
                '[B, BC, C, BC].YZ -> [B].Y', '[C, BC, B, AB].XY -> [C].Z', '[AB, A, AB, B].Y -> [AB].XY',
                '[AB, B, BC, C].Z -> [AB].exists', '[AB, B, BC, C].Z -> [AB].XY', '[BC, C, BC, B].Y -> [BC].YZ',
                '[BC, B, AB, A].X -> [BC].exists', '[BC, B, AB, A].X -> [BC].YZ']

        relDeps = RelationalSpace.getRelationalDependencies(schema, 3, includeExistence=True)
        self.assertTrue(all([isinstance(relDep, RelationalDependency) for relDep in relDeps]))
        TestUtil.assertUnorderedListEqual(self, hop0 + hop1 + hop2 + hop3, [str(relDep) for relDep in relDeps])


    def testNoExistenceDependencies(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addEntity('C')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.ONE))
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.MANY))
        schema.addAttribute('A', 'X')
        schema.addAttribute('B', 'Y')
        schema.addAttribute('C', 'Z')
        schema.addAttribute('AB', 'XY')
        schema.addAttribute('BC', 'YZ')

        hop0 = []

        hop1 = ['[A, AB].XY -> [A].X', '[B, AB].XY -> [B].Y', '[B, BC].YZ -> [B].Y', '[C, BC].YZ -> [C].Z',
                '[AB, A].X -> [AB].XY', '[AB, B].Y -> [AB].XY', '[BC, C].Z -> [BC].YZ', '[BC, B].Y -> [BC].YZ']

        hop2 = ['[A, AB, B].Y -> [A].X', '[B, AB, A].X -> [B].Y', '[B, BC, C].Z -> [B].Y', '[C, BC, B].Y -> [C].Z',
                '[AB, B, BC].YZ -> [AB].XY', '[BC, B, AB].XY -> [BC].YZ']

        hop3 = ['[A, AB, B, BC].YZ -> [A].X', '[B, AB, A, AB].XY -> [B].Y', '[B, BC, C, BC].YZ -> [B].Y',
                '[C, BC, B, AB].XY -> [C].Z', '[AB, A, AB, B].Y -> [AB].XY', '[AB, B, BC, C].Z -> [AB].XY',
                '[BC, C, BC, B].Y -> [BC].YZ', '[BC, B, AB, A].X -> [BC].YZ']

        relDeps = RelationalSpace.getRelationalDependencies(schema, 3)
        self.assertTrue(all([isinstance(relDep, RelationalDependency) for relDep in relDeps]))
        TestUtil.assertUnorderedListEqual(self, hop0 + hop1 + hop2 + hop3, [str(relDep) for relDep in relDeps])


if __name__ == '__main__':
    unittest.main()
