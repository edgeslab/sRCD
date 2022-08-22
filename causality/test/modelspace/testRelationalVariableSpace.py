import unittest
from causality.model.RelationalDependency import RelationalVariable
from causality.test import TestUtil
from causality.model.Schema import Schema
from causality.modelspace import RelationalSpace

class TestRelationalVariableSpace(unittest.TestCase):

    def testOneEntity(self):
        schema = Schema()
        schema.addEntity('A')
        relVars = RelationalSpace.getRelationalVariables(schema, 0, includeExistence=True)
        self.assertTrue(all([isinstance(relVar, RelationalVariable) for relVar in relVars]))
        TestUtil.assertUnorderedListEqual(self, ['[A].exists'], [str(relVar) for relVar in relVars])

        schema.addAttribute('A', 'X1')
        relVars = RelationalSpace.getRelationalVariables(schema, 0, includeExistence=True)
        self.assertTrue(all([isinstance(relVar, RelationalVariable) for relVar in relVars]))
        TestUtil.assertUnorderedListEqual(self, ['[A].exists', '[A].X1'], [str(relVar) for relVar in relVars])

        schema.addAttribute('A', 'X2')
        relVars = RelationalSpace.getRelationalVariables(schema, 0, includeExistence=True)
        self.assertTrue(all([isinstance(relVar, RelationalVariable) for relVar in relVars]))
        TestUtil.assertUnorderedListEqual(self, ['[A].exists', '[A].X1', '[A].X2'], [str(relVar) for relVar in relVars])

        schema = Schema()
        schema.addEntity('B')
        relVars = RelationalSpace.getRelationalVariables(schema, 0, includeExistence=True)
        self.assertTrue(all([isinstance(relVar, RelationalVariable) for relVar in relVars]))
        TestUtil.assertUnorderedListEqual(self, ['[B].exists'], [str(relVar) for relVar in relVars])

        schema.addAttribute('B', 'Y')
        relVars = RelationalSpace.getRelationalVariables(schema, 0, includeExistence=True)
        self.assertTrue(all([isinstance(relVar, RelationalVariable) for relVar in relVars]))
        TestUtil.assertUnorderedListEqual(self, ['[B].exists', '[B].Y'], [str(relVar) for relVar in relVars])

        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        relVars = RelationalSpace.getRelationalVariables(schema, 0, includeExistence=True)
        self.assertTrue(all([isinstance(relVar, RelationalVariable) for relVar in relVars]))
        TestUtil.assertUnorderedListEqual(self, ['[A].exists', '[A].X', '[B].exists', '[B].Y'],
            [str(relVar) for relVar in relVars])


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
        
        hop0 = ['[A].exists', '[B].exists', '[C].exists', '[AB].exists', '[BC].exists',
                '[A].X', '[B].Y', '[C].Z', '[AB].XY', '[BC].YZ']

        hop1 = ['[A, AB].exists', '[B, AB].exists', '[B, BC].exists', '[C, BC].exists', '[AB, A].exists',
                '[AB, B].exists', '[BC, C].exists', '[BC, B].exists',
                '[A, AB].XY', '[B, AB].XY', '[B, BC].YZ', '[C, BC].YZ', '[AB, A].X', 
                                '[AB, B].Y', '[BC, C].Z', '[BC, B].Y']

        hop2 = ['[A, AB, B].exists', '[B, AB, A].exists', '[B, BC, C].exists', '[C, BC, B].exists', '[AB, A, AB].exists',
                '[AB, B, BC].exists', '[BC, C, BC].exists', '[BC, B, AB].exists',
                '[A, AB, B].Y', '[B, AB, A].X', '[B, BC, C].Z', '[C, BC, B].Y', '[AB, A, AB].XY',
                                '[AB, B, BC].YZ', '[BC, C, BC].YZ', '[BC, B, AB].XY']

        hop3 = ['[A, AB, B, BC].exists', '[B, AB, A, AB].exists', '[B, BC, C, BC].exists', '[C, BC, B, AB].exists',
                '[AB, A, AB, B].exists', '[AB, B, BC, C].exists', '[BC, C, BC, B].exists', '[BC, B, AB, A].exists',
                '[A, AB, B, BC].YZ', '[B, AB, A, AB].XY', '[B, BC, C, BC].YZ', '[C, BC, B, AB].XY',
                                '[AB, A, AB, B].Y', '[AB, B, BC, C].Z', '[BC, C, BC, B].Y', '[BC, B, AB, A].X']

        relVars = RelationalSpace.getRelationalVariables(schema, 3, includeExistence=True)
        self.assertTrue(all([isinstance(relVar, RelationalVariable) for relVar in relVars]))
        TestUtil.assertUnorderedListEqual(self, hop0 + hop1 + hop2 + hop3, [str(relVar) for relVar in relVars])


    def testNoExistenceVariables(self):
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

        hop0 = ['[A].X', '[B].Y', '[C].Z', '[AB].XY', '[BC].YZ']

        hop1 = ['[A, AB].XY', '[B, AB].XY', '[B, BC].YZ', '[C, BC].YZ', '[AB, A].X',
                '[AB, B].Y', '[BC, C].Z', '[BC, B].Y']

        hop2 = ['[A, AB, B].Y', '[B, AB, A].X', '[B, BC, C].Z', '[C, BC, B].Y', '[AB, A, AB].XY',
                '[AB, B, BC].YZ', '[BC, C, BC].YZ', '[BC, B, AB].XY']

        hop3 = ['[A, AB, B, BC].YZ', '[B, AB, A, AB].XY', '[B, BC, C, BC].YZ', '[C, BC, B, AB].XY',
                '[AB, A, AB, B].Y', '[AB, B, BC, C].Z', '[BC, C, BC, B].Y', '[BC, B, AB, A].X']

        relVars = RelationalSpace.getRelationalVariables(schema, 3)
        self.assertTrue(all([isinstance(relVar, RelationalVariable) for relVar in relVars]))
        TestUtil.assertUnorderedListEqual(self, hop0 + hop1 + hop2 + hop3, [str(relVar) for relVar in relVars])


if __name__ == '__main__':
    unittest.main()
