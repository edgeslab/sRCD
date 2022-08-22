import unittest
from causality.model.Model import Model
from causality.model.Schema import Schema
from causality.citest.CITest import CITest
from causality.citest.CITest import Oracle


class TestOracle(unittest.TestCase):

    def testOracle(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X1')
        schema.addAttribute('A', 'X2')
        schema.addAttribute('A', 'X3')
        schema.addAttribute('A', 'X4')
        model = Model(schema, ['[A].X1 -> [A].X3', '[A].X2 -> [A].X3', '[A].X3 -> [A].X4'])
        oracle = Oracle(model)
        self.assertTrue(isinstance(oracle, CITest))
        self.assertTrue(oracle.isConditionallyIndependent('[A].X1', '[A].X2', []))
        self.assertFalse(oracle.isConditionallyIndependent('[A].X1', '[A].X2', ['[A].X3']))
        self.assertFalse(oracle.isConditionallyIndependent('[A].X1', '[A].X2', ['[A].X4'])) # tests conditioning on descendant of a collider
        self.assertFalse(oracle.isConditionallyIndependent('[A].X1', '[A].X2', ['[A].X3', '[A].X4']))

        # model has multiple paths
        model = Model(schema, ['[A].X1 -> [A].X3', '[A].X1 -> [A].X2', '[A].X2 -> [A].X4', '[A].X4 -> [A].X3'])
        oracle = Oracle(model)
        self.assertFalse(oracle.isConditionallyIndependent('[A].X1', '[A].X4', []))
        self.assertTrue(oracle.isConditionallyIndependent('[A].X1', '[A].X4', ['[A].X2']))
        self.assertFalse(oracle.isConditionallyIndependent('[A].X1', '[A].X4', ['[A].X2', '[A].X3']))
        self.assertFalse(oracle.isConditionallyIndependent('[A].X2', '[A].X3', []))
        self.assertFalse(oracle.isConditionallyIndependent('[A].X2', '[A].X3', ['[A].X1']))
        self.assertFalse(oracle.isConditionallyIndependent('[A].X2', '[A].X3', ['[A].X4']))
        self.assertTrue(oracle.isConditionallyIndependent('[A].X2', '[A].X3', ['[A].X1', '[A].X4']))


    def testOracleThreeEntityRDSJMLRExample(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addEntity('B')
        schema.addAttribute('B', 'Y')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.MANY))
        schema.addEntity('C')
        schema.addAttribute('C', 'Z')
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.MANY))
        model = Model(schema, ['[B, AB, A].X -> [B].Y', '[C, BC, B].Y -> [C].Z'])
        oracle = Oracle(model, 4)
        self.assertFalse(oracle.isConditionallyIndependent('[A].X', '[A, AB, B, BC, C].Z', []))
        self.assertTrue(oracle.isConditionallyIndependent('[A].X', '[A, AB, B, BC, C].Z', ['[A, AB, B].Y']))

        oracle = Oracle(model, 6)
        self.assertFalse(oracle.isConditionallyIndependent('[A].X', '[A, AB, B, BC, C].Z', []))
        self.assertFalse(oracle.isConditionallyIndependent('[A].X', '[A, AB, B, BC, C].Z', ['[A, AB, B].Y']))
        self.assertTrue(oracle.isConditionallyIndependent('[A].X', '[A, AB, B, BC, C].Z', ['[A, AB, B].Y', '[A, AB, B, AB, A].X']))
        self.assertTrue(oracle.isConditionallyIndependent('[A].X', '[A, AB, B, BC, C].Z', ['[A, AB, B].Y', '[A, AB, B, AB, A, AB, B].Y']))
        self.assertTrue(oracle.isConditionallyIndependent('[A].X', '[A, AB, B, BC, C].Z', ['[A, AB, B].Y', '[A, AB, B, BC, C, BC, B].Y']))

        oracle = Oracle(model, 8)
        self.assertFalse(oracle.isConditionallyIndependent('[A].X', '[A, AB, B, BC, C].Z', []))
        self.assertFalse(oracle.isConditionallyIndependent('[A].X', '[A, AB, B, BC, C].Z', ['[A, AB, B].Y']))
        self.assertTrue(oracle.isConditionallyIndependent('[A].X', '[A, AB, B, BC, C].Z', ['[A, AB, B].Y', '[A, AB, B, AB, A].X']))
        self.assertFalse(oracle.isConditionallyIndependent('[A].X', '[A, AB, B, BC, C].Z', ['[A, AB, B].Y', '[A, AB, B, AB, A, AB, B].Y']))
        self.assertTrue(oracle.isConditionallyIndependent('[A].X', '[A, AB, B, BC, C].Z',
                                          ['[A, AB, B].Y', '[A, AB, B, AB, A, AB, B].Y', '[A, AB, B, BC, C, BC, B].Y']))
        # forces intersection variables to be added to relVars2 in d-separation
        self.assertFalse(oracle.isConditionallyIndependent('[A].X', '[A, AB, B, BC, C].Z', ['[A, AB, B].Y', '[A, AB, B, BC, C, BC, B].Y']))
        self.assertFalse(oracle.isConditionallyIndependent('[A].X', '[A, AB, B, BC, C].Z',
                                           ['[A, AB, B].Y', '[A, AB, B, BC, C, BC, B].Y', '[A, AB, B, BC, C, BC, B, AB, A].X']))

        # forces intersection variables to be added to relVars1 in d-separation
        self.assertFalse(oracle.isConditionallyIndependent('[A, AB, B, BC, C].Z', '[A].X', ['[A, AB, B].Y', '[A, AB, B, BC, C, BC, B].Y']))


if __name__ == '__main__':
    unittest.main()
