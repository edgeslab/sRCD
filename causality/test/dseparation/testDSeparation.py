import unittest
from mock import MagicMock
from causality.model import ParserUtil
from causality.model import RelationalValidity
from causality.test import TestUtil
from causality.model.Model import Model
from causality.model.Schema import Schema
from causality.dseparation.DSeparation import DSeparation

class TestDSeparation(unittest.TestCase):

    def testTwoVariableModels(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X1')
        schema.addAttribute('A', 'X2')
        model = Model(schema, [])
        dsep = DSeparation(model)
        self.assertTrue(dsep.dSeparated(0, ['[A].X1'], ['[A].X2'], []))
        self.assertTrue(dsep.dSeparated(0, ['[A].X2'], ['[A].X1'], []))

        model = Model(schema, ['[A].X1 -> [A].X2'])
        dsep = DSeparation(model)
        self.assertFalse(dsep.dSeparated(0, ['[A].X1'], ['[A].X2'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X2'], ['[A].X1'], []))


    def testThreeVariableModels(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X1')
        schema.addAttribute('A', 'X2')
        schema.addAttribute('A', 'X3')

        model = Model(schema, [])
        dsep = DSeparation(model)
        self.assertTrue(dsep.dSeparated(0, ['[A].X1'], ['[A].X2'], []))
        self.assertTrue(dsep.dSeparated(0, ['[A].X2'], ['[A].X1'], []))
        self.assertTrue(dsep.dSeparated(0, ['[A].X1'], ['[A].X3'], []))
        self.assertTrue(dsep.dSeparated(0, ['[A].X3'], ['[A].X1'], []))
        self.assertTrue(dsep.dSeparated(0, ['[A].X2'], ['[A].X3'], []))
        self.assertTrue(dsep.dSeparated(0, ['[A].X3'], ['[A].X2'], []))
        self.assertTrue(dsep.dSeparated(0, ['[A].X1'], ['[A].X2'], ['[A].X3']))
        self.assertTrue(dsep.dSeparated(0, ['[A].X2'], ['[A].X3'], ['[A].X1']))
        self.assertTrue(dsep.dSeparated(0, ['[A].X3'], ['[A].X1'], ['[A].X2']))

        # one dependency model: only one in isomorphic class
        model = Model(schema, ['[A].X2 -> [A].X3'])
        dsep = DSeparation(model)
        self.assertTrue(dsep.dSeparated(0, ['[A].X1'], ['[A].X2'], []))
        self.assertTrue(dsep.dSeparated(0, ['[A].X2'], ['[A].X1'], []))
        self.assertTrue(dsep.dSeparated(0, ['[A].X1'], ['[A].X3'], []))
        self.assertTrue(dsep.dSeparated(0, ['[A].X3'], ['[A].X1'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X2'], ['[A].X3'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X3'], ['[A].X2'], []))
        self.assertTrue(dsep.dSeparated(0, ['[A].X1'], ['[A].X2'], ['[A].X3']))
        self.assertFalse(dsep.dSeparated(0, ['[A].X2'], ['[A].X3'], ['[A].X1']))
        self.assertTrue(dsep.dSeparated(0, ['[A].X3'], ['[A].X1'], ['[A].X2']))

        # two dependency models: four in isomorphic class
        model = Model(schema, ['[A].X1 -> [A].X2', '[A].X1 -> [A].X3'])
        dsep = DSeparation(model)
        self.assertFalse(dsep.dSeparated(0, ['[A].X1'], ['[A].X2'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X2'], ['[A].X1'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X1'], ['[A].X3'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X3'], ['[A].X1'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X2'], ['[A].X3'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X3'], ['[A].X2'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X1'], ['[A].X2'], ['[A].X3']))
        self.assertTrue(dsep.dSeparated(0, ['[A].X2'], ['[A].X3'], ['[A].X1']))
        self.assertFalse(dsep.dSeparated(0, ['[A].X3'], ['[A].X1'], ['[A].X2']))

        model = Model(schema, ['[A].X1 -> [A].X2', '[A].X3 -> [A].X1'])
        dsep = DSeparation(model)
        self.assertFalse(dsep.dSeparated(0, ['[A].X1'], ['[A].X2'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X2'], ['[A].X1'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X1'], ['[A].X3'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X3'], ['[A].X1'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X2'], ['[A].X3'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X3'], ['[A].X2'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X1'], ['[A].X2'], ['[A].X3']))
        self.assertTrue(dsep.dSeparated(0, ['[A].X2'], ['[A].X3'], ['[A].X1']))
        self.assertFalse(dsep.dSeparated(0, ['[A].X3'], ['[A].X1'], ['[A].X2']))

        model = Model(schema, ['[A].X2 -> [A].X1', '[A].X1 -> [A].X3'])
        dsep = DSeparation(model)
        self.assertFalse(dsep.dSeparated(0, ['[A].X1'], ['[A].X2'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X2'], ['[A].X1'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X1'], ['[A].X3'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X3'], ['[A].X1'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X2'], ['[A].X3'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X3'], ['[A].X2'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X1'], ['[A].X2'], ['[A].X3']))
        self.assertTrue(dsep.dSeparated(0, ['[A].X2'], ['[A].X3'], ['[A].X1']))
        self.assertFalse(dsep.dSeparated(0, ['[A].X3'], ['[A].X1'], ['[A].X2']))

        model = Model(schema, ['[A].X2 -> [A].X1', '[A].X3 -> [A].X1'])
        dsep = DSeparation(model)
        self.assertFalse(dsep.dSeparated(0, ['[A].X1'], ['[A].X2'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X2'], ['[A].X1'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X1'], ['[A].X3'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X3'], ['[A].X1'], []))
        self.assertTrue(dsep.dSeparated(0, ['[A].X2'], ['[A].X3'], []))
        self.assertTrue(dsep.dSeparated(0, ['[A].X3'], ['[A].X2'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X1'], ['[A].X2'], ['[A].X3']))
        self.assertFalse(dsep.dSeparated(0, ['[A].X2'], ['[A].X3'], ['[A].X1']))
        self.assertFalse(dsep.dSeparated(0, ['[A].X3'], ['[A].X1'], ['[A].X2']))

        # three dependency models: two in isomorphic class
        model = Model(schema, ['[A].X1 -> [A].X2', '[A].X1 -> [A].X3', '[A].X2 -> [A].X3'])
        dsep = DSeparation(model)
        self.assertFalse(dsep.dSeparated(0, ['[A].X1'], ['[A].X2'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X2'], ['[A].X1'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X1'], ['[A].X3'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X3'], ['[A].X1'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X2'], ['[A].X3'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X3'], ['[A].X2'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X1'], ['[A].X2'], ['[A].X3']))
        self.assertFalse(dsep.dSeparated(0, ['[A].X2'], ['[A].X3'], ['[A].X1']))
        self.assertFalse(dsep.dSeparated(0, ['[A].X3'], ['[A].X1'], ['[A].X2']))

        model = Model(schema, ['[A].X1 -> [A].X2', '[A].X1 -> [A].X3', '[A].X3 -> [A].X2'])
        dsep = DSeparation(model)
        self.assertFalse(dsep.dSeparated(0, ['[A].X1'], ['[A].X2'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X2'], ['[A].X1'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X1'], ['[A].X3'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X3'], ['[A].X1'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X2'], ['[A].X3'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X3'], ['[A].X2'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X1'], ['[A].X2'], ['[A].X3']))
        self.assertFalse(dsep.dSeparated(0, ['[A].X2'], ['[A].X3'], ['[A].X1']))
        self.assertFalse(dsep.dSeparated(0, ['[A].X3'], ['[A].X1'], ['[A].X2']))


    def testFourVariableModels(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X1')
        schema.addAttribute('A', 'X2')
        schema.addAttribute('A', 'X3')
        schema.addAttribute('A', 'X4')
        model = Model(schema, ['[A].X1 -> [A].X3', '[A].X2 -> [A].X3', '[A].X3 -> [A].X4'])
        dsep = DSeparation(model)
        self.assertTrue(dsep.dSeparated(0, ['[A].X1'], ['[A].X2'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X1'], ['[A].X2'], ['[A].X3']))
        self.assertFalse(dsep.dSeparated(0, ['[A].X1'], ['[A].X2'], ['[A].X4'])) # tests conditioning on descendant of a collider
        self.assertFalse(dsep.dSeparated(0, ['[A].X1'], ['[A].X2'], ['[A].X3', '[A].X4']))

        # model has multiple paths
        model = Model(schema, ['[A].X1 -> [A].X3', '[A].X1 -> [A].X2', '[A].X2 -> [A].X4', '[A].X4 -> [A].X3'])
        dsep = DSeparation(model)
        self.assertFalse(dsep.dSeparated(0, ['[A].X1'], ['[A].X4'], []))
        self.assertTrue(dsep.dSeparated(0, ['[A].X1'], ['[A].X4'], ['[A].X2']))
        self.assertFalse(dsep.dSeparated(0, ['[A].X1'], ['[A].X4'], ['[A].X2', '[A].X3']))
        self.assertFalse(dsep.dSeparated(0, ['[A].X2'], ['[A].X3'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X2'], ['[A].X3'], ['[A].X1']))
        self.assertFalse(dsep.dSeparated(0, ['[A].X2'], ['[A].X3'], ['[A].X4']))
        self.assertTrue(dsep.dSeparated(0, ['[A].X2'], ['[A].X3'], ['[A].X1', '[A].X4']))


    def testEightVariableModel(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'A')
        schema.addAttribute('A', 'B')
        schema.addAttribute('A', 'C')
        schema.addAttribute('A', 'D')
        schema.addAttribute('A', 'E')
        schema.addAttribute('A', 'F')
        schema.addAttribute('A', 'G')
        schema.addAttribute('A', 'H')
        dependencies = ['[A].A -> [A].C', '[A].A -> [A].D', '[A].B -> [A].D', '[A].B -> [A].F', '[A].C -> [A].E',
                        '[A].D -> [A].E', '[A].E -> [A].G', '[A].E -> [A].H', '[A].F -> [A].H']
        model = Model(schema, dependencies)
        dsep = DSeparation(model)
        self.assertFalse(dsep.dSeparated(0, ['[A].A'], ['[A].E'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].A'], ['[A].E'], ['[A].C']))
        self.assertTrue(dsep.dSeparated(0, ['[A].A'], ['[A].E'], ['[A].C', '[A].D']))
        self.assertFalse(dsep.dSeparated(0, ['[A].B'], ['[A].E'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].B'], ['[A].E'], ['[A].D']))
        self.assertTrue(dsep.dSeparated(0, ['[A].B'], ['[A].E'], ['[A].C', '[A].D']))
        self.assertFalse(dsep.dSeparated(0, ['[A].B'], ['[A].E'], ['[A].C', '[A].D', '[A].H']))

        # testing d-separation with sets of relVars
        self.assertFalse(dsep.dSeparated(0, ['[A].A', '[A].C'], {'[A].F', '[A].H'}, []))
        self.assertFalse(dsep.dSeparated(0, ('[A].A', '[A].C'), {'[A].F', '[A].H'}, ['[A].E']))
        self.assertTrue(dsep.dSeparated(0, ('[A].A', '[A].C'), {'[A].F', '[A].H'}, {'[A].B', '[A].E'}))


    def testBadRelVarInput(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X1')
        schema.addAttribute('A', 'X2')
        model = Model(schema, [])
        dsep = DSeparation(model)
        TestUtil.assertRaisesMessage(self, Exception, "relVars1 must be a non-empty sequence of parseable RelationalVariable strings",
            dsep.dSeparated, 0, None, ['[A].X2'], [])
        TestUtil.assertRaisesMessage(self, Exception, "relVars2 must be a non-empty sequence of parseable RelationalVariable strings",
            dsep.dSeparated, 0, ['[A].X1'], None, [])
        TestUtil.assertRaisesMessage(self, Exception, "condRelVars must be a sequence of parseable RelationalVariable strings",
            dsep.dSeparated, 0, ['[A].X1'], ['[A].X2'], None)

        TestUtil.assertRaisesMessage(self, Exception, "relVars1 must be a non-empty sequence of parseable RelationalVariable strings",
             dsep.dSeparated, 0, [], ['[A].X2'], [])

        TestUtil.assertRaisesMessage(self, Exception, "relVars2 must be a non-empty sequence of parseable RelationalVariable strings",
             DSeparation.dSeparated, model, 0, ['[A].X1'], [], [])


    def testIntersectingInputRelVars(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X1')
        schema.addAttribute('A', 'X2')
        model = Model(schema, [])
        dsep = DSeparation(model)
        self.assertFalse(dsep.dSeparated(0, ['[A].X1'], ['[A].X1'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X2', '[A].X1'], ['[A].X1', '[A].X2'], []))
        self.assertFalse(dsep.dSeparated(0, ['[A].X2', '[A].X1'], ['[A].X1'], []))

        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addEntity('B')
        schema.addAttribute('B', 'Y')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.MANY))
        schema.addEntity('C')
        schema.addAttribute('C', 'Z')
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.MANY))
        model = Model(schema, [])
        dsep = DSeparation(model)
        self.assertFalse(dsep.dSeparated(6, ['[A, AB, B, AB, A, AB, B].Y'], ['[A, AB, B, BC, C, BC, B].Y'], []))


    def testCondRelVarsSameAsOneRelVar(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X1')
        schema.addAttribute('A', 'X2')
        schema.addAttribute('A', 'X3')
        model = Model(schema, ['[A].X1 -> [A].X2', '[A].X1 -> [A].X3'])
        dsep = DSeparation(model)
        self.assertTrue(dsep.dSeparated(0, ['[A].X1'], ['[A].X2'], ['[A].X2']))
        self.assertTrue(dsep.dSeparated(0, ['[A].X1'], ['[A].X2'], ['[A].X1']))
        self.assertTrue(dsep.dSeparated(0, ['[A].X1'], ['[A].X2'], ['[A].X1', '[A].X2']))
        self.assertTrue(dsep.dSeparated(0, ['[A].X1'], ['[A].X2', '[A].X3'], ['[A].X1', '[A].X2']))
        self.assertTrue(dsep.dSeparated(0, ['[A].X1'], ['[A].X2', '[A].X3'], ['[A].X2', '[A].X3']))
        self.assertTrue(dsep.dSeparated(0, ['[A].X2', '[A].X3'], ['[A].X1'], ['[A].X2', '[A].X3']))

        # test for sequences of length > 1
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'A')
        schema.addAttribute('A', 'B')
        schema.addAttribute('A', 'C')
        schema.addAttribute('A', 'D')
        schema.addAttribute('A', 'E')
        schema.addAttribute('A', 'F')
        schema.addAttribute('A', 'G')
        schema.addAttribute('A', 'H')
        dependencies = ['[A].A -> [A].B', '[A].A -> [A].C', '[A].B -> [A].D', '[A].C -> [A].D',
                        '[A].E -> [A].F', '[A].E -> [A].G', '[A].F -> [A].H', '[A].G -> [A].H']
        model = Model(schema, dependencies)
        dsep = DSeparation(model)
        self.assertTrue(dsep.dSeparated(0, ['[A].A', '[A].E'], ['[A].D', '[A].H'],
            ['[A].B', '[A].C', '[A].E', '[A].H']))
        self.assertFalse(dsep.dSeparated(0, ['[A].A', '[A].E', '[A].F'], ['[A].D', '[A].H', '[A].G'],
            ['[A].B', '[A].C', '[A].E', '[A].H']))


    def testRemoveCommonCondRelVarOnlyFromRelVars(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X1')
        schema.addAttribute('A', 'X2')
        schema.addAttribute('A', 'X3')
        model = Model(schema, ['[A].X1 -> [A].X2', '[A].X2 -> [A].X3'])
        dsep = DSeparation(model)
        self.assertTrue(dsep.dSeparated(0, ['[A].X1', '[A].X2'], ['[A].X3'], ['[A].X2']))
        self.assertTrue(dsep.dSeparated(0, ['[A].X1', '[A].X2'], ['[A].X2', '[A].X3'], ['[A].X2']))


    def testOneToOneRDS(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addEntity('B')
        schema.addAttribute('B', 'Y')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        model = Model(schema, [])
        dsep = DSeparation(model)
        self.assertTrue(dsep.dSeparated(2, ['[A, AB, B].Y'], ['[A].X'], []))
        self.assertTrue(dsep.dSeparated(2, ['[A].X'], ['[A, AB, B].Y'], []))
        self.assertTrue(dsep.dSeparated(2, ['[B].Y'], ['[B, AB, A].X'], []))
        self.assertTrue(dsep.dSeparated(2, ['[B, AB, A].X'], ['[B].Y'], []))
        self.assertTrue(dsep.dSeparated(1, ['[AB, B].Y'], ['[AB, A].X'], []))
        self.assertTrue(dsep.dSeparated(1, ['[AB, A].X'], ['[AB, B].Y'], []))

        model = Model(schema, ['[B, AB, A].X -> [B].Y'])
        dsep = DSeparation(model)
        self.assertFalse(dsep.dSeparated(2, ['[B].Y'], ['[B, AB, A].X'], []))
        self.assertFalse(dsep.dSeparated(2, ['[B, AB, A].X'], ['[B].Y'], []))
        self.assertFalse(dsep.dSeparated(2, ['[A, AB, B].Y'], ['[A].X'], []))
        self.assertFalse(dsep.dSeparated(2, ['[A].X'], ['[A, AB, B].Y'], []))
        self.assertFalse(dsep.dSeparated(1, ['[AB, B].Y'], ['[AB, A].X'], []))
        self.assertFalse(dsep.dSeparated(1, ['[AB, A].X'], ['[AB, B].Y'], []))

        schema.addAttribute('A', 'Z')
        model = Model(schema, ['[B, AB, A].X -> [B].Y', '[A, AB, B].Y -> [A].Z' ])
        dsep = DSeparation(model)
        self.assertFalse(dsep.dSeparated(2, ['[B, AB, A].X'], ['[B, AB, A].Z'], []))
        self.assertFalse(dsep.dSeparated(2, ['[A].Z'], ['[A].X'], []))
        self.assertFalse(dsep.dSeparated(1, ['[AB, A].X'], ['[AB, A].Z'], []))

        self.assertTrue(dsep.dSeparated(2, ['[A].X'], ['[A].Z'], ['[A, AB, B].Y']))
        self.assertTrue(dsep.dSeparated(1, ['[AB, A].X'], ['[AB, A].Z'], ['[AB, B].Y']))
        self.assertTrue(dsep.dSeparated(2, ['[B, AB, A].X'], ['[B, AB, A].Z'], ['[B].Y']))


    def testOneToManyRDS(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addEntity('B')
        schema.addAttribute('B', 'Y')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.ONE))
        model = Model(schema, ['[B, AB, A].X -> [B].Y'])
        dsep = DSeparation(model)
        self.assertFalse(dsep.dSeparated(4, ['[B].Y'], ['[B, AB, A].X'], []))
        self.assertFalse(dsep.dSeparated(4, ['[B, AB, A].X'], ['[B].Y'], []))
        self.assertFalse(dsep.dSeparated(2, ['[A, AB, B].Y'], ['[A].X'], []))
        self.assertFalse(dsep.dSeparated(2, ['[A].X'], ['[A, AB, B].Y'], []))
        self.assertFalse(dsep.dSeparated(3, ['[AB, B].Y'], ['[AB, A].X'], []))
        self.assertFalse(dsep.dSeparated(3, ['[AB, A].X'], ['[AB, B].Y'], []))
        self.assertFalse(dsep.dSeparated(3, ['[AB, A].X'], ['[AB, A, AB, B].Y'], []))

        schema.addAttribute('A', 'Z')
        model = Model(schema, ['[B, AB, A].X -> [B].Y', '[A, AB, B].Y -> [A].Z' ])
        dsep = DSeparation(model)
        self.assertFalse(dsep.dSeparated(4, ['[B, AB, A].X'], ['[B, AB, A].Z'], []))
        self.assertFalse(dsep.dSeparated(2, ['[A].Z'], ['[A].X'], []))
        self.assertFalse(dsep.dSeparated(3, ['[AB, A].X'], ['[AB, A].Z'], []))

        self.assertTrue(dsep.dSeparated(2, ['[A].X'], ['[A].Z'], ['[A, AB, B].Y']))

        self.assertFalse(dsep.dSeparated(4, ['[B, AB, A].X'], ['[B, AB, A].Z'], ['[B].Y']))
        self.assertFalse(dsep.dSeparated(4, ['[B, AB, A].X'], ['[B, AB, A].Z'], ['[B, AB, A, AB, B].Y']))
        self.assertTrue(dsep.dSeparated(4, ['[B, AB, A].X'], ['[B, AB, A].Z'],
                                                                                    ['[B].Y', '[B, AB, A, AB, B].Y']))

        self.assertFalse(dsep.dSeparated(3, ['[AB, A].X'], ['[AB, A].Z'], ['[AB, B].Y']))
        self.assertFalse(dsep.dSeparated(3, ['[AB, A].X'], ['[AB, A].Z'], ['[AB, A, AB, B].Y']))
        # true with both, forces intersection between '[AB, B].Y', '[AB, A, AB, B].Y' to be included automatically
        self.assertTrue(dsep.dSeparated(3, ['[AB, A].X'], ['[AB, A].Z'], ['[AB, B].Y', '[AB, A, AB, B].Y']))


    def testThreeEntityRDSJMLRExample(self):
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
        dsep = DSeparation(model)
        self.assertFalse(dsep.dSeparated(4, ['[A].X'], ['[A, AB, B, BC, C].Z'], []))
        self.assertTrue(dsep.dSeparated(4, ['[A].X'], ['[A, AB, B, BC, C].Z'], ['[A, AB, B].Y']))

        self.assertFalse(dsep.dSeparated(6, ['[A].X'], ['[A, AB, B, BC, C].Z'], []))
        self.assertFalse(dsep.dSeparated(6, ['[A].X'], ['[A, AB, B, BC, C].Z'], ['[A, AB, B].Y']))
        self.assertTrue(dsep.dSeparated(6, ['[A].X'], ['[A, AB, B, BC, C].Z'], ['[A, AB, B].Y', '[A, AB, B, AB, A].X']))
        self.assertTrue(dsep.dSeparated(6, ['[A].X'], ['[A, AB, B, BC, C].Z'], ['[A, AB, B].Y', '[A, AB, B, AB, A, AB, B].Y']))
        self.assertTrue(dsep.dSeparated(6, ['[A].X'], ['[A, AB, B, BC, C].Z'], ['[A, AB, B].Y', '[A, AB, B, BC, C, BC, B].Y']))

        self.assertFalse(dsep.dSeparated(8, ['[A].X'], ['[A, AB, B, BC, C].Z'], []))
        self.assertFalse(dsep.dSeparated(8, ['[A].X'], ['[A, AB, B, BC, C].Z'], ['[A, AB, B].Y']))
        self.assertTrue(dsep.dSeparated(8, ['[A].X'], ['[A, AB, B, BC, C].Z'], ['[A, AB, B].Y', '[A, AB, B, AB, A].X']))
        self.assertFalse(dsep.dSeparated(8, ['[A].X'], ['[A, AB, B, BC, C].Z'], ['[A, AB, B].Y', '[A, AB, B, AB, A, AB, B].Y']))
        self.assertTrue(dsep.dSeparated(8, ['[A].X'], ['[A, AB, B, BC, C].Z'],
                                               ['[A, AB, B].Y', '[A, AB, B, AB, A, AB, B].Y', '[A, AB, B, BC, C, BC, B].Y']))
        # forces intersection variables to be added to relVars2 in d-separation
        self.assertFalse(dsep.dSeparated(8, ['[A].X'], ['[A, AB, B, BC, C].Z'], ['[A, AB, B].Y', '[A, AB, B, BC, C, BC, B].Y']))
        self.assertFalse(dsep.dSeparated(8, ['[A].X'], ['[A, AB, B, BC, C].Z'],
                                                ['[A, AB, B].Y', '[A, AB, B, BC, C, BC, B].Y', '[A, AB, B, BC, C, BC, B, AB, A].X']))

        # forces intersection variables to be added to relVars1 in d-separation
        self.assertFalse(dsep.dSeparated(8, ['[A, AB, B, BC, C].Z'], ['[A].X'], ['[A, AB, B].Y', '[A, AB, B, BC, C, BC, B].Y']))


    def testRelVarCheckerUsed(self):
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
        dsep = DSeparation(model)
        mockRelVarSetChecker = MagicMock(wraps=RelationalValidity.checkValidityOfRelationalVariableSet)
        dsep.dSeparated(8, ['[A].X'], ['[A, AB, B, BC, C].Z'], [],
                               relationalVariableSetChecker=mockRelVarSetChecker)
        self.assertEqual(1, mockRelVarSetChecker.call_count)
        relVarsPassed = {ParserUtil.parseRelVar('[A].X'), ParserUtil.parseRelVar('[A, AB, B, BC, C].Z')}
        mockRelVarSetChecker.assert_called_with(model.schema, 8, relVarsPassed)

        mockRelVarSetChecker = MagicMock(wraps=RelationalValidity.checkValidityOfRelationalVariableSet)
        dsep.dSeparated(8, ['[A].X'], ['[A, AB, B, BC, C].Z'], ['[A, AB, B].Y', '[A, AB, B, BC, C, BC, B].Y'],
                               relationalVariableSetChecker=mockRelVarSetChecker)
        self.assertEqual(1, mockRelVarSetChecker.call_count)
        relVarsPassed = {ParserUtil.parseRelVar('[A].X'), ParserUtil.parseRelVar('[A, AB, B, BC, C].Z'),
                         ParserUtil.parseRelVar('[A, AB, B].Y'), ParserUtil.parseRelVar('[A, AB, B, BC, C, BC, B].Y')}
        mockRelVarSetChecker.assert_called_with(model.schema, 8, relVarsPassed)


    if __name__ == '__main__':
        unittest.main()
