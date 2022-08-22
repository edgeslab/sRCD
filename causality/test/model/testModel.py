import unittest
from mock import MagicMock
from causality.model import RelationalValidity
from causality.test import TestUtil
from causality.model.Schema import Schema
from causality.model.Model import Model
from causality.model import ParserUtil

class TestModel(unittest.TestCase):

    def setUp(self):
        self.schema = Schema()


    def testMakeModel(self):
        model = Model(self.schema, [])
        self.assertEqual([], model.dag.nodes())

        self.schema.addEntity('A')
        self.schema.addAttribute('A', 'X')
        self.schema.addEntity('B')
        self.schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        model = Model(self.schema, [])
        TestUtil.assertUnorderedListEqual(self, ['[A].exists', '[B].exists', '[AB].exists', '[A].X'],
            [str(attr) for attr in model.dag.nodes()])
        TestUtil.assertUnorderedListEqual(self, [('[A].exists', '[AB].exists'), ('[B].exists', '[AB].exists'),
            ('[A].exists', '[A].X')], [(str(edge[0]), str(edge[1])) for edge in model.dag.edges()])
        self.assertEqual([], model.dependencies)

        self.schema.addAttribute('B', 'Y')
        dependencies = ['[A, AB, B].Y -> [A].X']
        model = Model(self.schema, dependencies)
        TestUtil.assertUnorderedListEqual(self, ['[A].exists', '[B].exists', '[AB].exists', '[B].Y', '[A].X'],
            [str(attr) for attr in model.dag.nodes()])
        TestUtil.assertUnorderedListEqual(self, [('[A].exists', '[AB].exists'), ('[B].exists', '[AB].exists'),
            ('[A].exists', '[A].X'), ('[AB].exists', '[A].X'), ('[B].exists', '[A].X'), ('[B].exists', '[B].Y'),
            ('[B].Y', '[A].X')], [(str(edge[0]), str(edge[1])) for edge in model.dag.edges()])
        self.assertEqual(dependencies, [str(relDep) for relDep in model.dependencies])


    def testDuplicateDependency(self):
        self.schema.addEntity('A')
        dependencies = ['[A].X -> [A].Y', '[A].X -> [A].Y']
        TestUtil.assertRaisesMessage(self, Exception, "Found duplicate dependency", Model, self.schema, dependencies)


    def testDependencyValidity(self):
        # Model needs to pass in mock for dependencyChecker, make sure gets called exactly once per dependency
        mockRelDepChecker = MagicMock(wraps=RelationalValidity.checkRelationalDependencyValidity)
        schema = Schema()
        dependencies = []
        Model(schema, dependencies, relationalDependencyChecker=mockRelDepChecker)
        self.assertEqual(0, mockRelDepChecker.call_count)

        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addAttribute('A', 'V')
        schema.addEntity('B')
        schema.addAttribute('B', 'Y')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.MANY))
        schema.addEntity('C')
        schema.addAttribute('C', 'Z')
        schema.addAttribute('C', 'W')
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.MANY))

        dependencies = ['[B, AB, A].X -> [B].Y', '[C, BC, B].Y -> [C].Z', '[C].Z -> [C].W',
                     '[A].X -> [A].V', '[A, AB, B, BC, C].W -> [A].V']
        mockRelDepChecker = MagicMock(wraps=RelationalValidity.checkRelationalDependencyValidity)
        Model(schema, dependencies, relationalDependencyChecker=mockRelDepChecker)
        self.assertEqual(5, mockRelDepChecker.call_count)


    def testSchemaToModelConsistency(self):
        self.schema.addEntity('A')
        model = Model(self.schema, [])
        TestUtil.assertUnorderedListEqual(self, ['[A].exists'],
            [str(itemRelVar) for itemRelVar in model.getItemRelVars()])
        TestUtil.assertUnorderedListEqual(self, [],
            [str(attrRelVar) for attrRelVar in model.getAttrRelVars()])

        self.schema.addAttribute('A', 'X')
        model = Model(self.schema, [])
        TestUtil.assertUnorderedListEqual(self, ['[A].exists'],
            [str(itemRelVar) for itemRelVar in model.getItemRelVars()])
        TestUtil.assertUnorderedListEqual(self, ['[A].X'],
            [str(attrRelVar) for attrRelVar in model.getAttrRelVars()])

        self.schema.addAttribute('A', 'Y')
        model = Model(self.schema, [])
        TestUtil.assertUnorderedListEqual(self, ['[A].exists'],
            [str(itemRelVar) for itemRelVar in model.getItemRelVars()])
        TestUtil.assertUnorderedListEqual(self, ['[A].X', '[A].Y'],
            [str(attrRelVar) for attrRelVar in model.getAttrRelVars()])

        self.schema.addEntity('B')
        model = Model(self.schema, [])
        TestUtil.assertUnorderedListEqual(self, ['[A].exists', '[B].exists'],
            [str(itemRelVar) for itemRelVar in model.getItemRelVars()])
        TestUtil.assertUnorderedListEqual(self, ['[A].X', '[A].Y'],
            [str(attrRelVar) for attrRelVar in model.getAttrRelVars()])

        self.schema.addAttribute('B', 'Z')
        model = Model(self.schema, [])
        TestUtil.assertUnorderedListEqual(self, ['[A].exists', '[B].exists'],
            [str(itemRelVar) for itemRelVar in model.getItemRelVars()])
        TestUtil.assertUnorderedListEqual(self, ['[A].X', '[A].Y', '[B].Z'],
            [str(attrRelVar) for attrRelVar in model.getAttrRelVars()])

        self.schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        model = Model(self.schema, [])
        TestUtil.assertUnorderedListEqual(self, ['[A].exists', '[B].exists', '[AB].exists'],
            [str(itemRelVar) for itemRelVar in model.getItemRelVars()])
        TestUtil.assertUnorderedListEqual(self, ['[A].X', '[A].Y', '[B].Z'],
            [str(attrRelVar) for attrRelVar in model.getAttrRelVars()])

        self.schema.addAttribute('AB', 'XY')
        model = Model(self.schema, [])
        TestUtil.assertUnorderedListEqual(self, ['[A].exists', '[B].exists', '[AB].exists'],
            [str(itemRelVar) for itemRelVar in model.getItemRelVars()])
        TestUtil.assertUnorderedListEqual(self, ['[A].X', '[A].Y', '[B].Z', '[AB].XY'],
            [str(attrRelVar) for attrRelVar in model.getAttrRelVars()])

        self.schema.addAttribute('AB', 'XY2')
        model = Model(self.schema, [])
        TestUtil.assertUnorderedListEqual(self, ['[A].exists', '[B].exists', '[AB].exists'],
            [str(itemRelVar) for itemRelVar in model.getItemRelVars()])
        TestUtil.assertUnorderedListEqual(self, ['[A].X', '[A].Y', '[B].Z', '[AB].XY', '[AB].XY2'],
            [str(attrRelVar) for attrRelVar in model.getAttrRelVars()])

        self.schema.addEntity('C')
        model = Model(self.schema, [])
        TestUtil.assertUnorderedListEqual(self, ['[A].exists', '[B].exists', '[AB].exists', '[C].exists'],
            [str(itemRelVar) for itemRelVar in model.getItemRelVars()])
        TestUtil.assertUnorderedListEqual(self, ['[A].X', '[A].Y', '[B].Z', '[AB].XY', '[AB].XY2'],
            [str(attrRelVar) for attrRelVar in model.getAttrRelVars()])

        self.schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.ONE))
        model = Model(self.schema, [])
        TestUtil.assertUnorderedListEqual(self, ['[A].exists', '[B].exists', '[AB].exists', '[C].exists', '[BC].exists'],
            [str(itemRelVar) for itemRelVar in model.getItemRelVars()])
        TestUtil.assertUnorderedListEqual(self, ['[A].X', '[A].Y', '[B].Z', '[AB].XY', '[AB].XY2'],
            [str(attrRelVar) for attrRelVar in model.getAttrRelVars()])

        self.schema.addAttribute('BC', 'YZ')
        model = Model(self.schema, [])
        TestUtil.assertUnorderedListEqual(self, ['[A].exists', '[B].exists', '[AB].exists', '[C].exists', '[BC].exists'],
            [str(itemRelVar) for itemRelVar in model.getItemRelVars()])
        TestUtil.assertUnorderedListEqual(self, ['[A].X', '[A].Y', '[B].Z', '[AB].XY', '[AB].XY2', '[BC].YZ'],
            [str(attrRelVar) for attrRelVar in model.getAttrRelVars()])


    def testGetNextItemOrAttribute(self):
        dependencies = []
        model = Model(self.schema, dependencies)
        self.assertEqual([], list(model.getNextItemOrAttribute()))

        self.schema.addEntity('A')
        model = Model(self.schema, dependencies)
        self.assertEqual(['[A].exists'], [str(attrRelVar) for attrRelVar in model.getNextItemOrAttribute()])

        self.schema.addAttribute('A', 'X')
        model = Model(self.schema, dependencies)
        self.assertEqual(['[A].exists', '[A].X'], [str(attrRelVar) for attrRelVar in model.getNextItemOrAttribute()])

        # order of [A].X and [A].Y doesn't matter
        self.schema.addAttribute('A', 'Y')
        model = Model(self.schema, dependencies)
        attrOrder = [str(attrRelVar) for attrRelVar in model.getNextItemOrAttribute()]
        self.assertEqual(['[A].exists'], attrOrder[0:1])
        TestUtil.assertUnorderedListEqual(self, ['[A].X', '[A].Y'], attrOrder[1:])

        # order of [A].X and [A].Y _does_ matter
        dependencies = ['[A].X -> [A].Y']
        model = Model(self.schema, dependencies)
        self.assertEqual(['[A].exists', '[A].X', '[A].Y'], [str(attrRelVar) for attrRelVar in model.getNextItemOrAttribute()])

        dependencies = ['[A].Y -> [A].X']
        model = Model(self.schema, dependencies)
        self.assertEqual(['[A].exists', '[A].Y', '[A].X'], [str(attrRelVar) for attrRelVar in model.getNextItemOrAttribute()])

        # [A].X -> [A].Y -> [A].Z
        self.schema.addAttribute('A', 'Z')
        dependencies = ['[A].X -> [A].Y', '[A].Y -> [A].Z']
        model = Model(self.schema, dependencies)
        self.assertEqual(['[A].exists', '[A].X', '[A].Y', '[A].Z'], [str(attrRelVar) for attrRelVar in model.getNextItemOrAttribute()])

        # [A].X -> [A].Z <- [A].Y
        dependencies = ['[A].X -> [A].Z', '[A].Y -> [A].Z']
        model = Model(self.schema, dependencies)
        attrOrder = [str(attrRelVar) for attrRelVar in model.getNextItemOrAttribute()]
        self.assertEqual(['[A].exists'], attrOrder[0:1])
        TestUtil.assertUnorderedListEqual(self, ['[A].X', '[A].Y'], attrOrder[1:3])
        self.assertEqual(['[A].Z'], attrOrder[3:])

        self.schema.addAttribute('A', 'W')
        model = Model(self.schema, dependencies)
        attrOrder = [str(attrRelVar) for attrRelVar in model.getNextItemOrAttribute()]
        self.assertIn('[A].W', attrOrder)
        attrOrder.remove('[A].W')
        self.assertEqual(['[A].exists'], attrOrder[0:1])
        TestUtil.assertUnorderedListEqual(self, ['[A].X', '[A].Y'], attrOrder[1:3])
        self.assertEqual(['[A].Z'], attrOrder[3:])

        # [A, AB, B].Y -> [A].X
        self.schema = Schema()
        self.schema.addEntity('A')
        self.schema.addAttribute('A', 'X')
        self.schema.addEntity('B')
        self.schema.addAttribute('B', 'Y')
        self.schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        dependencies = ['[A, AB, B].Y -> [A].X']
        model = Model(self.schema, dependencies)
        attrOrder = [str(attrRelVar) for attrRelVar in model.getNextItemOrAttribute()]
        self.assertTrue(attrOrder.index('[B].exists') < attrOrder.index('[B].Y'))
        self.assertTrue(attrOrder.index('[A].exists') < attrOrder.index('[AB].exists'))
        self.assertTrue(attrOrder.index('[B].exists') < attrOrder.index('[AB].exists'))
        self.assertTrue(attrOrder.index('[B].Y') < attrOrder.index('[A].X'))
        self.assertTrue(attrOrder.index('[AB].exists') < attrOrder.index('[A].X'))
        self.assertEqual(['[A].X'], attrOrder[4:])

        # [B, AB, A].X -> [B].Y, [BC, B].Y -> [BC].exists <- [BC, C].Z
        self.schema = Schema()
        self.schema.addEntity('A')
        self.schema.addAttribute('A', 'X')
        self.schema.addEntity('B')
        self.schema.addAttribute('B', 'Y')
        self.schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        self.schema.addEntity('C')
        self.schema.addAttribute('C', 'Z')
        self.schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.ONE))
        dependencies = ['[B, AB, A].X -> [B].Y', '[BC, B].Y -> [BC].exists', '[BC, C].Z -> [BC].exists']
        model = Model(self.schema, dependencies)
        attrOrder = [str(attrRelVar) for attrRelVar in model.getNextItemOrAttribute()]
        for attrRelVarStr in ['[A].exists', '[B].exists', '[C].exists', '[AB].exists', '[BC].exists',
                              '[A].X', '[B].Y', '[C].Z']:
            self.assertIn(attrRelVarStr, attrOrder)

        self.assertEqual('[BC].exists', attrOrder[-1])
        # precedence rule of items before their attrs
        self.assertTrue(attrOrder.index('[A].exists') < attrOrder.index('[A].X'))
        self.assertTrue(attrOrder.index('[B].exists') < attrOrder.index('[B].Y'))
        self.assertTrue(attrOrder.index('[C].exists') < attrOrder.index('[C].Z'))

        # precedence rule of entities before their relationships
        self.assertTrue(attrOrder.index('[A].exists') < attrOrder.index('[AB].exists'))
        self.assertTrue(attrOrder.index('[B].exists') < attrOrder.index('[AB].exists'))
        self.assertTrue(attrOrder.index('[B].exists') < attrOrder.index('[BC].exists'))
        self.assertTrue(attrOrder.index('[C].exists') < attrOrder.index('[BC].exists'))

        # precedence based on input dependencies
        self.assertTrue(attrOrder.index('[A].X') < attrOrder.index('[B].Y'))
        self.assertTrue(attrOrder.index('[B].Y') < attrOrder.index('[BC].exists'))
        self.assertTrue(attrOrder.index('[C].Z') < attrOrder.index('[BC].exists'))

        # precedence based on items involved in parent relational paths before their children
        self.assertTrue(attrOrder.index('[AB].exists') < attrOrder.index('[B].Y'))
        self.assertTrue(attrOrder.index('[A].exists') < attrOrder.index('[B].Y'))


    def testCycleDetection(self):
        # cycle
        self.schema.addEntity('A')
        self.schema.addAttribute('A', 'X')
        dependencies = ['[A].X -> [A].X']
        TestUtil.assertRaisesMessage(self, Exception, "dependencies encodes a cycle among relational variables",
                                     Model, self.schema, dependencies)

        self.schema.addAttribute('A', 'Y')
        dependencies = ['[A].X -> [A].Y', '[A].Y -> [A].X']
        TestUtil.assertRaisesMessage(self, Exception, "dependencies encodes a cycle among relational variables",
            Model, self.schema, dependencies)

        # parent involves the rel var (also a cycle)
        self.schema.addEntity('B')
        self.schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.MANY))
        dependencies = ['[A, AB, B, AB, A].X -> [A].X']
        TestUtil.assertRaisesMessage(self, Exception, "dependencies encodes a cycle among relational variables",
            Model, self.schema, dependencies)

        # cycle involving relationship existence and its ancestry in relational paths
        dependencies = ['[AB, B, AB, A].X -> [AB].exists']
        TestUtil.assertRaisesMessage(self, Exception, "dependencies encodes a cycle among relational variables",
            Model, self.schema, dependencies)

        self.schema.addAttribute('B', 'Z')
        dependencies = ['[AB, A].X -> [AB].exists', '[A, AB, B].Z -> [A].X']
        TestUtil.assertRaisesMessage(self, Exception, "dependencies encodes a cycle among relational variables",
            Model, self.schema, dependencies)


    def dictStrsToRelVars(self, attrToParents):
        return {ParserUtil.parseRelVar(attr): [ParserUtil.parseRelVar(parent) for parent in parents] for attr, parents in attrToParents.items()}


if __name__ == '__main__':
    unittest.main()
