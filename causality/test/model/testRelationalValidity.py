import unittest
from mock import MagicMock
from causality.model.Schema import Schema
from causality.test import TestUtil
from causality.model import RelationalValidity
from causality.model import ParserUtil

class TestModelUtil(unittest.TestCase):
    
    def testInvalidRelationalPaths(self):
        schema = Schema()
        TestUtil.assertRaisesMessage(self, Exception, "Schema has no item 'A' in relationalPath ['A']",
                                     RelationalValidity.checkRelationalPathValidity, schema, ['A'])

        schema.addEntity('A')
        TestUtil.assertRaisesMessage(self, Exception, "Schema has no item 'B' in relationalPath ['A', 'B']",
                                     RelationalValidity.checkRelationalPathValidity, schema, ['A', 'B'])

        schema.addEntity('B')
        TestUtil.assertRaisesMessage(self, Exception, "Schema has no item 'AB' in relationalPath ['A', 'AB', 'B']",
                                     RelationalValidity.checkRelationalPathValidity, schema, ['A', 'AB', 'B'])

        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addEntity('C')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.ONE))

        # Check that alternates between entity and relationships
        TestUtil.assertRaisesMessage(self, Exception, "Invalid item1Name 'A' and item2Name 'B' in relationalPath: "
                                                      "types must alternate between entities and relationships",
                                     RelationalValidity.checkRelationalPathValidity, schema, ['A', 'B'])

        TestUtil.assertRaisesMessage(self, Exception, "Invalid item1Name 'AB' and item2Name 'AB' in relationalPath: "
                                                      "types must alternate between entities and relationships",
                                     RelationalValidity.checkRelationalPathValidity, schema, ['AB', 'AB'])

        TestUtil.assertRaisesMessage(self, Exception, "Invalid item1Name 'AB' and item2Name 'AB' in relationalPath: "
                                                      "types must alternate between entities and relationships",
                                     RelationalValidity.checkRelationalPathValidity, schema, ['A', 'AB', 'AB'])

        TestUtil.assertRaisesMessage(self, Exception, "Invalid item1Name 'B' and item2Name 'B' in relationalPath: "
                                                      "types must alternate between entities and relationships",
                                     RelationalValidity.checkRelationalPathValidity, schema, ['A', 'AB', 'B', 'B'])

        # Check that entities participate in connected relationships
        TestUtil.assertRaisesMessage(self, Exception, "Entity 'C' does not participate in relationship 'AB'",
                                     RelationalValidity.checkRelationalPathValidity, schema, ['A', 'AB', 'C'])

        TestUtil.assertRaisesMessage(self, Exception, "Entity 'A' does not participate in relationship 'BC'",
                                     RelationalValidity.checkRelationalPathValidity, schema, ['BC', 'A'])

        TestUtil.assertRaisesMessage(self, Exception, "Entity 'C' does not participate in relationship 'AB'",
                                     RelationalValidity.checkRelationalPathValidity, schema, ['C', 'AB'])

        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.MANY))

        # Check cardinality constraints (no ERE, RER -> card(R, E) = MANY
        TestUtil.assertRaisesMessage(self, Exception, "Found ERE pattern in relationalPath",
                                     RelationalValidity.checkRelationalPathValidity, schema, ['A', 'AB', 'A'])

        TestUtil.assertRaisesMessage(self, Exception, "Found ERE pattern in relationalPath",
                                     RelationalValidity.checkRelationalPathValidity, schema, ['B', 'AB', 'B'])

        TestUtil.assertRaisesMessage(self, Exception, "Found RER pattern in relationalPath with card(R, E) = ONE",
                                     RelationalValidity.checkRelationalPathValidity, schema, ['AB', 'A', 'AB'])

        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.ONE))

        TestUtil.assertRaisesMessage(self, Exception, "Found RER pattern in relationalPath with card(R, E) = ONE",
                                     RelationalValidity.checkRelationalPathValidity, schema, ['AB', 'B', 'AB'])


    def testInvalidRelationalVariables(self):
        schema = Schema()
        schema.addEntity('A')
        TestUtil.assertRaisesMessage(self, Exception, "Schema item 'A' has no attribute 'X' in relationalVariable '[A].X'",
                                     RelationalValidity.checkRelationalVariableValidity, schema, ParserUtil.parseRelVar('[A].X'))

        schema.addAttribute('A', 'X')
        TestUtil.assertRaisesMessage(self, Exception, "Schema item 'A' has no attribute 'Y' in relationalVariable '[A].Y'",
                                     RelationalValidity.checkRelationalVariableValidity, schema, ParserUtil.parseRelVar('[A].Y'))

        self.assertIsNone(RelationalValidity.checkRelationalVariableValidity(schema, ParserUtil.parseRelVar('[A].exists')))

        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        TestUtil.assertRaisesMessage(self, Exception, "Schema item 'AB' has no attribute 'XY' in relationalVariable '[A, AB].XY'",
                                     RelationalValidity.checkRelationalVariableValidity, schema, ParserUtil.parseRelVar('[A, AB].XY'))

        self.assertIsNone(RelationalValidity.checkRelationalVariableValidity(schema, ParserUtil.parseRelVar('[A, AB].exists')))

        TestUtil.assertRaisesMessage(self, Exception, "Schema item 'B' has no attribute 'Y' in relationalVariable '[A, AB, B].Y'",
                                     RelationalValidity.checkRelationalVariableValidity, schema, ParserUtil.parseRelVar('[A, AB, B].Y'))
        schema.addAttribute('B', 'Y')

        # enforce that the relational paths are checked for consistency against the schema
        # using RelationalValidity.checkRelationalPathValidity
        mockRelPathChecker = MagicMock(wraps=RelationalValidity.checkRelationalPathValidity)
        RelationalValidity.checkRelationalVariableValidity(schema, ParserUtil.parseRelVar('[A, AB, B].Y'),
                                                  relationalPathChecker=mockRelPathChecker)
        self.assertEqual(1, mockRelPathChecker.call_count)


    def testInvalidRelationalDependencies(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        schema.addAttribute('A', 'X')
        schema.addAttribute('B', 'Y')

        # Check that entire dependency is canonical (relVar2 has a singleton path)
        TestUtil.assertRaisesMessage(self, Exception, "Dependency '[B].Y -> [B, AB, A].X' is not canonical",
                                     RelationalValidity.checkRelationalDependencyValidity, schema, ParserUtil.parseRelDep('[B].Y -> [B, AB, A].X'))

        # Check that base items are the same in both paths
        TestUtil.assertRaisesMessage(self, Exception, "Dependency '[B].Y -> [A].X' has inconsistent base items",
                                     RelationalValidity.checkRelationalDependencyValidity, schema, ParserUtil.parseRelDep('[B].Y -> [A].X'))

        # enforce that the relational variables are checked for consistency against the schema
        # using RelationalValidity.checkRelationalVariableValidity
        mockRelVarChecker = MagicMock(wraps=RelationalValidity.checkRelationalVariableValidity)
        RelationalValidity.checkRelationalDependencyValidity(schema, ParserUtil.parseRelDep('[A, AB, B].Y -> [A].X'),
                                                  relationalVariableChecker=mockRelVarChecker)
        self.assertEqual(2, mockRelVarChecker.call_count)


    def testInvalidSetOfRelationalVariables(self):
        # all relvars must have the same perspective
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        schema.addAttribute('A', 'X')
        schema.addAttribute('B', 'Y')

        TestUtil.assertRaisesMessage(self, Exception, "Perspective is not consistent across all relational variables",
                                 RelationalValidity.checkValidityOfRelationalVariableSet, schema, 0,
                                 [ParserUtil.parseRelVar(relVarStr) for relVarStr in ['[A].X', '[B].Y']])

        TestUtil.assertRaisesMessage(self, Exception, "Perspective is not consistent across all relational variables",
                                 RelationalValidity.checkValidityOfRelationalVariableSet, schema, 0,
                                 [ParserUtil.parseRelVar(relVarStr) for relVarStr in ['[B].Y', '[A].X']])

        # all relvars must be consistent with hop threshold
        TestUtil.assertRaisesMessage(self, Exception, "Relational variable '[A, AB, B].Y' is longer than the hop threshold",
                                     RelationalValidity.checkValidityOfRelationalVariableSet, schema, 0,
                                     [ParserUtil.parseRelVar(relVarStr) for relVarStr in ['[A].X', '[A, AB, B].Y']])

        TestUtil.assertRaisesMessage(self, Exception, "Relational variable '[A, AB, B].Y' is longer than the hop threshold",
                                     RelationalValidity.checkValidityOfRelationalVariableSet, schema, 1,
                                     [ParserUtil.parseRelVar(relVarStr) for relVarStr in ['[A, AB, B].Y', '[A].X']])

        # enforce that the relational variables are checked for consistency against the schema
        # using RelationalValidity.checkRelationalVariableValidity
        mockRelVarChecker = MagicMock(wraps=RelationalValidity.checkRelationalVariableValidity)
        RelationalValidity.checkValidityOfRelationalVariableSet(schema, 2,
                                                                [ParserUtil.parseRelVar(relVarStr) for relVarStr in ['[A, AB, B].Y', '[A].X']],
                                                                relationalVariableChecker=mockRelVarChecker)
        self.assertEqual(2, mockRelVarChecker.call_count)


if __name__ == '__main__':
    unittest.main()
