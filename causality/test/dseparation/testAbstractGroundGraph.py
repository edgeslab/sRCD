import unittest
from causality.modelspace import RelationalSpace
from causality.model import ParserUtil
from causality.model.Model import Model
from causality.model.Schema import Schema
from causality.dseparation.AbstractGroundGraph import AbstractGroundGraph
from causality.model.RelationalDependency import RelationalVariable
from causality.model.RelationalDependency import RelationalVariableIntersection
from causality.test import TestUtil

class TestAbstractGroundGraph(unittest.TestCase):

    def testPropositionalAGG(self):
        schema = Schema()
        schema.addEntity('A')
        model = Model(schema, [])
        agg = AbstractGroundGraph(model, 'A', 0)
        self.assertAGGEqualNoIntersection(schema, agg, [])

        schema.addAttribute('A', 'A')
        schema.addAttribute('A', 'B')
        schema.addAttribute('A', 'C')
        model = Model(schema, [])
        agg = AbstractGroundGraph(model, 'A', 0)
        self.assertAGGEqualNoIntersection(schema, agg, [])

        schema.addAttribute('A', 'D')
        schema.addAttribute('A', 'E')
        schema.addAttribute('A', 'F')
        schema.addAttribute('A', 'G')
        schema.addAttribute('A', 'H')
        dependencies = ['[A].A -> [A].B', '[A].A -> [A].C', '[A].B -> [A].D', '[A].C -> [A].D',
                        '[A].E -> [A].F', '[A].E -> [A].G', '[A].F -> [A].H', '[A].G -> [A].H']
        model = Model(schema, dependencies)
        agg = AbstractGroundGraph(model, 'A', 0)
        self.assertAGGEqualNoIntersection(schema, agg, dependencies)


    def testOneToOneTwoEntityAGG(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        model = Model(schema, [])
        self.assertAGGEqualNoIntersection(schema, AbstractGroundGraph(model, 'A', 0), [])
        self.assertAGGEqualNoIntersection(schema, AbstractGroundGraph(model, 'B', 0), [])
        self.assertAGGEqualNoIntersection(schema, AbstractGroundGraph(model, 'AB', 0), [])

        schema.addAttribute('A', 'X')
        model = Model(schema, [])
        self.assertAGGEqualNoIntersection(schema, AbstractGroundGraph(model, 'A', 0), [])
        self.assertAGGEqualNoIntersection(schema, AbstractGroundGraph(model, 'B', 1), []) # insufficient hop threshold to give it nodes
        self.assertAGGEqualNoIntersection(schema, AbstractGroundGraph(model, 'B', 2), [])
        self.assertAGGEqualNoIntersection(schema, AbstractGroundGraph(model, 'AB', 1), [])

        schema.addAttribute('B', 'Y')
        model = Model(schema, [])
        self.assertAGGEqualNoIntersection(schema, AbstractGroundGraph(model, 'A', 2), [])
        self.assertAGGEqualNoIntersection(schema, AbstractGroundGraph(model, 'B', 2), [])
        self.assertAGGEqualNoIntersection(schema, AbstractGroundGraph(model, 'AB', 1), [])

        schema.addAttribute('AB', 'XY')
        model = Model(schema, [])
        self.assertAGGEqualNoIntersection(schema, AbstractGroundGraph(model, 'A', 2), [])
        self.assertAGGEqualNoIntersection(schema, AbstractGroundGraph(model, 'B', 2), [])
        self.assertAGGEqualNoIntersection(schema, AbstractGroundGraph(model, 'AB', 1), [])

        model = Model(schema, ['[B, AB, A].X -> [B].Y'])
        self.assertAGGEqualNoIntersection(schema, AbstractGroundGraph(model, 'A', 2), ['[A].X -> [A, AB, B].Y'])
        self.assertAGGEqualNoIntersection(schema, AbstractGroundGraph(model, 'B', 2), ['[B, AB, A].X -> [B].Y'])
        self.assertAGGEqualNoIntersection(schema, AbstractGroundGraph(model, 'AB', 1), ['[AB, A].X -> [AB, B].Y'])


    def testOneToManyTwoEntityAGG(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.ONE))
        model = Model(schema, [])
        self.assertAGGEqualNoIntersection(schema, AbstractGroundGraph(model, 'A', 0), [])
        self.assertAGGEqualNoIntersection(schema, AbstractGroundGraph(model, 'B', 0), [])
        self.assertAGGEqualNoIntersection(schema, AbstractGroundGraph(model, 'AB', 0), [])

        schema.addAttribute('A', 'X')
        model = Model(schema, [])
        self.assertAGGEqualNoIntersection(schema, AbstractGroundGraph(model, 'A', 2), [])
        self.assertAGGEqualNoIntersection(schema, AbstractGroundGraph(model, 'B', 2), [])
        self.assertAGGEqualNoIntersection(schema, AbstractGroundGraph(model, 'AB', 2), [])

        schema.addAttribute('B', 'Y')
        model = Model(schema, [])
        self.assertAGGEqualNoIntersection(schema, AbstractGroundGraph(model, 'A', 2), []) # putting in max hop threshold
        self.assertAGGEqualNoIntersection(schema, AbstractGroundGraph(model, 'B', 4), [])
        abAGG = AbstractGroundGraph(model, 'AB', 3)
        expectedRelVarNodes = [ParserUtil.parseRelVar(relVarStr) for relVarStr in ['[AB, A].X', '[AB, B].Y', '[AB, A, AB, B].Y']]
        expectedRelVarIntNodes = self.relVarStrPairsToRelVarInts([('[AB, B].Y', '[AB, A, AB, B].Y')])
        TestUtil.assertUnorderedListEqual(self, expectedRelVarNodes, abAGG.getRelationalVariableNodes())
        TestUtil.assertUnorderedListEqual(self, expectedRelVarIntNodes, abAGG.getRelationalVariableIntersectionNodes())
        self.assertAGGEdgesEqual([], abAGG)
        # test that order of relVar1, relVar2 for RelationalVariableIntersection doesn't matter
        expectedRelVarIntNodes = self.relVarStrPairsToRelVarInts([('[AB, A, AB, B].Y', '[AB, B].Y')])
        TestUtil.assertUnorderedListEqual(self, expectedRelVarIntNodes, abAGG.getRelationalVariableIntersectionNodes())

        model = Model(schema, ['[A, AB, B].Y -> [A].X'])
        aAGG = AbstractGroundGraph(model, 'A', 2)
        expectedEdges = self.relVarStrPairsToRelVarPairs([('[A, AB, B].Y', '[A].X')])
        self.assertAGGEdgesEqual(expectedEdges, aAGG)

        # test that extended dependencies are only added among nodes that exist in the AGG (have an appropriate number of hops)
        bAGG = AbstractGroundGraph(model, 'B', 2)
        expectedEdges = self.relVarStrPairsToRelVarPairs([('[B].Y', '[B, AB, A].X')])
        self.assertAGGEdgesEqual(expectedEdges, bAGG)

        bAGG = AbstractGroundGraph(model, 'B', 4)
        expectedEdges = self.relVarStrPairsToRelVarPairs([('[B].Y', '[B, AB, A].X'),
                                                          ('[B, AB, A, AB, B].Y', '[B, AB, A].X')])
        self.assertAGGEdgesEqual(expectedEdges, bAGG)

        # test dependencies get inherited for intersection nodes
        abAGG = AbstractGroundGraph(model, 'AB', 3)
        expectedEdges = self.relVarStrPairsToRelVarPairs([('[AB, B].Y', '[AB, A].X'),
                                                          ('[AB, A, AB, B].Y', '[AB, A].X'),
                                                          (('[AB, A, AB, B].Y', '[AB, B].Y'), '[AB, A].X')])
        self.assertAGGEdgesEqual(expectedEdges, abAGG)

        schema.addAttribute('B', 'Z')
        model = Model(schema, [])
        abAGG = AbstractGroundGraph(model, 'AB', 3)
        expectedRelVarNodes = [ParserUtil.parseRelVar(relVarStr) for relVarStr in
                               ['[AB, A].X', '[AB, B].Y', '[AB, A, AB, B].Y', '[AB, B].Z', '[AB, A, AB, B].Z']]
        expectedRelVarIntNodes = self.relVarStrPairsToRelVarInts([('[AB, B].Y', '[AB, A, AB, B].Y'),
                                                                  ('[AB, B].Z', '[AB, A, AB, B].Z')])
        TestUtil.assertUnorderedListEqual(self, expectedRelVarNodes, abAGG.getRelationalVariableNodes())
        TestUtil.assertUnorderedListEqual(self, expectedRelVarIntNodes, abAGG.getRelationalVariableIntersectionNodes())
        self.assertAGGEdgesEqual([], abAGG)

        model = Model(schema, ['[A, AB, B].Y -> [A].X', '[A, AB, B].Z -> [A].X', '[B].Y -> [B].Z'])
        aAGG = AbstractGroundGraph(model, 'A', 2)
        expectedEdges = self.relVarStrPairsToRelVarPairs([('[A, AB, B].Y', '[A].X'), ('[A, AB, B].Z', '[A].X'),
                                                            ('[A, AB, B].Y', '[A, AB, B].Z')])
        self.assertAGGEdgesEqual(expectedEdges, aAGG)

        abAGG = AbstractGroundGraph(model, 'AB', 3)
        expectedEdges = self.relVarStrPairsToRelVarPairs([
            ('[AB, A, AB, B].Y', '[AB, A, AB, B].Z'), ('[AB, A, AB, B].Y', '[AB, A].X'), ('[AB, B].Y', '[AB, B].Z'),
            ('[AB, A, AB, B].Z', '[AB, A].X'), ('[AB, B].Y', '[AB, A].X'), ('[AB, B].Z', '[AB, A].X'),
            (('[AB, B].Y', '[AB, A, AB, B].Y'), '[AB, A].X'), (('[AB, B].Y', '[AB, A, AB, B].Y'), '[AB, B].Z'),
            (('[AB, B].Y', '[AB, A, AB, B].Y'), '[AB, A, AB, B].Z'),
            (('[AB, B].Z', '[AB, A, AB, B].Z'), '[AB, A].X'), ('[AB, A, AB, B].Y', ('[AB, B].Z', '[AB, A, AB, B].Z')),
            ('[AB, B].Y', ('[AB, B].Z', '[AB, A, AB, B].Z'))])
        self.assertAGGEdgesEqual(expectedEdges, abAGG)


    def testManyToManyTwoEntityTwoRelationshipsAGGNodes(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB1', ('A', Schema.MANY), ('B', Schema.MANY))
        schema.addRelationship('AB2', ('A', Schema.MANY), ('B', Schema.MANY))
        schema.addAttribute('A', 'X')
        schema.addAttribute('B', 'Y')
        schema.addAttribute('AB1', 'XY1')
        schema.addAttribute('AB2', 'XY2')
        model = Model(schema, [])
        aAGG = AbstractGroundGraph(model, 'A', 4)
        expectedRelVarNodes = [ParserUtil.parseRelVar(relVarStr) for relVarStr in
                               ['[A].X', '[A, AB1].XY1', '[A, AB2].XY2', '[A, AB1, B].Y', '[A, AB2, B].Y',
                                '[A, AB1, B, AB1].XY1', '[A, AB1, B, AB2].XY2', '[A, AB2, B, AB1].XY1',
                                '[A, AB2, B, AB2].XY2', '[A, AB1, B, AB1, A].X', '[A, AB1, B, AB2, A].X',
                                '[A, AB2, B, AB1, A].X', '[A, AB2, B, AB2, A].X']]
        expectedRelVarIntNodes = self.relVarStrPairsToRelVarInts([('[A, AB1].XY1', '[A, AB2, B, AB1].XY1'),
            ('[A, AB2].XY2', '[A, AB1, B, AB2].XY2'), ('[A, AB1, B].Y', '[A, AB2, B].Y'),
            ('[A, AB1, B, AB1].XY1', '[A, AB2, B, AB1].XY1'), ('[A, AB2, B, AB2].XY2', '[A, AB1, B, AB2].XY2'),
            ('[A, AB1, B, AB1, A].X', '[A, AB1, B, AB2, A].X'), ('[A, AB1, B, AB1, A].X', '[A, AB2, B, AB1, A].X'),
            ('[A, AB1, B, AB1, A].X', '[A, AB2, B, AB2, A].X'), ('[A, AB2, B, AB2, A].X', '[A, AB1, B, AB2, A].X'),
            ('[A, AB2, B, AB2, A].X', '[A, AB2, B, AB1, A].X'), ('[A, AB1, B, AB2, A].X', '[A, AB2, B, AB1, A].X')])
        TestUtil.assertUnorderedListEqual(self, expectedRelVarNodes, aAGG.getRelationalVariableNodes())
        TestUtil.assertUnorderedListEqual(self, expectedRelVarIntNodes, aAGG.getRelationalVariableIntersectionNodes())
        self.assertAGGEdgesEqual([], aAGG)

        ab1AGG = AbstractGroundGraph(model, 'AB1', 4)
        expectedRelVarNodes = [ParserUtil.parseRelVar(relVarStr) for relVarStr in
                               ['[AB1].XY1', '[AB1, A].X', '[AB1, B].Y', '[AB1, A, AB1].XY1', '[AB1, A, AB2].XY2',
                                '[AB1, B, AB1].XY1', '[AB1, B, AB2].XY2', '[AB1, A, AB1, B].Y', '[AB1, A, AB2, B].Y',
                                '[AB1, B, AB1, A].X', '[AB1, B, AB2, A].X', '[AB1, A, AB1, B, AB1].XY1',
                                '[AB1, A, AB2, B, AB1].XY1', '[AB1, B, AB1, A, AB1].XY1', '[AB1, B, AB2, A, AB1].XY1',
                                '[AB1, A, AB1, B, AB2].XY2', '[AB1, A, AB2, B, AB2].XY2',
                                '[AB1, B, AB1, A, AB2].XY2', '[AB1, B, AB2, A, AB2].XY2']]
        TestUtil.assertUnorderedListEqual(self, expectedRelVarNodes, ab1AGG.getRelationalVariableNodes())
        expectedRelVarIntNodes = self.relVarStrPairsToRelVarInts([('[AB1, A].X', '[AB1, B, AB1, A].X'),
            ('[AB1, A].X', '[AB1, B, AB2, A].X'), ('[AB1, B].Y', '[AB1, A, AB1, B].Y'),
            ('[AB1, B].Y', '[AB1, A, AB2, B].Y'), ('[AB1, A, AB1].XY1','[AB1, B, AB1].XY1'),
            ('[AB1, A, AB1].XY1','[AB1, A, AB2, B, AB1].XY1'), ('[AB1, A, AB1].XY1', '[AB1, B, AB1, A, AB1].XY1'),
            ('[AB1, A, AB1].XY1','[AB1, B, AB2, A, AB1].XY1'), ('[AB1, A, AB2].XY2','[AB1, B, AB2].XY2'),
            ('[AB1, A, AB2].XY2','[AB1, B, AB2, A, AB2].XY2'), ('[AB1, A, AB2].XY2', '[AB1, B, AB1, A, AB2].XY2'),
            ('[AB1, A, AB2].XY2','[AB1, A, AB1, B, AB2].XY2'), ('[AB1, B, AB1].XY1','[AB1, B, AB2, A, AB1].XY1'),
            ('[AB1, B, AB1].XY1', '[AB1, A, AB1, B, AB1].XY1'), ('[AB1, B, AB1].XY1','[AB1, A, AB2, B, AB1].XY1'),
            ('[AB1, B, AB2].XY2','[AB1, A, AB2, B, AB2].XY2'), ('[AB1, B, AB2].XY2', '[AB1, A, AB1, B, AB2].XY2'),
            ('[AB1, B, AB2].XY2','[AB1, B, AB1, A, AB2].XY2'), ('[AB1, A, AB1, B].Y', '[AB1, A, AB2, B].Y'),
            ('[AB1, B, AB1, A].X', '[AB1, B, AB2, A].X'), ('[AB1, A, AB1, B, AB1].XY1', '[AB1, A, AB2, B, AB1].XY1'),
            ('[AB1, A, AB1, B, AB1].XY1', '[AB1, B, AB1, A, AB1].XY1'),
            ('[AB1, A, AB1, B, AB1].XY1', '[AB1, B, AB2, A, AB1].XY1'),
            ('[AB1, A, AB2, B, AB1].XY1', '[AB1, B, AB1, A, AB1].XY1'),
            ('[AB1, A, AB2, B, AB1].XY1', '[AB1, B, AB2, A, AB1].XY1'),
            ('[AB1, B, AB1, A, AB1].XY1', '[AB1, B, AB2, A, AB1].XY1'),
            ('[AB1, A, AB1, B, AB2].XY2', '[AB1, A, AB2, B, AB2].XY2'),
            ('[AB1, A, AB1, B, AB2].XY2', '[AB1, B, AB1, A, AB2].XY2'),
            ('[AB1, A, AB1, B, AB2].XY2', '[AB1, B, AB2, A, AB2].XY2'),
            ('[AB1, A, AB2, B, AB2].XY2', '[AB1, B, AB1, A, AB2].XY2'),
            ('[AB1, A, AB2, B, AB2].XY2', '[AB1, B, AB2, A, AB2].XY2'),
            ('[AB1, B, AB1, A, AB2].XY2', '[AB1, B, AB2, A, AB2].XY2')])
        TestUtil.assertUnorderedListEqual(self, expectedRelVarIntNodes, ab1AGG.getRelationalVariableIntersectionNodes())
        self.assertAGGEdgesEqual([], aAGG)

        # test multiple relationships with two dependencies
        model = Model(schema, ['[AB2, B, AB1, A].X -> [AB2].XY2', '[AB1, B, AB2].XY2 -> [AB1].XY1'])
        ab1AGG = AbstractGroundGraph(model, 'AB1', 3)
        expectedRelVarNodes = [ParserUtil.parseRelVar(relVarStr) for relVarStr in
            ['[AB1].XY1', '[AB1, A].X', '[AB1, B].Y', '[AB1, A, AB1].XY1', '[AB1, A, AB2].XY2',
             '[AB1, B, AB1].XY1', '[AB1, B, AB2].XY2', '[AB1, A, AB1, B].Y', '[AB1, A, AB2, B].Y',
             '[AB1, B, AB1, A].X', '[AB1, B, AB2, A].X']]
        TestUtil.assertUnorderedListEqual(self, expectedRelVarNodes, ab1AGG.getRelationalVariableNodes())
        expectedRelVarIntNodes = self.relVarStrPairsToRelVarInts([('[AB1, A].X', '[AB1, B, AB1, A].X'),
            ('[AB1, A].X', '[AB1, B, AB2, A].X'), ('[AB1, B].Y', '[AB1, A, AB1, B].Y'),
            ('[AB1, B].Y', '[AB1, A, AB2, B].Y'), ('[AB1, A, AB1].XY1','[AB1, B, AB1].XY1'),
            ('[AB1, A, AB2].XY2','[AB1, B, AB2].XY2'), ('[AB1, A, AB1, B].Y', '[AB1, A, AB2, B].Y'),
            ('[AB1, B, AB1, A].X', '[AB1, B, AB2, A].X')])
        TestUtil.assertUnorderedListEqual(self, expectedRelVarIntNodes, ab1AGG.getRelationalVariableIntersectionNodes())
        expectedEdges = self.relVarStrPairsToRelVarPairs([('[AB1, B, AB2].XY2', '[AB1].XY1'),
            ('[AB1, B, AB2].XY2', '[AB1, B, AB1].XY1'), ('[AB1, A].X', '[AB1, B, AB2].XY2') ,
            ('[AB1, B, AB1, A].X', '[AB1, B, AB2].XY2'), (('[AB1, A].X', '[AB1, B, AB1, A].X'), '[AB1, B, AB2].XY2'),
            (('[AB1, A].X', '[AB1, B, AB2, A].X'), '[AB1, B, AB2].XY2'),
            ('[AB1, B, AB2].XY2', ('[AB1, A, AB1].XY1','[AB1, B, AB1].XY1')),
            ('[AB1, A].X', ('[AB1, A, AB2].XY2','[AB1, B, AB2].XY2')),
            ('[AB1, B, AB1, A].X', ('[AB1, A, AB2].XY2','[AB1, B, AB2].XY2')),
            (('[AB1, A, AB2].XY2','[AB1, B, AB2].XY2'), '[AB1].XY1'),
            (('[AB1, A, AB2].XY2','[AB1, B, AB2].XY2'), '[AB1, B, AB1].XY1'),
            (('[AB1, B, AB1, A].X', '[AB1, B, AB2, A].X'), '[AB1, B, AB2].XY2')])
        self.assertAGGEdgesEqual(expectedEdges, ab1AGG)


    def testThreeEntityTwoRelationshipsAGG(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addEntity('C')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.MANY))
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.MANY))
        schema.addAttribute('A', 'X')
        schema.addAttribute('B', 'Y')
        schema.addAttribute('C', 'Z')
        schema.addAttribute('AB', 'XY')
        schema.addAttribute('BC', 'YZ')
        model = Model(schema, [])
        aAGG = AbstractGroundGraph(model, 'A', 6)
        expectedRelVarNodes = [ParserUtil.parseRelVar(relVarStr) for relVarStr in
                               ['[A].X', '[A, AB].XY', '[A, AB, B].Y', '[A, AB, B, AB].XY', '[A, AB, B, BC].YZ',
                                '[A, AB, B, AB, A].X', '[A, AB, B, BC, C].Z', '[A, AB, B, AB, A, AB].XY',
                                '[A, AB, B, BC, C, BC].YZ', '[A, AB, B, AB, A, AB, B].Y', '[A, AB, B, BC, C, BC, B].Y']]
        TestUtil.assertUnorderedListEqual(self, expectedRelVarNodes, aAGG.getRelationalVariableNodes())
        expectedRelVarIntNodes = self.relVarStrPairsToRelVarInts(
            [('[A, AB, B, AB, A, AB, B].Y', '[A, AB, B, BC, C, BC, B].Y')])
        TestUtil.assertUnorderedListEqual(self, expectedRelVarIntNodes, aAGG.getRelationalVariableIntersectionNodes())
        self.assertAGGEdgesEqual([], aAGG)

        bcAGG = AbstractGroundGraph(model, 'BC', 4)
        expectedRelVarNodes = [ParserUtil.parseRelVar(relVarStr) for relVarStr in
                               ['[BC].YZ', '[BC, B].Y', '[BC, C].Z', '[BC, B, AB].XY', '[BC, C, BC].YZ',
                                '[BC, B, AB, A].X', '[BC, C, BC, B].Y', '[BC, B, AB, A, AB].XY', '[BC, C, BC, B, AB].XY']]
        TestUtil.assertUnorderedListEqual(self, expectedRelVarNodes, bcAGG.getRelationalVariableNodes())
        expectedRelVarIntNodes = self.relVarStrPairsToRelVarInts([('[BC, B].Y', '[BC, C, BC, B].Y'),
            ('[BC, B, AB].XY', '[BC, C, BC, B, AB].XY'), ('[BC, B, AB, A, AB].XY', '[BC, C, BC, B, AB].XY')])
        TestUtil.assertUnorderedListEqual(self, expectedRelVarIntNodes, bcAGG.getRelationalVariableIntersectionNodes())
        self.assertAGGEdgesEqual([], bcAGG)

        model = Model(schema, ['[BC, B, AB, A].X -> [BC].YZ', '[AB, B, BC, C].Z -> [AB].XY',
                               '[AB, B, AB, A, AB, B].Y -> [AB].XY'])
        aAGG = AbstractGroundGraph(model, 'A', 6)
        expectedEdges = self.relVarStrPairsToRelVarPairs([('[A].X', '[A, AB, B, BC].YZ'),
            ('[A, AB, B, AB, A].X', '[A, AB, B, BC].YZ'), ('[A, AB, B, BC, C].Z', '[A, AB].XY'),
            ('[A, AB, B, BC, C].Z', '[A, AB, B, AB].XY'), ('[A, AB, B, AB, A, AB, B].Y', '[A, AB].XY'),
            ('[A, AB, B].Y', '[A, AB, B, AB].XY'), ('[A, AB, B, AB, A, AB, B].Y', '[A, AB, B, AB].XY'),
            (('[A, AB, B, AB, A, AB, B].Y', '[A, AB, B, BC, C, BC, B].Y'), '[A, AB].XY'),
            (('[A, AB, B, AB, A, AB, B].Y', '[A, AB, B, BC, C, BC, B].Y'), '[A, AB, B, AB].XY')])
        self.assertAGGEdgesEqual(expectedEdges, aAGG)


    def testLongRangeDependencyIsIgnored(self):
        # Build AGG with model with a dependency that is longer than hop threshold
        # the long-range dependence is not (B,h)-reachable for the AGG from perspective B
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.ONE))
        schema.addAttribute('B', 'Y1')
        schema.addAttribute('B', 'Y2')
        model = Model(schema, ['[B, AB, A, AB, B].Y1 -> [B].Y2'])
        self.assertAGGEqualNoIntersection(schema, AbstractGroundGraph(model, 'B', 2), [])


    def testBadRelVarIntInput(self):
        TestUtil.assertRaisesMessage(self, Exception, "RelationalVariableIntersection expects two RelationalVariable objects",
                 RelationalVariableIntersection, None, RelationalVariable(['A'], 'X'))

        TestUtil.assertRaisesMessage(self, Exception, "RelationalVariableIntersection expects two RelationalVariable objects",
                 RelationalVariableIntersection, RelationalVariable(['A'], 'X'), None)


    def testBadPerspective(self):
        schema = Schema()
        model = Model(schema, [])
        # non-string
        TestUtil.assertRaisesMessage(self, Exception, "Perspective must be a valid schema item name",
                 AbstractGroundGraph, model, None, 0)

        # string, but bad item name
        TestUtil.assertRaisesMessage(self, Exception, "Perspective must be a valid schema item name",
                 AbstractGroundGraph, model, 'A', 0)

        schema = Schema()
        schema.addEntity('A')
        model = Model(schema, [])
        TestUtil.assertRaisesMessage(self, Exception, "Perspective must be a valid schema item name",
                 AbstractGroundGraph, model, 'B', 0)


    def testBadHopThresholdInput(self):
        # hop thresholds must be non-negative integers
        schema = Schema()
        schema.addEntity('A')
        model = Model(schema, [])
        TestUtil.assertRaisesMessage(self, Exception, "hopThreshold must be a non-negative integer",
                 AbstractGroundGraph, model, 'A', None)
        TestUtil.assertRaisesMessage(self, Exception, "hopThreshold must be a non-negative integer",
                 AbstractGroundGraph, model, 'A', 1.5)
        TestUtil.assertRaisesMessage(self, Exception, "hopThreshold must be a non-negative integer",
                 AbstractGroundGraph, model, 'A', -1)


    def testGetSubsumedRelVarInts(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.ONE))
        schema.addAttribute('A', 'X')
        schema.addAttribute('B', 'Y')
        model = Model(schema, [])
        abAGG = AbstractGroundGraph(model, 'AB', 3)
        self.assertSameSubsumedVariables(['[AB, A].X'], abAGG, '[AB, A].X')
        self.assertSameSubsumedVariables(['[AB, B].Y', ('[AB, B].Y', '[AB, A, AB, B].Y')], abAGG, '[AB, B].Y')
        self.assertSameSubsumedVariables(['[AB, A, AB, B].Y', ('[AB, B].Y', '[AB, A, AB, B].Y')], abAGG, '[AB, A, AB, B].Y')

        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB1', ('A', Schema.MANY), ('B', Schema.MANY))
        schema.addRelationship('AB2', ('A', Schema.MANY), ('B', Schema.MANY))
        schema.addAttribute('A', 'X')
        schema.addAttribute('B', 'Y')
        schema.addAttribute('AB1', 'XY1')
        schema.addAttribute('AB2', 'XY2')
        model = Model(schema, [])
        aAGG = AbstractGroundGraph(model, 'A', 4)
        self.assertSameSubsumedVariables(['[A, AB1, B, AB1, A].X', ('[A, AB1, B, AB1, A].X', '[A, AB1, B, AB2, A].X'),
            ('[A, AB1, B, AB1, A].X', '[A, AB2, B, AB1, A].X'), ('[A, AB1, B, AB1, A].X', '[A, AB2, B, AB2, A].X')],
            aAGG, '[A, AB1, B, AB1, A].X')

        # test bad relVar input to getSubsumedVariables
        schema = Schema()
        schema.addEntity('A')
        model = Model(schema, [])
        agg = AbstractGroundGraph(model, 'A', 0)
        TestUtil.assertRaisesMessage(self, Exception, "relVar must be a RelationalVariable: found 'None'",
            agg.getSubsumedVariables, None)

        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X1')
        model = Model(schema, [])
        agg = AbstractGroundGraph(model, 'A', 0)
        TestUtil.assertRaisesMessage(self, Exception, "relVar '[A].X2' is not a node in the abstract ground graph",
            agg.getSubsumedVariables, RelationalVariable(['A'], 'X2'))


    def testGetAncestors(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        model = Model(schema, [])
        agg = AbstractGroundGraph(model, 'A', 0)
        self.assertGetAncestorsEquals(['[A].X'], agg, '[A].X')

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
        dependencies = ['[A].A -> [A].C', '[A].A -> [A].D', '[A].B -> [A].D', '[A].C -> [A].E',
                        '[A].D -> [A].E', '[A].E -> [A].G', '[A].E -> [A].H', '[A].F -> [A].H']
        model = Model(schema, dependencies)
        agg = AbstractGroundGraph(model, 'A', 0)
        self.assertGetAncestorsEquals(['[A].G', '[A].A', '[A].B','[A].C', '[A].D', '[A].E'], agg, '[A].G')
        self.assertGetAncestorsEquals(['[A].F', '[A].A', '[A].B','[A].C', '[A].D', '[A].E', '[A].H'], agg, '[A].H')

        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addEntity('C')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.MANY))
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.MANY))
        schema.addAttribute('A', 'X')
        schema.addAttribute('B', 'Y')
        schema.addAttribute('C', 'Z')
        schema.addAttribute('AB', 'XY')
        schema.addAttribute('BC', 'YZ')
        model = Model(schema, ['[BC, B, AB, A].X -> [BC].YZ', '[AB, B, BC, C].Z -> [AB].XY',
                               '[AB, B, AB, A, AB, B].Y -> [AB].XY'])
        agg = AbstractGroundGraph(model, 'A', 6)
        self.assertGetAncestorsEquals(['[A, AB].XY', '[A, AB, B, BC, C].Z', '[A, AB, B, AB, A, AB, B].Y',
                                       ('[A, AB, B, AB, A, AB, B].Y', '[A, AB, B, BC, C, BC, B].Y')], agg, '[A, AB].XY')

        self.assertGetAncestorsEquals([('[A, AB, B, AB, A, AB, B].Y', '[A, AB, B, BC, C, BC, B].Y')], agg,
                                      ('[A, AB, B, AB, A, AB, B].Y', '[A, AB, B, BC, C, BC, B].Y'))


    def testRemoveEdgesForDependency(self):
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
        agg = AbstractGroundGraph(model, 'A', 0)
        agg.removeEdgesForDependency(ParserUtil.parseRelDep('[A].B -> [A].A'))
        self.assertEqual(8, len(agg.edges()))
        agg.removeEdgesForDependency(ParserUtil.parseRelDep('[A].A -> [A].B'))
        self.assertEqual(7, len(agg.edges()))
        self.assertNotIn((ParserUtil.parseRelVar('[A].A'), ParserUtil.parseRelVar('[A].B')), agg.edges())
        agg.removeEdgesForDependency(ParserUtil.parseRelDep('[A].F -> [A].H'))
        self.assertEqual(6, len(agg.edges()))
        self.assertNotIn((ParserUtil.parseRelVar('[A].F'), ParserUtil.parseRelVar('[A].H')), agg.edges())

        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addEntity('C')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.MANY))
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.MANY))
        schema.addAttribute('A', 'X')
        schema.addAttribute('B', 'Y')
        schema.addAttribute('C', 'Z')
        schema.addAttribute('AB', 'XY')
        schema.addAttribute('BC', 'YZ')
        model = Model(schema, ['[BC, B, AB, A].X -> [BC].YZ', '[AB, B, BC, C].Z -> [AB].XY',
                               '[AB, B, AB, A, AB, B].Y -> [AB].XY'])
        aAGG = AbstractGroundGraph(model, 'A', 6)
        self.assertEqual(9, len(aAGG.edges()))
        aAGG.removeEdgesForDependency(ParserUtil.parseRelDep('[BC, B, AB, A].X -> [BC].YZ'))
        self.assertEqual(7, len(aAGG.edges()))
        aAGG.removeEdgesForDependency(ParserUtil.parseRelDep('[AB, B, AB, A, AB, B].Y -> [AB].XY'))
        self.assertEqual(2, len(aAGG.edges()))


    def testExtendPaths(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.MANY))
        schema.addAttribute('AB', 'XY1')
        schema.addAttribute('AB', 'XY2')
        schema.addAttribute('AB', 'XY3')

        from causality.dseparation import AbstractGroundGraph as AGG_module
        self.assertEqual([['AB', 'B', 'AB', 'A', 'AB', 'B', 'AB']],
                         AGG_module.extendPath(schema, ['AB', 'B', 'AB'], ['AB', 'A', 'AB', 'B', 'AB']))


    def assertAGGEqualNoIntersection(self, schema, actualAgg, expectedRelDepStrs):
        expectedNodes = [relVar for relVar in RelationalSpace.getRelationalVariables(schema, actualAgg.hopThreshold,
                            includeExistence=False) if relVar.getBaseItemName() == actualAgg.perspective]
        expectedEdges = [(ParserUtil.parseRelDep(depStr).relVar1, ParserUtil.parseRelDep(depStr).relVar2)
                         for depStr in expectedRelDepStrs]
        TestUtil.assertUnorderedListEqual(self, expectedNodes, actualAgg.nodes())
        TestUtil.assertUnorderedListEqual(self, expectedEdges, actualAgg.edges())


    def relVarStrPairsToRelVarInts(self, relVarStrPairs):
        return [RelationalVariableIntersection(ParserUtil.parseRelVar(relVarStr1), ParserUtil.parseRelVar(relVarStr2))
            for relVarStr1, relVarStr2 in relVarStrPairs]


    def relVarStrPairsToRelVarPairs(self, relVarStrPairs):
        edges = []
        for relVarStrPair in relVarStrPairs:
            if isinstance(relVarStrPair[0], str) and isinstance(relVarStrPair[1], str):
                # relVar -> relVar
                edges.append((ParserUtil.parseRelVar(relVarStrPair[0]), ParserUtil.parseRelVar(relVarStrPair[1])))
            elif isinstance(relVarStrPair[0], str) and not isinstance(relVarStrPair[1], str):
                # relVar -> relVarInt
                edges.append((ParserUtil.parseRelVar(relVarStrPair[0]),
                              RelationalVariableIntersection(ParserUtil.parseRelVar(relVarStrPair[1][0]),
                                                             ParserUtil.parseRelVar(relVarStrPair[1][1]))))
            elif not isinstance(relVarStrPair[0], str) and isinstance(relVarStrPair[1], str):
                # relVarInt -> relVar
                edges.append((RelationalVariableIntersection(ParserUtil.parseRelVar(relVarStrPair[0][0]),
                                                             ParserUtil.parseRelVar(relVarStrPair[0][1])),
                              ParserUtil.parseRelVar(relVarStrPair[1])))
            else:
                raise Exception("Unknown pairing in relVarStrPairs: {}".format(relVarStrPair))
        return edges


    def assertAGGEdgesEqual(self, expectedEdges, agg):
        actualEdges = agg.edges()
        self.assertEqual(len(expectedEdges), len(actualEdges))
        self.assertEqual(set(expectedEdges), set(actualEdges))


    def assertSameSubsumedVariables(self, expectedAGGNodeStrs, agg, relVarStr):
        relVar = self.createAGGNodeObj(relVarStr)
        expectedVariableObjs = self.createAGGNodeObjs(expectedAGGNodeStrs)
        actualSubsumedVariables = agg.getSubsumedVariables(relVar)
        self.assertEqual(len(expectedVariableObjs), len(actualSubsumedVariables))
        self.assertEqual(set(expectedVariableObjs), set(actualSubsumedVariables))


    def assertGetAncestorsEquals(self, expectedAGGNodeStrs, agg, aggNodeStr):
        aggNode = self.createAGGNodeObj(aggNodeStr)
        expectedAGGNodeObjs = self.createAGGNodeObjs(expectedAGGNodeStrs)
        actualAncestors = agg.getAncestors(aggNode)
        self.assertEqual(len(expectedAGGNodeObjs), len(actualAncestors))
        self.assertEqual(set(expectedAGGNodeObjs), set(actualAncestors))


    def createAGGNodeObj(self, aggNodeStr):
        if isinstance(aggNodeStr, str):
            return ParserUtil.parseRelVar(aggNodeStr)
        else:
            return RelationalVariableIntersection(ParserUtil.parseRelVar(aggNodeStr[0]),
                                                  ParserUtil.parseRelVar(aggNodeStr[1]))


    def createAGGNodeObjs(self, aggNodeStrs):
        return [self.createAGGNodeObj(aggNodeStr) for aggNodeStr in aggNodeStrs]


if __name__ == '__main__':
    unittest.main()
