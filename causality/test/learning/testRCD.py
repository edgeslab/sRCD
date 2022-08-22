from collections import OrderedDict
import unittest
import itertools
from causality.dseparation.AbstractGroundGraph import AbstractGroundGraph
from causality.learning import ModelEvaluation
from causality.test import TestUtil
from causality.model import ParserUtil
from causality.model.Schema import Schema
from causality.model.Model import Model
from causality.model import RelationalValidity
from causality.model.RelationalDependency import RelationalVariable
from causality.learning.RCD import RCD, SchemaDependencyWrapper
from causality.learning import RCD as RCDmodule
from mock import MagicMock
from mock import PropertyMock
from causality.citest.CITest import Oracle

class TestRCD(unittest.TestCase):

    def testRCDObj(self):
        schema = Schema()
        model = Model(schema, [])
        oracle = Oracle(model)
        rcd = RCD(schema, oracle, 0)
        self.assertEqual(schema, rcd.schema)
        self.assertEqual(oracle.model, rcd.citest.model)


    def testBadHopThreshold(self):
        schema = Schema()
        model = Model(schema, [])
        oracle = Oracle(model)
        TestUtil.assertRaisesMessage(self, Exception, "Hop threshold must be a non-negative integer: found None",
            RCD, schema, oracle, None)

        TestUtil.assertRaisesMessage(self, Exception, "Hop threshold must be a non-negative integer: found -1",
            RCD, schema, oracle, -1)

        TestUtil.assertRaisesMessage(self, Exception, "Hop threshold must be a non-negative integer: found 1.5",
            RCD, schema, oracle, 1.5)


    def testIdentifyUndirectedDependenciesOneEntity(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        model = Model(schema, [])
        mockOracle = MagicMock(wraps=Oracle(model))
        mockModelProperty = PropertyMock()
        type(mockOracle).model = mockModelProperty
        rcd = RCD(schema, mockOracle, 0)
        rcd.identifyUndirectedDependencies()
        self.assertEqual(0, mockModelProperty.call_count)   # forces us not to cheat by simply returning the model
        self.assertRCDOutputEqual([], {}, 0, rcd.undirectedDependencies, rcd.sepsets, mockOracle)

        schema.addAttribute('A', 'Y')
        model = Model(schema, [])
        mockOracle = MagicMock(wraps=Oracle(model))
        mockModelProperty = PropertyMock()
        type(mockOracle).model = mockModelProperty
        rcd = RCD(schema, mockOracle, 0)
        rcd.identifyUndirectedDependencies()
        self.assertEqual(0, mockModelProperty.call_count)
        expectedSepset = {('[A].X', '[A].Y'): set(), ('[A].Y', '[A].X'): set()}
        self.assertRCDOutputEqual([], expectedSepset, 1, rcd.undirectedDependencies, rcd.sepsets, mockOracle)

        model = Model(schema, ['[A].X -> [A].Y'])
        mockOracle = MagicMock(wraps=Oracle(model))
        rcd = RCD(schema, mockOracle, 0)
        rcd.identifyUndirectedDependencies()
        expectedDeps = ['[A].X -> [A].Y', '[A].Y -> [A].X']
        self.assertRCDOutputEqual(expectedDeps, expectedSepset, 2, rcd.undirectedDependencies, rcd.sepsets, mockOracle)

        schema.addAttribute('A', 'Z')
        model = Model(schema, ['[A].X -> [A].Y'])
        mockOracle = MagicMock(wraps=Oracle(model))
        rcd = RCD(schema, mockOracle, 0)
        rcd.identifyUndirectedDependencies()
        expectedDeps = ['[A].X -> [A].Y', '[A].Y -> [A].X']
        expectedSepset = {('[A].X', '[A].Z'): set(), ('[A].Z', '[A].X'): set(), ('[A].Y', '[A].Z'): set(),
                          ('[A].Z', '[A].Y'): set()}
        self.assertRCDOutputEqual(expectedDeps, expectedSepset, 4, rcd.undirectedDependencies, rcd.sepsets, mockOracle)

        model = Model(schema, ['[A].X -> [A].Z', '[A].Z -> [A].Y'])
        mockOracle = MagicMock(wraps=Oracle(model))
        rcd = RCD(schema, mockOracle, 0)
        rcd.potentialDependencySorter = lambda l: sorted(l)
        rcd.identifyUndirectedDependencies()
        expectedDeps = ['[A].X -> [A].Z', '[A].Z -> [A].X', '[A].Y -> [A].Z', '[A].Z -> [A].Y']
        expectedSepset = {('[A].X', '[A].Y'): {'[A].Z'}, ('[A].Y', '[A].X'): {'[A].Z'}}
        expectedDSepCount = 9
        self.assertRCDOutputEqual(expectedDeps, expectedSepset, expectedDSepCount, rcd.undirectedDependencies,
                                  rcd.sepsets, mockOracle)


    def testPhaseIBiggerConditionalSets(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addAttribute('A', 'Y')
        schema.addAttribute('A', 'Z')
        schema.addAttribute('A', 'W')
        model = Model(schema, ['[A].X -> [A].Y', '[A].X -> [A].Z', '[A].Y -> [A].W', '[A].Z -> [A].W'])
        rcd = RCD(schema, Oracle(model), 0)
        rcd.identifyUndirectedDependencies()
        expectedDeps = ['[A].X -> [A].Y', '[A].Y -> [A].X', '[A].X -> [A].Z', '[A].Z -> [A].X',
                        '[A].W -> [A].Y', '[A].Y -> [A].W', '[A].W -> [A].Z', '[A].Z -> [A].W']
        expectedSepset = {('[A].X', '[A].W'): {'[A].Y', '[A].Z'}, ('[A].W', '[A].X'): {'[A].Y', '[A].Z'},
                          ('[A].Y', '[A].Z'): {'[A].X'}, ('[A].Z', '[A].Y'): {'[A].X'}}
        self.assertRCDOutputEqual(expectedDeps, expectedSepset, None, rcd.undirectedDependencies, rcd.sepsets, None)


    def testIdentifyUndirectedDependenciesMultipleEntitiesOneToOne(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addEntity('B')
        schema.addAttribute('B', 'Y')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        model = Model(schema, [])
        mockOracle = MagicMock(wraps=Oracle(model, 2))
        mockModelProperty = PropertyMock()
        type(mockOracle).model = mockModelProperty
        rcd = RCD(schema, mockOracle, 2)
        rcd.identifyUndirectedDependencies()
        self.assertEqual(0, mockModelProperty.call_count)   # forces us not to cheat by simply returning the model
        self.assertRCDOutputEqual([], {}, 1, rcd.undirectedDependencies, rcd.sepsets, mockOracle)

        model = Model(schema, ['[B, AB, A].X -> [B].Y'])
        mockOracle = MagicMock(wraps=Oracle(model, 2))
        mockModelProperty = PropertyMock()
        type(mockOracle).model = mockModelProperty
        rcd = RCD(schema, mockOracle, 2)
        rcd.identifyUndirectedDependencies()
        self.assertEqual(0, mockModelProperty.call_count)
        self.assertRCDOutputEqual(['[B, AB, A].X -> [B].Y', '[A, AB, B].Y -> [A].X'], {}, 2,
                                  rcd.undirectedDependencies, rcd.sepsets, mockOracle)

        # three one-to-one entities in a chain to force longer hop threshold
        schema.addEntity('C')
        schema.addAttribute('C', 'Z')
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.ONE))
        model = Model(schema, ['[B, AB, A].X -> [B].Y', '[C, BC, B].Y -> [C].Z'])
        mockOracle = MagicMock(wraps=Oracle(model, 4))
        mockModelProperty = PropertyMock()
        type(mockOracle).model = mockModelProperty
        rcd = RCD(schema, mockOracle, 4)
        rcd.potentialDependencySorter = lambda l: sorted(l)
        rcd.identifyUndirectedDependencies()
        self.assertEqual(0, mockModelProperty.call_count)
        self.assertRCDOutputEqual(['[B, AB, A].X -> [B].Y', '[A, AB, B].Y -> [A].X', '[C, BC, B].Y -> [C].Z',
                                   '[B, BC, C].Z -> [B].Y'], {}, 10, rcd.undirectedDependencies, rcd.sepsets, mockOracle)


    def testMissedDependencyWithShortHopThreshold(self):
        # Dependency in model with longer path than given hop threshold.
        # RCD shouldn't discover (forces hop threshold as an argument)
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addEntity('B')
        schema.addAttribute('B', 'Y')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        schema.addEntity('C')
        schema.addAttribute('C', 'Z')
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.ONE))

        model = Model(schema, ['[A, AB, B, BC, C].Z -> [A].X'])
        mockOracle = MagicMock(wraps=Oracle(model, 4))
        rcd = RCD(schema, mockOracle, 2)
        rcd.identifyUndirectedDependencies()
        self.assertRCDOutputEqual([], {}, 2, rcd.undirectedDependencies, rcd.sepsets, mockOracle)


    def testPropagateOrientationCD(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addAttribute('A', 'Z')
        schema.addEntity('B')
        schema.addAttribute('B', 'Y')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))

        modelDeps = ['[A].X -> [A].Z', '[A, AB, B].Y -> [A].Z']
        model = Model(schema, modelDeps)
        mockOracle = MagicMock(wraps=Oracle(model, 2))
        rcd = RCD(schema, mockOracle, 2)
        rcd.identifyUndirectedDependencies()
        rcd.orientDependencies()
        self.assertRCDOutputEqual(modelDeps, None, None, rcd.orientedDependencies, None, None)
        self.assertEqual(2, rcd.edgeOrientationRuleFrequency['CD'])

        # do one for 3 usages (don't doubly orient the same edge)
        schema.addAttribute('B', 'W')
        modelDeps = ['[A].X -> [A].Z', '[A, AB, B].Y -> [A].Z', '[A, AB, B].W -> [A].Z']
        model = Model(schema, modelDeps)
        mockOracle = MagicMock(wraps=Oracle(model, 2))
        rcd = RCD(schema, mockOracle, 2)
        rcd.identifyUndirectedDependencies()
        rcd.orientDependencies()
        self.assertRCDOutputEqual(modelDeps, None, None, rcd.orientedDependencies, None, None)
        self.assertEqual(3, rcd.edgeOrientationRuleFrequency['CD'])


    def testPropagateOrientationKNC(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addAttribute('A', 'Z')
        schema.addEntity('B')
        schema.addAttribute('B', 'Y')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))

        modelDeps = ['[A].X -> [A].Z', '[B, AB, A].Z -> [B].Y']
        model = Model(schema, modelDeps)
        mockOracle = MagicMock(wraps=Oracle(model, 2))
        rcd = RCD(schema, mockOracle, 2)
        rcd.setUndirectedDependencies(['[A].X -> [A].Z', '[B, AB, A].Z -> [B].Y', '[A, AB, B].Y -> [A].Z'])
        rcd.setSepsets({})
        rcd.orientDependencies()
        self.assertRCDOutputEqual(modelDeps, None, None, rcd.orientedDependencies, None, None)
        self.assertEqual(1, rcd.edgeOrientationRuleFrequency['KNC'])

        modelDeps = ['[A].Z -> [A].X', '[A, AB, B].Y -> [A].Z']
        model = Model(schema, modelDeps)
        mockOracle = MagicMock(wraps=Oracle(model, 2))
        rcd = RCD(schema, mockOracle, 2)
        rcd.setUndirectedDependencies(['[A].X -> [A].Z', '[A].Z -> [A].X', '[A, AB, B].Y -> [A].Z'])
        rcd.setSepsets({})
        rcd.orientDependencies()
        self.assertRCDOutputEqual(modelDeps, None, None, rcd.orientedDependencies, None, None)
        self.assertEqual(1, rcd.edgeOrientationRuleFrequency['KNC'])


    def testPropagateOrientationCA(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addAttribute('A', 'Z')
        schema.addEntity('B')
        schema.addAttribute('B', 'Y')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))

        modelDeps = ['[A].X -> [A].Z', '[B, AB, A].Z -> [B].Y', '[B, AB, A].X -> [B].Y']
        model = Model(schema, modelDeps)
        mockOracle = MagicMock(wraps=Oracle(model, 2))
        rcd = RCD(schema, mockOracle, 2)
        rcd.setUndirectedDependencies(['[A].X -> [A].Z', '[B, AB, A].Z -> [B].Y',
                                       '[B, AB, A].X -> [B].Y', '[A, AB, B].Y -> [A].X'])
        rcd.setSepsets({})
        rcd.orientDependencies()
        self.assertRCDOutputEqual(modelDeps, None, None, rcd.orientedDependencies, None, None)
        self.assertEqual(1, rcd.edgeOrientationRuleFrequency['CA'])

        modelDeps = ['[A].Z -> [A].X', '[A, AB, B].Y -> [A].Z', '[A, AB, B].Y -> [A].X']
        model = Model(schema, modelDeps)
        mockOracle = MagicMock(wraps=Oracle(model, 2))
        rcd = RCD(schema, mockOracle, 2)
        rcd.setUndirectedDependencies(['[A].Z -> [A].X', '[A, AB, B].Y -> [A].Z',
                                       '[A, AB, B].Y -> [A].X', '[B, AB, A].X -> [B].Y'])
        rcd.setSepsets({})
        rcd.orientDependencies()
        self.assertRCDOutputEqual(modelDeps, None, None, rcd.orientedDependencies, None, None)
        self.assertEqual(1, rcd.edgeOrientationRuleFrequency['CA'])


    def testPropagateOrientationMR3(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addAttribute('A', 'Z')
        schema.addEntity('B')
        schema.addAttribute('B', 'Y')
        schema.addAttribute('B', 'W')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))

        modelDeps = ['[A].X -> [A].Z', '[B, AB, A].Z -> [B].Y', '[B, AB, A].X -> [B].Y',
                        '[B, AB, A].X -> [B].W', '[B].W -> [B].Y']
        model = Model(schema, modelDeps)
        mockOracle = MagicMock(wraps=Oracle(model, 2))
        rcd = RCD(schema, mockOracle, 2)
        undirectedDependencies = ['[B, AB, A].X -> [B].Y', '[A, AB, B].Y -> [A].X', '[A].X -> [A].Z',
                                  '[A].Z -> [A].X', '[B, AB, A].X -> [B].W', '[A, AB, B].W -> [A].X',
                                  '[B, AB, A].Z -> [B].Y', '[B].W -> [B].Y']
        rcd.setUndirectedDependencies(undirectedDependencies)
        rcd.setSepsets({})
        rcd.orientDependencies()
        self.assertEqual(3, len(rcd.orientedDependencies))
        self.assertEqual(1, rcd.edgeOrientationRuleFrequency['MR3'])


    def testFindNewSepsetInCD(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addEntity('B')
        schema.addAttribute('B', 'Y')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        schema.addEntity('C')
        schema.addAttribute('C', 'Z')
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.ONE))

        model = Model(schema, ['[B, AB, A].X -> [B].Y', '[B, BC, C].Z -> [B].Y'])
        mockOracle = MagicMock(wraps=Oracle(model, 4))
        rcd = RCD(schema, mockOracle, 2)
        rcd.identifyUndirectedDependencies()
        expectedDeps = ['[B, AB, A].X -> [B].Y', '[B, BC, C].Z -> [B].Y',
                        '[A, AB, B].Y -> [A].X', '[C, BC, B].Y -> [C].Z']
        self.assertRCDOutputEqual(expectedDeps, {}, 6, rcd.undirectedDependencies, rcd.sepsets, mockOracle)

        od = OrderedDict()
        od['A'] = rcd.perspectiveToAgg['A']
        od['C'] = rcd.perspectiveToAgg['C']
        od['B'] = rcd.perspectiveToAgg['B']
        rcd.perspectiveToAgg = od

        rcd.orientDependencies()
        expectedDeps = ['[B, AB, A].X -> [B].Y', '[B, BC, C].Z -> [B].Y']
        expectedSepsets = {('[A, AB, B, BC, C].Z', '[A].X'): set(), ('[A].X', '[A, AB, B, BC, C].Z'): set()}
        # Should run one extra marginal test in Phase II to find sepset that's longer than dependency hop threshold
        self.assertRCDOutputEqual(expectedDeps, expectedSepsets, 7, rcd.orientedDependencies, rcd.sepsets, mockOracle)

        # rerun with B perspective first, should skip and use A since no "canonical" dependencies
        mockOracle = MagicMock(wraps=Oracle(model, 4))
        rcd = RCD(schema, mockOracle, 2)
        rcd.identifyUndirectedDependencies()
        od = OrderedDict()
        od['B'] = rcd.perspectiveToAgg['B']
        od['A'] = rcd.perspectiveToAgg['A']
        od['C'] = rcd.perspectiveToAgg['C']
        rcd.perspectiveToAgg = od
        rcd.orientDependencies()
        expectedDeps = ['[B, AB, A].X -> [B].Y', '[B, BC, C].Z -> [B].Y']
        expectedSepsets = {('[A, AB, B, BC, C].Z', '[A].X'): set(), ('[A].X', '[A, AB, B, BC, C].Z'): set()}
        self.assertRCDOutputEqual(expectedDeps, expectedSepsets, 7, rcd.orientedDependencies, rcd.sepsets, mockOracle)

        # case where it finds a sepset at depth>0 and can orient as collider
        schema.addAttribute('B', 'W')
        modelDeps = ['[B, AB, A].X -> [B].Y', '[B, BC, C].Z -> [B].Y',
                     '[A, AB, B].W -> [A].X', '[C, BC, B].W -> [C].Z']
        model = Model(schema, modelDeps)
        mockOracle = MagicMock(wraps=Oracle(model, 4))
        rcd = RCD(schema, mockOracle, 2)
        rcd.potentialDependencySorter = lambda l : sorted(l)
        rcd.generateSepsetCombinations = lambda l, n: (v for v in sorted([sorted(sepset) for sepset in itertools.combinations(l, n)]))
        rcd.identifyUndirectedDependencies()
        phaseINumTests = 27
        self.assertEqual(phaseINumTests, mockOracle.isConditionallyIndependent.call_count)
        od = OrderedDict()
        od['C'] = rcd.perspectiveToAgg['C']
        od['A'] = rcd.perspectiveToAgg['A']
        od['B'] = rcd.perspectiveToAgg['B']
        rcd.perspectiveToAgg = od
        rcd.orientDependencies()
        expectedDeps = ['[B, AB, A].X -> [B].Y', '[B, BC, C].Z -> [B].Y']
        expectedSepsets = {('[C, BC, B, AB, A].X', '[C].Z'): {'[C, BC, B].W'},
                           ('[C].Z', '[C, BC, B, AB, A].X'): {'[C, BC, B].W'},
                           ('[B].W', '[B].Y'): {'[B, BC, C].Z', '[B, AB, A].X'},
                           ('[B].Y', '[B].W'): {'[B, BC, C].Z', '[B, AB, A].X'},
                           ('[A, AB, B, BC, C].Z', '[A].X'): {'[A, AB, B].W'},
                           ('[A].X', '[A, AB, B, BC, C].Z'): {'[A, AB, B].W'}}
        phaseIINumTests = 4
        self.assertRCDOutputEqual(expectedDeps, expectedSepsets, phaseINumTests+phaseIINumTests,
                                  rcd.orientedDependencies, rcd.sepsets, mockOracle)

        # case where it finds a sepset at depth>0 and can't orient as collider
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addEntity('B')
        schema.addAttribute('B', 'Y')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        schema.addEntity('C')
        schema.addAttribute('C', 'Z')
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.ONE))

        model = Model(schema, ['[A, AB, B].Y -> [A].X', '[C, BC, B].Y -> [C].Z'])
        mockOracle = MagicMock(wraps=Oracle(model, 4))
        rcd = RCD(schema, mockOracle, 2)
        rcd.potentialDependencySorter = lambda l : sorted(l)
        rcd.generateSepsetCombinations = lambda l, n: (v for v in sorted([sorted(sepset) for sepset in itertools.combinations(l, n)]))
        rcd.identifyUndirectedDependencies()
        expectedDeps = ['[B, AB, A].X -> [B].Y', '[B, BC, C].Z -> [B].Y',
                        '[A, AB, B].Y -> [A].X', '[C, BC, B].Y -> [C].Z']
        phaseINumTests = 6
        self.assertRCDOutputEqual(expectedDeps, {}, phaseINumTests, rcd.undirectedDependencies, rcd.sepsets, mockOracle)

        od = OrderedDict()
        od['C'] = rcd.perspectiveToAgg['C']
        od['B'] = rcd.perspectiveToAgg['B']
        od['A'] = rcd.perspectiveToAgg['A']
        rcd.perspectiveToAgg = od

        rcd.orientDependencies()
        expectedSepsets = {('[C, BC, B, AB, A].X', '[C].Z'): {'[C, BC, B].Y'},
                           ('[C].Z', '[C, BC, B, AB, A].X'): {'[C, BC, B].Y'},
                           ('[A, AB, B, BC, C].Z', '[A].X'): {'[A, AB, B].Y'},
                           ('[A].X', '[A, AB, B, BC, C].Z'): {'[A, AB, B].Y'}}
        # checks in _both_ C and A perspectives because it doesn't orient, and it remains an unshielded triple for both
        phaseIINumTests = 4
        self.assertRCDOutputEqual(expectedDeps, expectedSepsets, phaseINumTests+phaseIINumTests,
                                  rcd.undirectedDependencies, rcd.sepsets, mockOracle)


    def testEnforceTwiceHopThresholdInAggs(self):
        # enforce number of nodes in underlying AGGs is exactly due to using twice the supplied hop threshold
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addEntity('B')
        schema.addAttribute('B', 'Y')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        schema.addEntity('C')
        schema.addAttribute('C', 'Z')
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.ONE))

        model = Model(schema, [])
        rcd = RCD(schema, Oracle(model, 0), 0)
        rcd.identifyUndirectedDependencies()
        self.assertEqual(1, len(rcd.perspectiveToAgg['A'].nodes()))
        self.assertEqual(1, len(rcd.perspectiveToAgg['B'].nodes()))
        self.assertEqual(1, len(rcd.perspectiveToAgg['C'].nodes()))
        self.assertEqual(0, len(rcd.perspectiveToAgg['AB'].nodes()))
        self.assertEqual(0, len(rcd.perspectiveToAgg['BC'].nodes()))

        rcd = RCD(schema, Oracle(model, 0), 1)
        rcd.identifyUndirectedDependencies()
        self.assertEqual(2, len(rcd.perspectiveToAgg['A'].nodes()))
        self.assertEqual(3, len(rcd.perspectiveToAgg['B'].nodes()))
        self.assertEqual(2, len(rcd.perspectiveToAgg['C'].nodes()))
        self.assertEqual(2, len(rcd.perspectiveToAgg['AB'].nodes()))
        self.assertEqual(2, len(rcd.perspectiveToAgg['BC'].nodes()))

        rcd = RCD(schema, Oracle(model, 2), 2)
        rcd.identifyUndirectedDependencies()
        self.assertEqual(3, len(rcd.perspectiveToAgg['A'].nodes()))
        self.assertEqual(3, len(rcd.perspectiveToAgg['B'].nodes()))
        self.assertEqual(3, len(rcd.perspectiveToAgg['C'].nodes()))
        self.assertEqual(3, len(rcd.perspectiveToAgg['AB'].nodes()))
        self.assertEqual(3, len(rcd.perspectiveToAgg['BC'].nodes()))


    def testPhaseIEdgeRemovalPropagation(self):
        # make sure after Phase I all AGGs have the right edges -- tests the removal propagation
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addEntity('B')
        schema.addAttribute('B', 'Y')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        schema.addEntity('C')
        schema.addAttribute('C', 'Z')
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.ONE))

        model = Model(schema, ['[B, AB, A].X -> [B].Y'])
        rcd = RCD(schema, Oracle(model, 8), 4)
        rcd.identifyUndirectedDependencies()
        self.assertEqual(2, len(rcd.perspectiveToAgg['A'].edges()))
        self.assertEqual(2, len(rcd.perspectiveToAgg['B'].edges()))
        self.assertEqual(2, len(rcd.perspectiveToAgg['C'].edges()))
        self.assertEqual(2, len(rcd.perspectiveToAgg['AB'].edges()))
        self.assertEqual(2, len(rcd.perspectiveToAgg['BC'].edges()))

        schema.addAttribute('A', 'W')
        schema.addAttribute('AB', 'XY')
        model = Model(schema, ['[A].W -> [A].X', '[AB, A].X -> [AB].XY'])
        rcd = RCD(schema, Oracle(model, 2), 1)
        rcd.identifyUndirectedDependencies()
        self.assertEqual(4, len(rcd.perspectiveToAgg['A'].edges()))
        self.assertEqual(4, len(rcd.perspectiveToAgg['B'].edges()))
        self.assertEqual(0, len(rcd.perspectiveToAgg['C'].edges()))
        self.assertEqual(4, len(rcd.perspectiveToAgg['AB'].edges()))
        self.assertEqual(0, len(rcd.perspectiveToAgg['BC'].edges()))

        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addAttribute('B', 'Y1')
        schema.addAttribute('B', 'Y2')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.MANY))

        model = Model(schema, ['[B, AB, A, AB, B].Y1 -> [B].Y2'])
        rcd = RCD(schema, Oracle(model, 8), 4)
        rcd.identifyUndirectedDependencies()
        sdw = SchemaDependencyWrapper(schema, [ParserUtil.parseRelDep(relDepStr) for relDepStr in
                                               ['[B, AB, A, AB, B].Y1 -> [B].Y2', '[B, AB, A, AB, B].Y2 -> [B].Y1']])
        self.assertEqual(set(AbstractGroundGraph(sdw, 'A', 8).edges()), set(rcd.perspectiveToAgg['A'].edges()))
        self.assertEqual(set(AbstractGroundGraph(sdw, 'B', 8).edges()), set(rcd.perspectiveToAgg['B'].edges()))
        self.assertEqual(set(AbstractGroundGraph(sdw, 'AB', 8).edges()), set(rcd.perspectiveToAgg['AB'].edges()))

        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.MANY))
        schema.addAttribute('AB', 'XY1')
        schema.addAttribute('AB', 'XY2')
        schema.addAttribute('AB', 'XY3')

        modelDeps = ['[AB, A, AB, B, AB].XY1 -> [AB].XY2',
                     '[AB, A, AB, B, AB].XY2 -> [AB].XY3']
        model = Model(schema, modelDeps)
        rcd = RCD(schema, Oracle(model, 8), 4)
        rcd.identifyUndirectedDependencies()
        sdw = SchemaDependencyWrapper(schema, [ParserUtil.parseRelDep(relDepStr) for relDepStr in
                                               ['[AB, A, AB, B, AB].XY1 -> [AB].XY2',
                                                '[AB, A, AB, B, AB].XY2 -> [AB].XY3',
                                                '[AB, B, AB, A, AB].XY2 -> [AB].XY1',
                                                 '[AB, B, AB, A, AB].XY3 -> [AB].XY2']])
        self.assertEqual(set(AbstractGroundGraph(sdw, 'AB', 8).edges()), set(rcd.perspectiveToAgg['AB'].edges()))


    def testDepthAsArgument(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addEntity('B')
        schema.addAttribute('B', 'Y')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        schema.addEntity('C')
        schema.addAttribute('C', 'Z')
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.ONE))

        model = Model(schema, ['[B, AB, A].X -> [B].Y', '[C, BC, B].Y -> [C].Z'])
        rcd = RCD(schema, Oracle(model, 8), hopThreshold=4, depth=0)
        rcd.identifyUndirectedDependencies()
        expectedDepStrs = ['[B, AB, A].X -> [B].Y', '[A, AB, B].Y -> [A].X',
                        '[C, BC, B].Y -> [C].Z', '[B, BC, C].Z -> [B].Y',
                        '[C, BC, B, AB, A].X -> [C].Z', '[A, AB, B, BC, C].Z -> [A].X']
        TestUtil.assertUnorderedListEqual(self, [ParserUtil.parseRelDep(relDepStr) for relDepStr in expectedDepStrs],
                                          rcd.undirectedDependencies)

        model = Model(schema, ['[B, AB, A].X -> [B].Y', '[B, BC, C].Z -> [B].Y'])
        rcd = RCD(schema, Oracle(model, 8), hopThreshold=4, depth=0)
        rcd.identifyUndirectedDependencies()
        expectedDepStrs = ['[B, AB, A].X -> [B].Y', '[A, AB, B].Y -> [A].X',
                           '[C, BC, B].Y -> [C].Z', '[B, BC, C].Z -> [B].Y']
        TestUtil.assertUnorderedListEqual(self, [ParserUtil.parseRelDep(relDepStr) for relDepStr in expectedDepStrs],
                                          rcd.undirectedDependencies)


    def testDepthDefaultExitsEarly(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addEntity('B')
        schema.addAttribute('B', 'Y')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        schema.addEntity('C')
        schema.addAttribute('C', 'Z')
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.ONE))
        model = Model(schema, ['[B, AB, A].X -> [B].Y', '[C, BC, B].Y -> [C].Z'])

        rcd = RCD(schema, Oracle(model, 8), 4, depth=3)
        rcd.identifyUndirectedDependencies()
        self.assertEqual(2, rcd.maxDepthReached)

        # case where default needs to be n-2 (default should be None already, but this forces it)
        mockOracle = MagicMock(wraps=Oracle(model, 8))
        mockModelProperty = PropertyMock()
        type(mockOracle).model = mockModelProperty
        rcd = RCD(schema, mockOracle, 4, depth=None)
        rcd.potentialDependencySorter = lambda l: sorted(l)
        rcd.identifyUndirectedDependencies()
        self.assertEqual(0, mockModelProperty.call_count)
        self.assertRCDOutputEqual(['[B, AB, A].X -> [B].Y', '[A, AB, B].Y -> [A].X', '[C, BC, B].Y -> [C].Z',
                                   '[B, BC, C].Z -> [B].Y'], {}, 10, rcd.undirectedDependencies, rcd.sepsets, mockOracle)


    def testDepthAsArgumentForFindSepset(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addEntity('B')
        schema.addAttribute('B', 'Y')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        schema.addEntity('C')
        schema.addAttribute('C', 'Z')
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.ONE))
        schema.addAttribute('B', 'W')

        modelDeps = ['[B, AB, A].X -> [B].Y', '[B, BC, C].Z -> [B].Y',
                     '[A, AB, B].W -> [A].X', '[C, BC, B].W -> [C].Z']
        model = Model(schema, modelDeps)
        mockOracle = MagicMock(wraps=Oracle(model, 4))
        rcd = RCD(schema, mockOracle, 2, depth=0) # passing in depth=0, so it should NOT orient in phase II
        rcd.identifyUndirectedDependencies()
        rcd.orientDependencies()
        expectedDepStrs = ['[B, AB, A].X -> [B].Y', '[A, AB, B].Y -> [A].X',
                           '[B, BC, C].Z -> [B].Y', '[C, BC, B].Y -> [C].Z',
                            '[A, AB, B].W -> [A].X', '[C, BC, B].W -> [C].Z',
                            '[B, AB, A].X -> [B].W', '[B, BC, C].Z -> [B].W',
                            '[B].W -> [B].Y', '[B].Y -> [B].W']
        expectedDeps = [ParserUtil.parseRelDep(expectedDepStr) for expectedDepStr in expectedDepStrs]
        TestUtil.assertUnorderedListEqual(self, expectedDeps, list(rcd.undirectedDependencies))


    def testDepthErrorConditions(self):
        schema = Schema()
        model = Model(schema, [])
        oracle = Oracle(model)
        TestUtil.assertRaisesMessage(self, Exception, "Depth must be a non-negative integer or None: found depth",
                                     RCD, schema, oracle, 0, 'depth')

        TestUtil.assertRaisesMessage(self, Exception, "Depth must be a non-negative integer or None: found -1",
                                     RCD, schema, oracle, 0, -1)

        TestUtil.assertRaisesMessage(self, Exception, "Depth must be a non-negative integer or None: found 1.5",
                                     RCD, schema, oracle, 0, 1.5)


    def testIdentifyUndirectedDependenciesOneToMany(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addEntity('B')
        schema.addAttribute('B', 'Y')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.ONE))
        model = Model(schema, [])
        mockOracle = MagicMock(wraps=Oracle(model, 2))
        mockModelProperty = PropertyMock()
        type(mockOracle).model = mockModelProperty
        rcd = RCD(schema, mockOracle, 2)
        rcd.identifyUndirectedDependencies()
        self.assertEqual(0, mockModelProperty.call_count)   # forces us not to cheat by simply returning the model
        self.assertRCDOutputEqual([], {}, 1, rcd.undirectedDependencies, rcd.sepsets, mockOracle)

        model = Model(schema, ['[B, AB, A].X -> [B].Y'])
        mockOracle = MagicMock(wraps=Oracle(model, 4))
        mockModelProperty = PropertyMock()
        type(mockOracle).model = mockModelProperty
        rcd = RCD(schema, mockOracle, 2)
        rcd.potentialDependencySorter = lambda l: sorted(l)
        rcd.identifyUndirectedDependencies()
        self.assertEqual(0, mockModelProperty.call_count)
        self.assertRCDOutputEqual(['[B, AB, A].X -> [B].Y', '[A, AB, B].Y -> [A].X'], {}, 2,
                                  rcd.undirectedDependencies, rcd.sepsets, mockOracle)

        schema.addAttribute('B', 'Z')
        model = Model(schema, ['[B, AB, A].X -> [B].Y', '[B].Y -> [B].Z'])
        mockOracle = MagicMock(wraps=Oracle(model, 4))
        mockModelProperty = PropertyMock()
        type(mockOracle).model = mockModelProperty
        rcd = RCD(schema, mockOracle, 2)
        rcd.potentialDependencySorter = lambda l: sorted(l)
        rcd.identifyUndirectedDependencies()
        self.assertEqual(0, mockModelProperty.call_count)
        self.assertRCDOutputEqual(['[B, AB, A].X -> [B].Y', '[A, AB, B].Y -> [A].X', '[B].Y -> [B].Z', '[B].Z -> [B].Y'],
                                    {}, 10, rcd.undirectedDependencies, rcd.sepsets, mockOracle)

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
        mockOracle = MagicMock(wraps=Oracle(model, 8))
        mockModelProperty = PropertyMock()
        type(mockOracle).model = mockModelProperty
        rcd = RCD(schema, mockOracle, 4)
        rcd.potentialDependencySorter = lambda l: sorted(l)
        rcd.identifyUndirectedDependencies()
        self.assertEqual(0, mockModelProperty.call_count)
        self.assertRCDOutputEqual(['[B, AB, A].X -> [B].Y', '[A, AB, B].Y -> [A].X',
                                   '[C, BC, B].Y -> [C].Z', '[B, BC, C].Z -> [B].Y'],
                                    {}, 12, rcd.undirectedDependencies, rcd.sepsets, mockOracle)

        mockOracle = MagicMock(wraps=Oracle(model, 8))
        rcd = RCD(schema, mockOracle, 4)
        rcd.potentialDependencySorter = lambda l: sorted(l, reverse=True)
        rcd.identifyUndirectedDependencies()
        self.assertEqual(0, mockModelProperty.call_count)
        self.assertRCDOutputEqual(['[B, AB, A].X -> [B].Y', '[A, AB, B].Y -> [A].X',
                                   '[C, BC, B].Y -> [C].Z', '[B, BC, C].Z -> [B].Y'],
                                    {}, 9, rcd.undirectedDependencies, rcd.sepsets, mockOracle)

        # test to check that intersection neighbors for conditioning are <= hop threshold
        model = Model(schema, ['[B, AB, A].X -> [B].Y', '[C, BC, B, AB, A].X -> [C].Z'])
        mockOracle = MagicMock(wraps=Oracle(model, 8))
        mockModelProperty = PropertyMock()
        type(mockOracle).model = mockModelProperty
        rcd = RCD(schema, mockOracle, 4)
        rcd.potentialDependencySorter = lambda l: sorted(l)
        rcd.identifyUndirectedDependencies()
        self.assertEqual(10, mockOracle.isConditionallyIndependent.call_count)


    def testOrientDependenciesWithRBO(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addEntity('B')
        schema.addAttribute('B', 'Y')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.MANY))
        schema.addEntity('C')
        schema.addAttribute('C', 'Z')
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.MANY))

        modelDeps = ['[B, AB, A].X -> [B].Y', '[C, BC, B].Y -> [C].Z']
        model = Model(schema, modelDeps)
        mockOracle = MagicMock(wraps=Oracle(model, 8))
        mockModelProperty = PropertyMock()
        type(mockOracle).model = mockModelProperty
        rcd = RCD(schema, mockOracle, 4)
        rcd.potentialDependencySorter = lambda l: sorted(l)
        rcd.generateSepsetCombinations = lambda l, n: (v for v in sorted([sorted(sepset) for sepset in itertools.combinations(l, n)]))
        rcd.identifyUndirectedDependencies()
        od = OrderedDict()
        od['A'] = rcd.perspectiveToAgg['A']
        od['B'] = rcd.perspectiveToAgg['B']
        od['C'] = rcd.perspectiveToAgg['C']
        rcd.perspectiveToAgg = od
        rcd.orientDependencies()
        expectedDeps = modelDeps
        expectedSepsets = {('[A].X', '[A, AB, B, AB, A].X'): set(), ('[A, AB, B, AB, A].X', '[A].X'): set(),
                           ('[B].Y', '[B, BC, C, BC, B].Y'): {'[B, AB, A].X'},
                           ('[B, BC, C, BC, B].Y', '[B].Y'): {'[B, AB, A].X'},
                           ('[C].Z', '[C, BC, B, AB, A].X'): {'[C, BC, B].Y'},
                           ('[C, BC, B, AB, A].X', '[C].Z'): {'[C, BC, B].Y'}}
        numPhaseICITests = 12
        numPhaseIICITests = 3 # with CI caching, otherwise 5
        self.assertRCDOutputEqual(expectedDeps, expectedSepsets, numPhaseICITests+numPhaseIICITests,
                                  rcd.orientedDependencies, rcd.sepsets, mockOracle)
        self.assertEqual(0, rcd.edgeOrientationRuleFrequency['CD'])
        self.assertEqual(0, rcd.edgeOrientationRuleFrequency['KNC'])
        self.assertEqual(0, rcd.edgeOrientationRuleFrequency['CA'])
        self.assertEqual(0, rcd.edgeOrientationRuleFrequency['MR3'])
        self.assertEqual(2, rcd.edgeOrientationRuleFrequency['RBO'])

        # RBO common cause
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addEntity('B')
        schema.addAttribute('B', 'Y')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.ONE))

        modelDeps = ['[B, AB, A].X -> [B].Y']
        model = Model(schema, modelDeps)
        mockOracle = MagicMock(wraps=Oracle(model, 4))
        mockModelProperty = PropertyMock()
        type(mockOracle).model = mockModelProperty
        rcd = RCD(schema, mockOracle, 2)
        rcd.potentialDependencySorter = lambda l: sorted(l)
        rcd.generateSepsetCombinations = lambda l, n: (v for v in sorted([sorted(sepset) for sepset in itertools.combinations(l, n)]))
        rcd.identifyUndirectedDependencies()
        od = OrderedDict()
        od['B'] = rcd.perspectiveToAgg['B']
        od['A'] = rcd.perspectiveToAgg['A']
        rcd.perspectiveToAgg = od
        rcd.orientDependencies()
        expectedDeps = modelDeps
        expectedSepsets = {('[B].Y', '[B, AB, A, AB, B].Y'): {'[B, AB, A].X'},
                           ('[B, AB, A, AB, B].Y', '[B].Y'): {'[B, AB, A].X'}}
        numPhaseICITests = 2
        numPhaseIICITests = 2
        self.assertRCDOutputEqual(expectedDeps, expectedSepsets, numPhaseICITests+numPhaseIICITests,
                                  rcd.orientedDependencies, rcd.sepsets, mockOracle)
        self.assertEqual(0, rcd.edgeOrientationRuleFrequency['CD'])
        self.assertEqual(0, rcd.edgeOrientationRuleFrequency['KNC'])
        self.assertEqual(0, rcd.edgeOrientationRuleFrequency['CA'])
        self.assertEqual(0, rcd.edgeOrientationRuleFrequency['MR3'])
        self.assertEqual(1, rcd.edgeOrientationRuleFrequency['RBO'])


    def testMultiplePathsRBO(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addEntity('B')
        schema.addAttribute('B', 'Y')
        schema.addRelationship('AB1', ('A', Schema.MANY), ('B', Schema.ONE))
        schema.addRelationship('AB2', ('A', Schema.MANY), ('B', Schema.ONE))

        modelDeps = ['[B, AB1, A].X -> [B].Y', '[B, AB2, A].X -> [B].Y']
        model = Model(schema, modelDeps)
        mockOracle = MagicMock(wraps=Oracle(model, 4))
        mockModelProperty = PropertyMock()
        type(mockOracle).model = mockModelProperty
        rcd = RCD(schema, mockOracle, 2)
        rcd.potentialDependencySorter = lambda l: sorted(l)
        rcd.generateSepsetCombinations = lambda l, n: (v for v in sorted([sorted(sepset) for sepset in itertools.combinations(l, n)]))
        rcd.identifyUndirectedDependencies()
        od = OrderedDict()
        od['B'] = rcd.perspectiveToAgg['B']
        od['A'] = rcd.perspectiveToAgg['A']
        rcd.perspectiveToAgg = od
        rcd.orientDependencies()
        numPhaseICITests = 8
        numPhaseIICITests = 4
        self.assertEqual(numPhaseICITests+numPhaseIICITests, mockOracle.isConditionallyIndependent.call_count)
        self.assertEqual(1, rcd.edgeOrientationRuleFrequency['RBO'])


    def testLargerCausalModel(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addAttribute('A', 'V')
        schema.addEntity('B')
        schema.addAttribute('B', 'Y')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.MANY))
        schema.addEntity('C')
        schema.addAttribute('C', 'Z')
        schema.addAttribute('C', 'W')
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.MANY))

        modelDeps = ['[B, AB, A].X -> [B].Y', '[C, BC, B].Y -> [C].Z', '[C].Z -> [C].W',
                     '[A].X -> [A].V', '[A, AB, B, BC, C].W -> [A].V']
        model = Model(schema, modelDeps)
        mockOracle = MagicMock(wraps=Oracle(model, 8))
        mockModelProperty = PropertyMock()
        type(mockOracle).model = mockModelProperty
        rcd = RCD(schema, mockOracle, 4)
        rcd.potentialDependencySorter = lambda l: sorted(l)
        rcd.generateSepsetCombinations = lambda l, n: (v for v in sorted([sorted(sepset) for sepset in itertools.combinations(l, n)]))
        rcd.identifyUndirectedDependencies()
        numPhaseICITests = 99
        self.assertEqual(numPhaseICITests, mockOracle.isConditionallyIndependent.call_count)
        od = OrderedDict()
        od['A'] = rcd.perspectiveToAgg['A']
        od['B'] = rcd.perspectiveToAgg['B']
        od['C'] = rcd.perspectiveToAgg['C']
        rcd.perspectiveToAgg = od
        rcd.orientDependencies()
        expectedDeps = modelDeps
        numPhaseIICITests = 5 # with CI caching, otherwise 23
        self.assertRCDOutputEqual(expectedDeps, None, numPhaseICITests+numPhaseIICITests,
                                  rcd.orientedDependencies, None, mockOracle)
        self.assertEqual(2, rcd.edgeOrientationRuleFrequency['CD'])
        self.assertEqual(1, rcd.edgeOrientationRuleFrequency['KNC'])
        self.assertEqual(0, rcd.edgeOrientationRuleFrequency['CA'])
        self.assertEqual(0, rcd.edgeOrientationRuleFrequency['MR3'])
        self.assertEqual(2, rcd.edgeOrientationRuleFrequency['RBO'])
        ciBreakDown, eoFrequencies = rcd.report()
        self.assertEqual({'Phase I': 99, 'depth 0': 22, 'depth 1': 63, 'depth 2': 14,
                          'Phase II': 5, 'total': 104}, ciBreakDown)
        self.assertEqual({'CD': 2, 'KNC': 1, 'CA': 0, 'MR3': 0, 'RBO': 2}, eoFrequencies)


    def testPhaseII(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        model = Model(schema, [])
        rcd = RCD(schema, Oracle(model), 0)
        rcd.identifyUndirectedDependencies()
        rcd.orientDependencies()
        self.assertRCDOutputEqual([], None, None, rcd.orientedDependencies, None, None)

        schema.addAttribute('A', 'Y')
        schema.addAttribute('A', 'Z')
        model = Model(schema, ['[A].X -> [A].Y'])
        rcd = RCD(schema, Oracle(model), 0)
        rcd.identifyUndirectedDependencies()
        rcd.orientDependencies()
        expectedDeps = ['[A].X -> [A].Y', '[A].Y -> [A].X']
        self.assertRCDOutputEqual(expectedDeps, None, None, rcd.undirectedDependencies, None, None)

        model = Model(schema, ['[A].X -> [A].Z', '[A].Y -> [A].Z'])
        rcd = RCD(schema, Oracle(model), 0)
        rcd.identifyUndirectedDependencies()
        rcd.orientDependencies()
        expectedEdges = ['[A].X -> [A].Z', '[A].Y -> [A].Z']
        self.assertRCDOutputEqual(expectedEdges, None, None, rcd.orientedDependencies, None, None)

        model = Model(schema, ['[A].X -> [A].Z', '[A].Z -> [A].Y'])
        rcd = RCD(schema, Oracle(model), 0)
        rcd.identifyUndirectedDependencies()
        rcd.orientDependencies()
        expectedEdges = ['[A].X -> [A].Z', '[A].Y -> [A].Z', '[A].Z -> [A].X', '[A].Z -> [A].Y']
        self.assertRCDOutputEqual(expectedEdges, None, None, rcd.undirectedDependencies, None, None)

        schema.addAttribute('A', 'W')
        model = Model(schema, ['[A].X -> [A].Z', '[A].Y -> [A].Z'])
        rcd = RCD(schema, Oracle(model), 0)
        rcd.identifyUndirectedDependencies()
        rcd.orientDependencies()
        expectedEdges = ['[A].X -> [A].Z', '[A].Y -> [A].Z']
        self.assertRCDOutputEqual(expectedEdges, None, None, rcd.orientedDependencies, None, None)

        schema.addAttribute('A', 'V')
        model = Model(schema, ['[A].X -> [A].Z', '[A].Y -> [A].Z', '[A].Y -> [A].W', '[A].V -> [A].W'])
        rcd = RCD(schema, Oracle(model), 0)
        rcd.identifyUndirectedDependencies()
        rcd.orientDependencies()
        expectedEdges = ['[A].X -> [A].Z', '[A].Y -> [A].Z', '[A].Y -> [A].W', '[A].V -> [A].W']
        self.assertRCDOutputEqual(expectedEdges, None, None, rcd.orientedDependencies, None, None)

        model = Model(schema, ['[A].X -> [A].Z', '[A].Y -> [A].Z', '[A].Z -> [A].W', '[A].W -> [A].V'])
        rcd = RCD(schema, Oracle(model), 0)
        rcd.identifyUndirectedDependencies()
        rcd.orientDependencies()
        expectedEdges = ['[A].X -> [A].Z', '[A].Y -> [A].Z', '[A].Z -> [A].W', '[A].W -> [A].V']
        self.assertRCDOutputEqual(expectedEdges, None, None, rcd.orientedDependencies, None, None)

        model = Model(schema, ['[A].X -> [A].Z', '[A].Y -> [A].Z', '[A].Z -> [A].W', '[A].Y -> [A].W'])
        rcd = RCD(schema, Oracle(model), 0)
        rcd.identifyUndirectedDependencies()
        rcd.orientDependencies()
        expectedEdges = ['[A].X -> [A].Z', '[A].Y -> [A].Z', '[A].Z -> [A].W', '[A].Y -> [A].W']
        self.assertRCDOutputEqual(expectedEdges, None, None, rcd.orientedDependencies, None, None)

        model = Model(schema, ['[A].X -> [A].Y', '[A].X -> [A].W', '[A].X -> [A].Z', '[A].Y -> [A].Z', '[A].W -> [A].Z'])
        rcd = RCD(schema, Oracle(model), 0)
        rcd.identifyUndirectedDependencies()
        rcd.orientDependencies()
        expectedEdges = ['[A].X -> [A].Z', '[A].Y -> [A].Z', '[A].W -> [A].Z']
        self.assertRCDOutputEqual(expectedEdges, None, None, rcd.orientedDependencies, None, None)

        model = Model(schema, ['[A].W -> [A].X', '[A].W -> [A].Z', '[A].W -> [A].Y', '[A].X -> [A].V', '[A].X -> [A].Z',
                               '[A].Y -> [A].V', '[A].Y -> [A].Z', '[A].Z -> [A].V'])
        rcd = RCD(schema, Oracle(model), 0)
        rcd.setUndirectedDependencies(['[A].X -> [A].Z', '[A].X -> [A].W', '[A].X -> [A].V', '[A].Y -> [A].Z',
                                       '[A].Y -> [A].W', '[A].Y -> [A].V', '[A].Z -> [A].X', '[A].Z -> [A].Y',
                                       '[A].Z -> [A].W', '[A].Z -> [A].V', '[A].W -> [A].X', '[A].W -> [A].Y',
                                       '[A].W -> [A].Z', '[A].V -> [A].X', '[A].V -> [A].Y', '[A].V -> [A].Z'])
        rcd.setSepsets({('[A].X', '[A].Y'): {'[A].W'}, ('[A].Y', '[A].X'): {'[A].W'},
                        ('[A].V', '[A].W'): {'[A].Y', '[A].X', '[A].Z'}, ('[A].W', '[A].V'): {'[A].Y', '[A].X', '[A].Z'}})

        rcd.orientDependencies()
        expectedEdges = ['[A].X -> [A].V', '[A].Y -> [A].V', '[A].Z -> [A].V', '[A].Y -> [A].Z',
                         '[A].W -> [A].Z', '[A].X -> [A].Z']
        self.assertRCDOutputEqual(expectedEdges, None, None, rcd.orientedDependencies, None, None)


    def testNoSkeletonBeforePhaseII(self):
        # must have an undirected skeleton and sepsets before running Phase II
        schema = Schema()
        model = Model(schema, [])
        rcd = RCD(schema, Oracle(model), 0)
        TestUtil.assertRaisesMessage(self, Exception, "No undirected dependencies found. Try running Phase I first.",
             rcd.orientDependencies)

        # what if we set the skeleton to None?
        rcd = RCD(schema, Oracle(model), 0)
        rcd.undirectedDependencies = None
        TestUtil.assertRaisesMessage(self, Exception, "No undirected dependencies found. Try running Phase I first.",
             rcd.orientDependencies)

        # what if we don't set the sepset?
        rcd = RCD(schema, Oracle(model), 0)
        rcd.setUndirectedDependencies([])
        TestUtil.assertRaisesMessage(self, Exception, "No sepsets found. Try running Phase I first.",
             rcd.orientDependencies)

        # what if we set the sepsets to None?
        rcd = RCD(schema, Oracle(model), 0)
        rcd.setUndirectedDependencies([])
        rcd.sepsets = None
        TestUtil.assertRaisesMessage(self, Exception, "No sepsets found. Try running Phase I first.",
             rcd.orientDependencies)


    def testSetUndirectedDependencies(self):
        schema = Schema()
        model = Model(schema, [])
        rcd = RCD(schema, Oracle(model), 0)
        rcd.setUndirectedDependencies([])
        self.assertEqual([], rcd.undirectedDependencies)

        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addAttribute('A', 'Y')
        model = Model(schema, [])
        rcd = RCD(schema, Oracle(model), 0)
        undirectedDependencies = ['[A].X -> [A].Y']
        rcd.setUndirectedDependencies(undirectedDependencies)
        self.assertEqual(undirectedDependencies, [str(dep) for dep in rcd.undirectedDependencies])

        TestUtil.assertRaisesMessage(self, Exception, "Undirected dependencies must be an iterable sequence of "
                                                      "parseable RelationalDependency strings: found None",
             rcd.setUndirectedDependencies, None)

        TestUtil.assertRaisesMessage(self, Exception, "Undirected dependencies must be an iterable sequence of "
                                                      "parseable RelationalDependency strings: found 5",
             rcd.setUndirectedDependencies, 5)

        # enforce that the undirected dependencies are checked for consistency against the schema
        # using ModelUtil.checkDependencyConsistency
        schema = Schema()
        model = Model(schema, [])
        rcd = RCD(schema, Oracle(model), 0)
        undirectedDependencies = []
        mockDepChecker = MagicMock(wraps=RelationalValidity.checkRelationalDependencyValidity)
        rcd.setUndirectedDependencies(undirectedDependencies, dependencyChecker=mockDepChecker)
        self.assertEqual(0, mockDepChecker.call_count)

        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addAttribute('A', 'Y')
        model = Model(schema, [])
        rcd = RCD(schema, Oracle(model), 0)
        undirectedDependencies = ['[A].X -> [A].Y']
        mockDepChecker = MagicMock(wraps=RelationalValidity.checkRelationalDependencyValidity)
        rcd.setUndirectedDependencies(undirectedDependencies, dependencyChecker=mockDepChecker)
        self.assertEqual(1, mockDepChecker.call_count)

        undirectedDependencies = ['[A].X -> [A].Y', '[A].Y -> [A].X']
        mockDepChecker = MagicMock(wraps=RelationalValidity.checkRelationalDependencyValidity)
        rcd.setUndirectedDependencies(undirectedDependencies, dependencyChecker=mockDepChecker)
        self.assertEqual(2, mockDepChecker.call_count)


    def testSetSepsets(self):
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
        rcd = RCD(schema, Oracle(model), 4)
        rcd.setSepsets({})
        self.assertEqual({}, rcd.sepsets)

        rcd.setSepsets({('[A].X', '[A, AB, B].Y'): {'[A, AB, B, BC, C].Z'}})
        self.assertEqual({(RelationalVariable(['A'], 'X'), RelationalVariable(['A', 'AB', 'B'], 'Y')):
                      {RelationalVariable(['A', 'AB', 'B', 'BC', 'C'], 'Z')}}, rcd.sepsets)

        TestUtil.assertRaisesMessage(self, Exception, "Sepsets must be a dictionary: found None",
             rcd.setSepsets, None)

        # enforce that the method is using the right validity checker
        mockRelVarSetChecker = MagicMock(wraps=RelationalValidity.checkValidityOfRelationalVariableSet)
        rcd = RCD(schema, Oracle(model), 4)
        rcd.setSepsets({('[A].X', '[A, AB, B].Y'): {'[A, AB, B, BC, C].Z'},
                        ('[B].Y', '[B, AB, A].X'): {'[B, BC, C].Z'}},
                       relationalVariableSetChecker=mockRelVarSetChecker)
        self.assertEqual(2, mockRelVarSetChecker.call_count)


    def testSetPhaseIPattern(self):
        # edges in undirected dependencies and sepsets should be useful for Phase II
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addAttribute('A', 'Y')
        schema.addAttribute('A', 'Z')
        model = Model(schema, ['[A].X -> [A].Z', '[A].Y -> [A].Z'])
        rcd = RCD(schema, Oracle(model), 0)
        undirectedDependencies = ['[A].X -> [A].Z', '[A].Y -> [A].Z', '[A].Z -> [A].X', '[A].Z -> [A].Y']
        rcd.setUndirectedDependencies(undirectedDependencies)
        rcd.setSepsets({('[A].X', '[A].Y'): set(), ('[A].Y', '[A].X'): set()})
        rcd.orientDependencies()
        self.assertRCDOutputEqual(['[A].X -> [A].Z', '[A].Y -> [A].Z'], None, None,
                  rcd.orientedDependencies, None, None)


    def testCacheCITests(self):
        # make sure we cache the results of CI tests (no repeated tests in findRecordAndReturnSepset)
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addAttribute('A', 'Y')
        schema.addAttribute('A', 'W')
        schema.addEntity('B')
        schema.addAttribute('B', 'Z')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        model = Model(schema, ['[A].X -> [A].Y', '[A, AB, B].Z -> [A].Y', '[A].W -> [A].X', '[B, AB, A].W -> [B].Z'])
        mockOracle = MagicMock(wraps=Oracle(model, 8))
        rcd = RCD(schema, mockOracle, 2)
        rcd.potentialDependencySorter = lambda l: sorted(l)
        rcd.generateSepsetCombinations = lambda l, n: (v for v in sorted([sorted(sepset) for sepset in itertools.combinations(l, n)]))
        rcd.identifyUndirectedDependencies()
        numPhaseICITests = 32
        self.assertEqual(numPhaseICITests, mockOracle.isConditionallyIndependent.call_count)
        od = OrderedDict()
        od['B'] = rcd.perspectiveToAgg['B']
        od['A'] = rcd.perspectiveToAgg['A']
        rcd.perspectiveToAgg = od
        rcd.orientDependencies()
        numPhaseIICITests = 1
        self.assertEqual(numPhaseICITests+numPhaseIICITests, mockOracle.isConditionallyIndependent.call_count)


    def testReportRCD(self):
        # tests for a "report" method that returns (1) the number of CI tests broken down by sepset size
        # and within phases and (2) the number of times each edge orientation rule was used to orient an edge
        # see also the end of testLargerCausalModel
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        model = Model(schema, [])
        rcd = RCD(schema, Oracle(model), 0)
        rcd.identifyUndirectedDependencies()
        ciBreakDown, eoFrequencies = rcd.report()
        self.assertEqual({'Phase I': 0, 'Phase II': 0, 'total': 0}, ciBreakDown)
        self.assertEqual({'CD': 0, 'KNC': 0, 'CA': 0, 'MR3': 0, 'RBO': 0}, eoFrequencies)

        schema.addAttribute('A', 'Y')
        model = Model(schema, [])
        rcd = RCD(schema, Oracle(model), 0)
        rcd.identifyUndirectedDependencies()
        ciBreakDown, eoFrequencies = rcd.report()
        self.assertEqual({'Phase I': 1, 'depth 0': 1, 'Phase II': 0, 'total': 1}, ciBreakDown)
        self.assertEqual({'CD': 0, 'KNC': 0, 'CA': 0, 'MR3': 0, 'RBO': 0}, eoFrequencies)

        schema.addAttribute('A', 'Z')
        schema.addAttribute('A', 'W')
        schema.addAttribute('A', 'V')
        model = Model(schema, ['[A].W -> [A].X', '[A].W -> [A].Z', '[A].W -> [A].Y', '[A].X -> [A].V', '[A].X -> [A].Z',
                               '[A].Y -> [A].V', '[A].Y -> [A].Z', '[A].Z -> [A].V'])
        rcd = RCD(schema, Oracle(model), 0)
        rcd.potentialDependencySorter = lambda l: sorted(l)
        rcd.generateSepsetCombinations = lambda l, n: (v for v in sorted([sorted(sepset) for sepset in itertools.combinations(l, n)]))
        rcd.identifyUndirectedDependencies()
        rcd.orientDependencies()
        ciBreakDown, eoFrequencies = rcd.report()
        self.assertEqual({'Phase I': 121, 'depth 0': 20, 'depth 1': 54, 'depth 2': 42, 'depth 3': 5,
                          'Phase II': 0, 'total': 121}, ciBreakDown)
        self.assertEqual({'CD': 4, 'KNC': 1, 'CA': 0, 'MR3': 1, 'RBO': 0}, eoFrequencies)


    def testRunRCD(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addAttribute('A', 'V')
        schema.addEntity('B')
        schema.addAttribute('B', 'Y')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.MANY))
        schema.addEntity('C')
        schema.addAttribute('C', 'Z')
        schema.addAttribute('C', 'W')
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.MANY))

        modelDeps = ['[B, AB, A].X -> [B].Y', '[C, BC, B].Y -> [C].Z', '[C].Z -> [C].W',
                     '[A].X -> [A].V', '[A, AB, B, BC, C].W -> [A].V']
        model = Model(schema, modelDeps)
        rcd = RCD.runRCD(schema, Oracle(model, 8), 4)
        self.assertRCDOutputEqual(modelDeps, None, None, rcd.orientedDependencies, None, None)


    def testEdgeOrientationOrderArgument(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addEntity('B')
        schema.addAttribute('B', 'Y')
        schema.addAttribute('B', 'W')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.MANY))
        schema.addEntity('C')
        schema.addAttribute('C', 'Z')
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.MANY))
        modelDeps = ['[B, AB, A].X -> [B].Y', '[C, BC, B].Y -> [C].Z', '[B].W -> [B].Y']
        model = Model(schema, modelDeps)
        rcd = RCD(schema, Oracle(model, 8), 4)
        undirectedDeps =['[B].Y -> [B].W', '[B].W -> [B].Y', '[A, AB, B].Y -> [A].X', '[C, BC, B].Y -> [C].Z',
                         '[B, BC, C].Z -> [B].Y', '[B, AB, A].X -> [B].Y']
        sepsets = {('[B].W', '[B, AB, A, AB, B].Y'): set(), ('[C].Z', '[C, BC, B].W'): {'[C, BC, B].Y'},
                   ('[A].X', '[A, AB, B].W'): set(), ('[C, BC, B].W', '[C].Z'): {'[C, BC, B].Y'},
                   ('[A, AB, B].W', '[A].X'): set(), ('[C].Z', '[C, BC, B, AB, A].X'): {'[C, BC, B].Y'},
                   ('[B, AB, A, AB, B].Y', '[B].W'): set(), ('[B, BC, C, BC, B].Y', '[B].W'): set(),
                   ('[C, BC, B, AB, A].X', '[C].Z'): {'[C, BC, B].Y'}, ('[B].W', '[B, BC, C, BC, B].Y'): set()}

        rcd.setUndirectedDependencies(undirectedDeps)
        rcd.setSepsets(sepsets)
        rcd.orientDependencies() # rboOrder='normal' by default
        self.assertEqual(2, rcd.edgeOrientationRuleFrequency['CD'])
        self.assertEqual(1, rcd.edgeOrientationRuleFrequency['RBO'])
        self.assertEqual(0, rcd.edgeOrientationRuleFrequency['KNC'])

        rcd.setUndirectedDependencies(undirectedDeps)
        rcd.setSepsets(sepsets)
        rcd.resetEdgeOrientationUsage()
        rcd.orientDependencies(rboOrder='first')
        self.assertEqual(1, rcd.edgeOrientationRuleFrequency['CD'])
        self.assertEqual(2, rcd.edgeOrientationRuleFrequency['RBO'])
        self.assertEqual(0, rcd.edgeOrientationRuleFrequency['KNC'])

        rcd.setUndirectedDependencies(undirectedDeps)
        rcd.setSepsets(sepsets)
        rcd.resetEdgeOrientationUsage()
        rcd.orientDependencies(rboOrder='last')
        self.assertEqual(2, rcd.edgeOrientationRuleFrequency['CD'])
        self.assertEqual(0, rcd.edgeOrientationRuleFrequency['RBO'])
        self.assertEqual(1, rcd.edgeOrientationRuleFrequency['KNC'])

        # bad rboOrder value
        TestUtil.assertRaisesMessage(self, Exception, "rboOrder must be one of 'normal', 'first', or 'last': found 'xx'",
             rcd.orientDependencies, 'xx')


    def testOrderIndependentSkeletonsArgument(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'W')
        schema.addAttribute('A', 'X')
        schema.addAttribute('A', 'Y')
        schema.addAttribute('A', 'Z')
        model = Model(schema, ['[A].X -> [A].Z', '[A].X -> [A].W', '[A].W -> [A].Y'])
        rcd = RCD(schema, Oracle(model), 0)
        rcd.potentialDependencySorter = lambda l: sorted(l)
        rcd.generateSepsetCombinations = lambda l, n: (v for v in sorted([sorted(sepset) for sepset in itertools.combinations(l, n)]))
        rcd.identifyUndirectedDependencies() # orderIndependentSkeleton defaults to False
        ciBreakDown, eoFrequencies = rcd.report()
        self.assertEqual({'Phase I': 23, 'depth 0': 12, 'depth 1': 11, 'Phase II': 0, 'total': 23}, ciBreakDown)
        expectedDeps = ['[A].X -> [A].Z', '[A].X -> [A].W', '[A].W -> [A].Y',
                        '[A].Z -> [A].X', '[A].W -> [A].X', '[A].Y -> [A].W']
        expectedSepsets = {('[A].W', '[A].Z'): {'[A].X'}, ('[A].Z', '[A].W'): {'[A].X'},
                           ('[A].X', '[A].Y'): {'[A].W'}, ('[A].Y', '[A].X'): {'[A].W'},
                           ('[A].Y', '[A].Z'): {'[A].X'}, ('[A].Z', '[A].Y'): {'[A].X'}}
        self.assertRCDOutputEqual(expectedDeps, expectedSepsets, None, rcd.undirectedDependencies, rcd.sepsets, None)

        rcd = RCD(schema, Oracle(model), 0)
        rcd.potentialDependencySorter = lambda l: sorted(l)
        rcd.generateSepsetCombinations = lambda l, n: (v for v in sorted([sorted(sepset) for sepset in itertools.combinations(l, n)]))
        rcd.identifyUndirectedDependencies(orderIndependentSkeleton=True)
        ciBreakDown, eoFrequencies = rcd.report()
        self.assertEqual({'Phase I': 27, 'depth 0': 12, 'depth 1': 15, 'Phase II': 0, 'total': 27}, ciBreakDown)
        expectedDeps = ['[A].X -> [A].Z', '[A].X -> [A].W', '[A].W -> [A].Y',
                        '[A].Z -> [A].X', '[A].W -> [A].X', '[A].Y -> [A].W']
        expectedSepsets = {('[A].W', '[A].Z'): {'[A].X'}, ('[A].Z', '[A].W'): {'[A].X'},
                           ('[A].X', '[A].Y'): {'[A].W'}, ('[A].Y', '[A].X'): {'[A].W'},
                           ('[A].Y', '[A].Z'): {'[A].W'}, ('[A].Z', '[A].Y'): {'[A].W'}} # this line is the difference from above
        self.assertRCDOutputEqual(expectedDeps, expectedSepsets, None, rcd.undirectedDependencies, rcd.sepsets, None)


    def testColliderWithIntersectionsInSepset(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.MANY))
        schema.addAttribute('A', 'X1')
        schema.addAttribute('B', 'Y1')
        schema.addAttribute('AB', 'XY1')
        modelDeps = ['[AB, B].Y1 -> [AB].XY1',
                     '[B, AB, A].X1 -> [B].Y1',
                     '[AB, A].X1 -> [AB].XY1']
        model = Model(schema, modelDeps)
        rcd = RCD(schema, Oracle(model, 8), 4)
        rcd.potentialDependencySorter = lambda l: sorted(l)
        rcd.generateSepsetCombinations = lambda l, n: (v for v in sorted([sorted(sepset) for sepset in itertools.combinations(l, n)]))
        rcd.identifyUndirectedDependencies()
        od = OrderedDict()
        od['AB'] = rcd.perspectiveToAgg['AB']
        od['A'] = rcd.perspectiveToAgg['A']
        od['B'] = rcd.perspectiveToAgg['B']
        rcd.perspectiveToAgg = od
        rcd.orientDependencies()
        self.assertEqual(1.0, ModelEvaluation.orientedPrecision(model, rcd.orientedDependencies))


    def testColliderSingletonPathEither1or3(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.ONE))
        schema.addAttribute('A', 'X1')
        schema.addAttribute('A', 'X2')
        schema.addAttribute('B', 'Y1')
        schema.addAttribute('B', 'Y2')
        schema.addAttribute('AB', 'XY1')
        modelDeps = ['[B, AB, A, AB, B].Y2 -> [B].Y1',
                     '[A, AB].XY1 -> [A].X2',
                     '[B, AB, A, AB].XY1 -> [B].Y2',
                     '[B, AB, A, AB].XY1 -> [B].Y1',
                     '[B, AB, A].X2 -> [B].Y1']

        model = Model(schema, modelDeps)
        rcd = RCD(schema, Oracle(model, 8), 4)
        rcd.potentialDependencySorter = lambda l: sorted(l)
        rcd.generateSepsetCombinations = lambda l, n: (v for v in sorted([sorted(sepset) for sepset in itertools.combinations(l, n)]))
        rcd.identifyUndirectedDependencies()
        od = OrderedDict()
        od['B'] = rcd.perspectiveToAgg['B']
        od['A'] = rcd.perspectiveToAgg['A']
        od['AB'] = rcd.perspectiveToAgg['AB']
        rcd.perspectiveToAgg = od
        rcd.orientDependencies()
        self.assertEqual(1.0, ModelEvaluation.orientedPrecision(model, rcd.orientedDependencies))


    def testRBOWithIntersectionsInSepset(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.MANY))
        schema.addAttribute('B', 'Y')
        schema.addAttribute('AB', 'XY')

        modelDeps = ['[AB, A, AB, B].Y -> [AB].XY']
        model = Model(schema, modelDeps)
        rcd = RCD(schema, Oracle(model, 8), 4)
        rcd.identifyUndirectedDependencies()
        rcd.orientDependencies(rboOrder='first')
        self.assertEqual(1.0, ModelEvaluation.orientedPrecision(model, rcd.orientedDependencies))


    def testExposedAggConstructionError(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.MANY))
        schema.addAttribute('AB', 'XY1')
        schema.addAttribute('AB', 'XY2')
        schema.addAttribute('AB', 'XY3')

        modelDeps = ['[AB, A, AB, B, AB].XY1 -> [AB].XY2',
                     '[AB, A, AB, B, AB].XY2 -> [AB].XY3']
        model = Model(schema, modelDeps)

        rcd = RCD(schema, Oracle(model, 8), 4, depth=3)
        rcd.identifyUndirectedDependencies()
        rcd.orientDependencies(rboOrder='last')
        self.assertEqual(1.0, ModelEvaluation.orientedPrecision(model, rcd.orientedDependencies))


    def testExposedMR3UndirectedError(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.MANY))
        schema.addAttribute('A', 'X1')
        schema.addAttribute('A', 'X2')
        schema.addAttribute('A', 'X3')
        schema.addAttribute('B', 'Y1')
        schema.addAttribute('B', 'Y2')
        schema.addAttribute('AB', 'XY1')
        schema.addAttribute('AB', 'XY2')
        schema.addAttribute('AB', 'XY3')

        modelDeps = ['[A, AB, B, AB, A].X3 -> [A].X1',
                     '[AB, B].Y2 -> [AB].XY3',
                     '[AB, A].X3 -> [AB].XY1',
                     '[A, AB, B].Y2 -> [A].X2',
                     '[AB].XY1 -> [AB].XY2',
                     '[A, AB, B, AB].XY1 -> [A].X1',
                     '[A, AB, B, AB].XY2 -> [A].X1',
                     '[AB, A].X3 -> [AB].XY2',
                     '[A, AB, B].Y1 -> [A].X3',
                     '[AB, B, AB, A].X3 -> [AB].XY3']
        model = Model(schema, modelDeps)

        rcd = RCD(schema, Oracle(model, 8), 4, depth=3)
        rcd.identifyUndirectedDependencies()
        rcd.orientDependencies(rboOrder='last')
        self.assertEqual(1.0, ModelEvaluation.skeletonPrecision(model, rcd.undirectedDependencies))
        self.assertEqual(1.0, ModelEvaluation.skeletonRecall(model, rcd.undirectedDependencies))
        self.assertEqual(1.0, ModelEvaluation.orientedPrecision(model, rcd.orientedDependencies))


    def testRBONoSymmetryPattern(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.ONE))
        schema.addAttribute('B', 'Y1')
        schema.addAttribute('AB', 'XY1')

        modelDeps = ['[B, AB, A, AB].XY1 -> [B].Y1']
        model = Model(schema, modelDeps)

        rcd = RCD(schema, Oracle(model, 8), 4, depth=3)
        rcd.identifyUndirectedDependencies()
        rcd.orientDependencies(rboOrder='first')
        self.assertEqual(1.0, ModelEvaluation.skeletonPrecision(model, rcd.undirectedDependencies))
        self.assertEqual(1.0, ModelEvaluation.skeletonRecall(model, rcd.undirectedDependencies))
        self.assertEqual(1.0, ModelEvaluation.orientedPrecision(model, rcd.orientedDependencies))
        self.assertEqual(1.0, ModelEvaluation.orientedRecall(model, rcd.orientedDependencies))


    def assertRCDOutputEqual(self, expectedDepStrs, expectedSepsetStrs, expectedNumDSepCalls,
                                actualDeps, actualSepset, mockOracle):
        # test dependencies are equal
        expectedDeps = [ParserUtil.parseRelDep(expectedDepStr) for expectedDepStr in expectedDepStrs]
        TestUtil.assertUnorderedListEqual(self, expectedDeps, list(actualDeps))

        # test sepsets are equal, if passed in
        if expectedSepsetStrs and actualSepset:
            expectedSepset = {(ParserUtil.parseRelVar(relVar1Str), ParserUtil.parseRelVar(relVar2Str)):
                                  {ParserUtil.parseRelVar(condVarStr) for condVarStr in sepsetStr}
                              for (relVar1Str, relVar2Str), sepsetStr in expectedSepsetStrs.items()}
            self.assertDictEqual(expectedSepset, actualSepset)

        # test the number of d-separation calls, if passed in
        if expectedNumDSepCalls and mockOracle:
            self.assertEqual(expectedNumDSepCalls, mockOracle.isConditionallyIndependent.call_count)


if __name__ == '__main__':
    unittest.main()