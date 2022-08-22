import unittest
from causality.model import ParserUtil
from causality.model.Schema import Schema
from causality.model.Model import Model
from causality.learning.PC import PC
from causality.test import TestUtil
from mock import MagicMock
from mock import PropertyMock
import networkx as nx
from citest.CITest import Oracle


class TestPC(unittest.TestCase):

    def testPCObj(self):
        schema = Schema()
        model = Model(schema, [])
        oracle = Oracle(model)
        pc = PC(schema, oracle)
        self.assertEqual(schema, pc.schema)
        self.assertEqual(oracle.model, pc.oracle.model)


    def testPhaseI(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        model = Model(schema, [])
        mockOracle = MagicMock(wraps=Oracle(model))
        mockModelProperty = PropertyMock()
        type(mockOracle).model = mockModelProperty
        pc = PC(schema, mockOracle)
        pc.pcPhaseI()
        self.assertEqual(0, mockModelProperty.call_count)   # forces us not to cheat by simply returning the model
        self.assertPCOutputEqual(['[A].X'], [], {}, 0, pc.undirectedSkeleton, pc.sepsets, mockOracle)

        schema.addAttribute('A', 'Y')
        model = Model(schema, [])
        mockOracle = MagicMock(wraps=Oracle(model))
        mockModelProperty = PropertyMock()
        type(mockOracle).model = mockModelProperty
        pc = PC(schema, mockOracle)
        pc.pcPhaseI()
        self.assertEqual(0, mockModelProperty.call_count)   # forces us not to cheat by simply returning the model
        expectedSepset = {('[A].X', '[A].Y'): set(), ('[A].Y', '[A].X'): set()}
        self.assertPCOutputEqual(['[A].X', '[A].Y'], [], expectedSepset, 1, pc.undirectedSkeleton,
                                 pc.sepsets, mockOracle)

        model = Model(schema, ['[A].X -> [A].Y'])
        mockOracle = MagicMock(wraps=Oracle(model))
        pc = PC(schema, mockOracle)
        pc.pcPhaseI()
        expectedNodes = ['[A].X', '[A].Y']
        expectedEdges = [('[A].X', '[A].Y'), ('[A].Y', '[A].X')]
        self.assertPCOutputEqual(expectedNodes, expectedEdges, {}, 2, pc.undirectedSkeleton,
                                 pc.sepsets, mockOracle)

        schema.addAttribute('A', 'Z')
        model = Model(schema, ['[A].X -> [A].Y'])
        mockOracle = MagicMock(wraps=Oracle(model))
        pc = PC(schema, mockOracle)
        pc.pcPhaseI()
        expectedNodes = ['[A].X', '[A].Y', '[A].Z']
        expectedEdges = [('[A].X', '[A].Y'), ('[A].Y', '[A].X')]
        expectedSepset = {('[A].X', '[A].Z'): set(), ('[A].Z', '[A].X'): set(), ('[A].Y', '[A].Z'): set(),
                          ('[A].Z', '[A].Y'): set()}
        self.assertPCOutputEqual(expectedNodes, expectedEdges, expectedSepset, 4, pc.undirectedSkeleton,
                                 pc.sepsets, mockOracle)

        model = Model(schema, ['[A].X -> [A].Z', '[A].Z -> [A].Y'])
        mockOracle = MagicMock(wraps=Oracle(model))
        pc = PC(schema, mockOracle)
        pc.pcPhaseI()
        expectedNodes = ['[A].X', '[A].Y', '[A].Z']
        expectedEdges = [('[A].X', '[A].Z'), ('[A].Z', '[A].X'), ('[A].Z', '[A].Y'), ('[A].Y', '[A].Z')]
        expectedSepset = {('[A].X', '[A].Y'): {'[A].Z'}, ('[A].Y', '[A].X'): {'[A].Z'}}
        expectedDSepCount = 9

        self.assertPCOutputEqual(expectedNodes, expectedEdges, expectedSepset, expectedDSepCount, pc.undirectedSkeleton,
                                 pc.sepsets, mockOracle)


    def testPhaseIBiggerConditionalSets(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addAttribute('A', 'Y')
        schema.addAttribute('A', 'Z')
        schema.addAttribute('A', 'W')
        model = Model(schema, ['[A].X -> [A].Y', '[A].X -> [A].Z', '[A].Y -> [A].W', '[A].Z -> [A].W'])
        pc = PC(schema, Oracle(model))
        pc.pcPhaseI()
        expectedNodes = ['[A].X', '[A].Y', '[A].Z', '[A].W']
        expectedEdges = [('[A].X', '[A].Y'), ('[A].Y', '[A].X'), ('[A].X', '[A].Z'), ('[A].Z', '[A].X'),
                         ('[A].W', '[A].Y'), ('[A].Y', '[A].W'), ('[A].W', '[A].Z'), ('[A].Z', '[A].W')]
        expectedSepset = {('[A].X', '[A].W'): {'[A].Y', '[A].Z'}, ('[A].W', '[A].X'): {'[A].Y', '[A].Z'},
                          ('[A].Y', '[A].Z'): {'[A].X'}, ('[A].Z', '[A].Y'): {'[A].X'}}
        self.assertPCOutputEqual(expectedNodes, expectedEdges, expectedSepset, None, pc.undirectedSkeleton,
                                 pc.sepsets, None)


    def testPhaseII(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        model = Model(schema, [])
        pc = PC(schema, Oracle(model))
        pc.pcPhaseI()
        pc.pcPhaseII()
        expectedNodes = ['[A].X']
        expectedEdges = []
        self.assertPCOutputEqual(expectedNodes, expectedEdges, None, None, pc.partiallyDirectedGraph, None, None)

        schema.addAttribute('A', 'Y')
        schema.addAttribute('A', 'Z')
        model = Model(schema, ['[A].X -> [A].Y'])
        pc = PC(schema, Oracle(model))
        pc.pcPhaseI()
        pc.pcPhaseII()
        expectedNodes = ['[A].X', '[A].Y', '[A].Z']
        expectedEdges = [('[A].X', '[A].Y'), ('[A].Y', '[A].X')]
        self.assertPCOutputEqual(expectedNodes, expectedEdges, None, None, pc.partiallyDirectedGraph, None, None)

        model = Model(schema, ['[A].X -> [A].Z', '[A].Y -> [A].Z'])
        pc = PC(schema, Oracle(model))
        pc.pcPhaseI()
        pc.pcPhaseII()
        expectedNodes = ['[A].X', '[A].Y', '[A].Z']
        expectedEdges = [('[A].X', '[A].Z'), ('[A].Y', '[A].Z')]
        self.assertPCOutputEqual(expectedNodes, expectedEdges, None, None, pc.partiallyDirectedGraph, None, None)

        model = Model(schema, ['[A].X -> [A].Z', '[A].Z -> [A].Y'])
        pc = PC(schema, Oracle(model))
        pc.pcPhaseI()
        pc.pcPhaseII()
        expectedNodes = ['[A].X', '[A].Y', '[A].Z']
        expectedEdges = [('[A].X', '[A].Z'), ('[A].Y', '[A].Z'), ('[A].Z', '[A].X'), ('[A].Z', '[A].Y')]
        self.assertPCOutputEqual(expectedNodes, expectedEdges, None, None, pc.partiallyDirectedGraph, None, None)

        schema.addAttribute('A', 'W')
        model = Model(schema, ['[A].X -> [A].Z', '[A].Y -> [A].Z'])
        pc = PC(schema, Oracle(model))
        pc.pcPhaseI()
        pc.pcPhaseII()
        expectedNodes = ['[A].X', '[A].Y', '[A].Z', '[A].W']
        expectedEdges = [('[A].X', '[A].Z'), ('[A].Y', '[A].Z')]
        self.assertPCOutputEqual(expectedNodes, expectedEdges, None, None, pc.partiallyDirectedGraph, None, None)

        schema.addAttribute('A', 'V')
        model = Model(schema, ['[A].X -> [A].Z', '[A].Y -> [A].Z', '[A].Y -> [A].W', '[A].V -> [A].W'])
        pc = PC(schema, Oracle(model))
        pc.pcPhaseI()
        pc.pcPhaseII()
        expectedNodes = ['[A].X', '[A].Y', '[A].Z', '[A].W', '[A].V']
        expectedEdges = [('[A].X', '[A].Z'), ('[A].Y', '[A].Z'), ('[A].Y', '[A].W'), ('[A].V', '[A].W')]
        self.assertPCOutputEqual(expectedNodes, expectedEdges, None, None, pc.partiallyDirectedGraph, None, None)

        model = Model(schema, ['[A].X -> [A].Z', '[A].Y -> [A].Z', '[A].Z -> [A].W', '[A].W -> [A].V'])
        pc = PC(schema, Oracle(model))
        pc.pcPhaseI()
        pc.pcPhaseII()
        expectedNodes = ['[A].X', '[A].Y', '[A].Z', '[A].W', '[A].V']
        expectedEdges = [('[A].X', '[A].Z'), ('[A].Y', '[A].Z'), ('[A].Z', '[A].W'), ('[A].W', '[A].V')]
        self.assertPCOutputEqual(expectedNodes, expectedEdges, None, None, pc.partiallyDirectedGraph, None, None)

        model = Model(schema, ['[A].X -> [A].Z', '[A].Y -> [A].Z', '[A].Z -> [A].W', '[A].Y -> [A].W'])
        pc = PC(schema, Oracle(model))
        pc.pcPhaseI()
        pc.pcPhaseII()
        expectedNodes = ['[A].X', '[A].Y', '[A].Z', '[A].W', '[A].V']
        expectedEdges = [('[A].X', '[A].Z'), ('[A].Y', '[A].Z'), ('[A].Z', '[A].W'), ('[A].Y', '[A].W')]
        self.assertPCOutputEqual(expectedNodes, expectedEdges, None, None, pc.partiallyDirectedGraph, None, None)

        model = Model(schema, ['[A].X -> [A].Y', '[A].X -> [A].W', '[A].X -> [A].Z', '[A].Y -> [A].Z', '[A].W -> [A].Z'])
        pc = PC(schema, Oracle(model))
        pc.pcPhaseI()
        pc.pcPhaseII()
        expectedNodes = ['[A].X', '[A].Y', '[A].Z', '[A].W', '[A].V']
        expectedEdges = [('[A].X', '[A].Z'), ('[A].Y', '[A].Z'), ('[A].W', '[A].Z'), ('[A].X', '[A].Y'),
                         ('[A].Y', '[A].X'), ('[A].X', '[A].W'), ('[A].W', '[A].X')]
        self.assertPCOutputEqual(expectedNodes, expectedEdges, None, None, pc.partiallyDirectedGraph, None, None)

        model = Model(schema, ['[A].W -> [A].X', '[A].W -> [A].Z', '[A].W -> [A].Y', '[A].X -> [A].V', '[A].X -> [A].Z',
                               '[A].Y -> [A].V', '[A].Y -> [A].Z', '[A].Z -> [A].V'])
        pc = PC(schema, Oracle(model))
        pc.pcPhaseI()
        pc.pcPhaseII()
        expectedNodes = ['[A].X', '[A].Y', '[A].Z', '[A].W', '[A].V']
        expectedEdges = [('[A].X', '[A].V'), ('[A].Y', '[A].V'), ('[A].Z', '[A].V'), ('[A].Y', '[A].Z'),
                         ('[A].W', '[A].Z'), ('[A].X', '[A].Z'), ('[A].W', '[A].X'), ('[A].X', '[A].W'),
                         ('[A].W', '[A].Y'), ('[A].Y', '[A].W')]
        self.assertPCOutputEqual(expectedNodes, expectedEdges, None, None, pc.partiallyDirectedGraph, None, None)


    def testNoSkeletonBeforePhaseII(self):
        # must have an undirected skeleton and sepsets before running Phase II
        schema = Schema()
        model = Model(schema, [])
        pc = PC(schema, Oracle(model))
        TestUtil.assertRaisesMessage(self, Exception, "No undirected skeleton found. Try running Phase I first.",
            pc.pcPhaseII)

        # what if we set the skeleton to None?
        pc = PC(schema, Oracle(model))
        pc.undirectedSkeleton = None
        TestUtil.assertRaisesMessage(self, Exception, "No undirected skeleton found. Try running Phase I first.",
             pc.pcPhaseII)

        # what if we don't set the sepset?
        pc = PC(schema, Oracle(model))
        pc.setUndirectedSkeleton(nx.DiGraph())
        TestUtil.assertRaisesMessage(self, Exception, "No sepsets found. Try running Phase I first.",
             pc.pcPhaseII)

        # what if we set the sepsets to None?
        pc = PC(schema, Oracle(model))
        pc.setUndirectedSkeleton(nx.DiGraph())
        pc.sepsets = None
        TestUtil.assertRaisesMessage(self, Exception, "No sepsets found. Try running Phase I first.",
             pc.pcPhaseII)


    def testSetUndirectedSkeleton(self):
        schema = Schema()
        model = Model(schema, [])
        pc = PC(schema, Oracle(model))
        undirectedSkeleton = nx.DiGraph()
        pc.setUndirectedSkeleton(undirectedSkeleton)
        self.assertEqual(undirectedSkeleton, pc.undirectedSkeleton)

        TestUtil.assertRaisesMessage(self, Exception, "Undirected skeleton must be a networkx DiGraph: found None",
             pc.setUndirectedSkeleton, None)

        # nodes must match the attributes of the schema
        undirectedSkeleton = nx.DiGraph()
        undirectedSkeleton.add_node(ParserUtil.parseRelVar('[A].X'))
        TestUtil.assertRaisesMessage(self, Exception, "Undirected skeleton's nodes must match schema attributes",
             pc.setUndirectedSkeleton, undirectedSkeleton)

        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        model = Model(schema, [])
        pc = PC(schema, Oracle(model))
        undirectedSkeleton = nx.DiGraph()
        TestUtil.assertRaisesMessage(self, Exception, "Undirected skeleton's nodes must match schema attributes",
             pc.setUndirectedSkeleton, undirectedSkeleton)


    def testSetSepsets(self):
        schema = Schema()
        model = Model(schema, [])
        pc = PC(schema, Oracle(model))
        pc.setSepsets({})
        self.assertEqual({}, pc.sepsets)

        TestUtil.assertRaisesMessage(self, Exception, "Sepsets must be a dictionary: found None",
                                     pc.setSepsets, None)


    def testSetPhaseIPattern(self):
        # edges in skeleton and sepsets should be useful for Phase II
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addAttribute('A', 'Y')
        schema.addAttribute('A', 'Z')
        model = Model(schema, ['[A].X -> [A].Z', '[A].Y -> [A].Z'])
        pc = PC(schema, Oracle(model))
        undirectedSkeleton = nx.DiGraph()
        relVarX = ParserUtil.parseRelVar('[A].X')
        relVarY = ParserUtil.parseRelVar('[A].Y')
        relVarZ = ParserUtil.parseRelVar('[A].Z')
        undirectedSkeleton.add_edges_from([(relVarX, relVarZ), (relVarZ, relVarX),
                                           (relVarY, relVarZ), (relVarZ, relVarY)])
        pc.setUndirectedSkeleton(undirectedSkeleton)
        pc.setSepsets({(relVarX, relVarY): set(), (relVarY, relVarX): set()})
        pc.pcPhaseII()
        self.assertPCOutputEqual(['[A].X', '[A].Y', '[A].Z'], [('[A].X', '[A].Z'), ('[A].Y', '[A].Z')],
                                 None, None, pc.partiallyDirectedGraph, None, None)


    def testPCLearnModel(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X')
        schema.addAttribute('A', 'Y')
        schema.addAttribute('A', 'Z')
        schema.addAttribute('A', 'W')
        model = Model(schema, ['[A].X -> [A].Y', '[A].X -> [A].W', '[A].X -> [A].Z', '[A].Y -> [A].Z', '[A].W -> [A].Z'])
        pc = PC(schema, Oracle(model))
        pc.learnModel()
        expectedNodes = ['[A].X', '[A].Y', '[A].Z', '[A].W']
        expectedEdges = [('[A].X', '[A].Z'), ('[A].Y', '[A].Z'), ('[A].W', '[A].Z'), ('[A].X', '[A].Y'),
                         ('[A].Y', '[A].X'), ('[A].W', '[A].X'), ('[A].X', '[A].W')]
        self.assertPCOutputEqual(expectedNodes, expectedEdges, None, None, pc.partiallyDirectedGraph, None, None)


    def assertPCOutputEqual(self, expectedNodeStrs, expectedEdgeStrs, expectedSepsetStrs, expectedNumDSepCalls,
                                  dag, sepset, mockOracle):
        # test nodes are equal
        expectedNodes = [ParserUtil.parseRelVar(nodeStr) for nodeStr in expectedNodeStrs]
        TestUtil.assertUnorderedListEqual(self, expectedNodes, dag.nodes())

        # test edges are equal
        self.assertEqual(len(expectedEdgeStrs), len(dag.edges()))
        for expectedEdgeStr in expectedEdgeStrs:
            expectedEdge = (ParserUtil.parseRelVar(expectedEdgeStr[0]), ParserUtil.parseRelVar(expectedEdgeStr[1]))
            self.assertIn(expectedEdge, dag.edges())

        # test sepsets are equal, if passed in
        if expectedSepsetStrs and sepset:
            expectedSepset = {(ParserUtil.parseRelVar(relVar1Str), ParserUtil.parseRelVar(relVar2Str)):
                              {ParserUtil.parseRelVar(condVarStr) for condVarStr in sepsetStr}
                          for (relVar1Str, relVar2Str), sepsetStr in expectedSepsetStrs.items()}
            self.assertDictEqual(expectedSepset, sepset)

        # test the number of d-separation calls, if passed in
        if expectedNumDSepCalls and mockOracle:
            self.assertEqual(expectedNumDSepCalls, mockOracle.isConditionallyIndependent.call_count)


if __name__ == '__main__':
    unittest.main()
