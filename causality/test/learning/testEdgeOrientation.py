import unittest
import networkx as nx
from causality.learning import EdgeOrientation

class TestEdgeOrientation(unittest.TestCase):

    def testColliderDetection(self):
        # should orient X->Z<-Y
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z'])
        edgePairs = [('X', 'Z'), ('Z', 'X'), ('Y', 'Z'), ('Z', 'Y')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        sepsets = {('X', 'Y'): set(), ('Y', 'X'): set()}
        didOrient = EdgeOrientation.applyColliderDetectionNaive(partiallyDirectedGraph, sepsets)
        self.assertTrue(didOrient)
        self.assertEqual(2, len(partiallyDirectedGraph.edges()))

        # should orient X->Z<-Y, even though X and Y have a sepset (W), it doesn't include Z
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z', 'W'])
        edgePairs = [('X', 'Z'), ('Z', 'X'), ('Y', 'Z'), ('Z', 'Y'), ('X', 'W'), ('W', 'X'), ('Y', 'W'), ('W', 'Y')]
        sepsets = {('X', 'Y'): {'W'}, ('Y', 'X'): {'W'}, ('Z', 'W'): {'X', 'Y'}, ('W', 'Z'): {'X', 'Y'}}
        partiallyDirectedGraph.add_edges_from(edgePairs)
        didOrient = EdgeOrientation.applyColliderDetectionNaive(partiallyDirectedGraph, sepsets)
        self.assertTrue(didOrient)
        self.assertEqual(6, len(partiallyDirectedGraph.edges()))

        # should orient no edges (Z is in sepset(X, Y))
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z'])
        edgePairs = [('X', 'Z'), ('Z', 'X'), ('Y', 'Z'), ('Z', 'Y')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        sepsets = {('X', 'Y'): {'Z'}, ('Y', 'X'): {'Z'}}
        didOrient = EdgeOrientation.applyColliderDetectionNaive(partiallyDirectedGraph, sepsets)
        self.assertFalse(didOrient)
        self.assertEqual(4, len(partiallyDirectedGraph.edges()))

        # should orient no edges because X -> Y
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z'])
        edgePairs = [('X', 'Z'), ('Z', 'X'), ('Y', 'Z'), ('Z', 'Y'), ('X', 'Y')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        sepsets = {}
        didOrient = EdgeOrientation.applyColliderDetectionNaive(partiallyDirectedGraph, sepsets)
        self.assertFalse(didOrient)
        self.assertEqual(5, len(partiallyDirectedGraph.edges()))

        # should orient no edges because X <- Y
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z'])
        edgePairs = [('X', 'Z'), ('Z', 'X'), ('Y', 'Z'), ('Z', 'Y'), ('Y', 'X')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        sepsets = {}
        didOrient = EdgeOrientation.applyColliderDetectionNaive(partiallyDirectedGraph, sepsets)
        self.assertFalse(didOrient)
        self.assertEqual(5, len(partiallyDirectedGraph.edges()))

        # should orient no edges because X is adjacent to Y
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z'])
        edgePairs = [('X', 'Z'), ('Z', 'X'), ('Y', 'Z'), ('Z', 'Y'), ('X', 'Y'), ('Y', 'X')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        sepsets = {}
        didOrient = EdgeOrientation.applyColliderDetectionNaive(partiallyDirectedGraph, sepsets)
        self.assertFalse(didOrient)
        self.assertEqual(6, len(partiallyDirectedGraph.edges()))

        # should orient no edges because already oriented
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z'])
        edgePairs = [('X', 'Z'), ('Y', 'Z')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        sepsets = {('X', 'Y'): set(), ('Y', 'X'): set()}
        didOrient = EdgeOrientation.applyColliderDetectionNaive(partiallyDirectedGraph, sepsets)
        self.assertFalse(didOrient)
        self.assertEqual(2, len(partiallyDirectedGraph.edges()))

        # should orient X->Z even though Z<-Y already oriented
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z'])
        edgePairs = [('X', 'Z'), ('Z', 'X'), ('Y', 'Z')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        sepsets = {('X', 'Y'): set(), ('Y', 'X'): set()}
        didOrient = EdgeOrientation.applyColliderDetectionNaive(partiallyDirectedGraph, sepsets)
        self.assertTrue(didOrient)
        self.assertEqual(2, len(partiallyDirectedGraph.edges()))

        # should orient Y->Z even though Z<-X already oriented
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z'])
        edgePairs = [('X', 'Z'), ('Z', 'Y'), ('Y', 'Z')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        sepsets = {('X', 'Y'): set(), ('Y', 'X'): set()}
        didOrient = EdgeOrientation.applyColliderDetectionNaive(partiallyDirectedGraph, sepsets)
        self.assertTrue(didOrient)
        self.assertEqual(2, len(partiallyDirectedGraph.edges()))

        # should orient X->Z<-Y and Y->Z<-W
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z', 'W'])
        edgePairs = [('X', 'Z'), ('Z', 'X'), ('Y', 'Z'), ('Z', 'Y'), ('W', 'Z'), ('Z', 'W')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        sepsets = {('X', 'Y'): set(), ('Y', 'X'): set(), ('W', 'Y'): set(), ('Y', 'W'): set(),
                   ('X', 'W'): set(), ('W', 'X'): set()}
        didOrient = EdgeOrientation.applyColliderDetectionNaive(partiallyDirectedGraph, sepsets)
        self.assertTrue(didOrient)
        self.assertEqual(3, len(partiallyDirectedGraph.edges()))

        # should only orient one given inconsistent sepsets: X->Z<-Y<-W
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z', 'W'])
        edgePairs = [('X', 'Z'), ('Z', 'X'), ('Y', 'Z'), ('Z', 'Y'), ('W', 'Y'), ('Y', 'W')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        sepsets = {('X', 'Y'): set(), ('Y', 'X'): set(), ('W', 'Z'): set(), ('Z', 'W'): set()}
        didOrient = EdgeOrientation.applyColliderDetectionNaive(partiallyDirectedGraph, sepsets)
        self.assertTrue(didOrient)
        self.assertEqual(4, len(partiallyDirectedGraph.edges()))


    def testKnownNonColliders(self):
        # should orient X->Y->Z
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z'])
        edgePairs = [('X', 'Y'), ('Y', 'Z'), ('Z', 'Y')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        didOrient = EdgeOrientation.applyKnownNonColliders(partiallyDirectedGraph)
        self.assertTrue(didOrient)
        self.assertEqual(2, len(partiallyDirectedGraph.edges()))

        # should orient no edges because X and Y have an undirected edge
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z'])
        edgePairs = [('X', 'Y'), ('Y', 'X'), ('Y', 'Z'), ('Z', 'Y')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        didOrient = EdgeOrientation.applyKnownNonColliders(partiallyDirectedGraph)
        self.assertFalse(didOrient)
        self.assertEqual(4, len(partiallyDirectedGraph.edges()))

        # should orient no edges because X -> Z
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z'])
        edgePairs = [('X', 'Y'), ('Y', 'Z'), ('Z', 'Y'), ('X', 'Z')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        didOrient = EdgeOrientation.applyKnownNonColliders(partiallyDirectedGraph)
        self.assertFalse(didOrient)
        self.assertEqual(4, len(partiallyDirectedGraph.edges()))

        # should orient no edges because X <- Z
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z'])
        edgePairs = [('X', 'Y'), ('Y', 'Z'), ('Z', 'Y'), ('Z', 'X')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        didOrient = EdgeOrientation.applyKnownNonColliders(partiallyDirectedGraph)
        self.assertFalse(didOrient)
        self.assertEqual(4, len(partiallyDirectedGraph.edges()))

        # should orient no edges because X - Z
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z'])
        edgePairs = [('X', 'Y'), ('Y', 'Z'), ('Z', 'Y'), ('X', 'Z'), ('Z', 'X')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        didOrient = EdgeOrientation.applyKnownNonColliders(partiallyDirectedGraph)
        self.assertFalse(didOrient)
        self.assertEqual(5, len(partiallyDirectedGraph.edges()))

        # should orient no edges because it already oriented Y->Z
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z'])
        edgePairs = [('X', 'Y'), ('Y', 'Z')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        didOrient = EdgeOrientation.applyKnownNonColliders(partiallyDirectedGraph)
        self.assertFalse(didOrient)
        self.assertEqual(2, len(partiallyDirectedGraph.edges()))


    def testCycleAvoidance(self):
        # should orient X->Y
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z'])
        edgePairs = [('X', 'Y'), ('Y', 'X'), ('X', 'Z'), ('Z', 'Y')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        didOrient = EdgeOrientation.applyCycleAvoidance(partiallyDirectedGraph)
        self.assertTrue(didOrient)
        self.assertEqual(3, len(partiallyDirectedGraph.edges()))

        # should orient X<-Y
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z'])
        edgePairs = [('X', 'Y'), ('Y', 'X'), ('Z', 'X'), ('Y', 'Z')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        didOrient = EdgeOrientation.applyCycleAvoidance(partiallyDirectedGraph)
        self.assertTrue(didOrient)
        self.assertEqual(3, len(partiallyDirectedGraph.edges()))

        # should orient no edges if two are unoriented in triple
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z'])
        edgePairs = [('X', 'Y'), ('Y', 'X'), ('X', 'Z'), ('Z', 'X'), ('Y', 'Z')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        didOrient = EdgeOrientation.applyCycleAvoidance(partiallyDirectedGraph)
        self.assertFalse(didOrient)
        self.assertEqual(5, len(partiallyDirectedGraph.edges()))

        # should orient no edges if X and Y are not adjacent (unshieled triple)
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z'])
        edgePairs = [('X', 'Z'), ('Z', 'X'), ('Y', 'Z')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        didOrient = EdgeOrientation.applyCycleAvoidance(partiallyDirectedGraph)
        self.assertFalse(didOrient)
        self.assertEqual(3, len(partiallyDirectedGraph.edges()))

        # should orient no edges because X already oriented to Y
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z'])
        edgePairs = [('X', 'Y'), ('X', 'Z'), ('Z', 'Y')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        didOrient = EdgeOrientation.applyCycleAvoidance(partiallyDirectedGraph)
        self.assertFalse(didOrient)
        self.assertEqual(3, len(partiallyDirectedGraph.edges()))

        # should orient no edges because common cause (no cycle to avoid for a shielded triple)
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z'])
        edgePairs = [('X', 'Y'), ('Y', 'X'), ('Z', 'X'), ('Z', 'Y')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        didOrient = EdgeOrientation.applyCycleAvoidance(partiallyDirectedGraph)
        self.assertFalse(didOrient)
        self.assertEqual(4, len(partiallyDirectedGraph.edges()))

        # should orient no edges because common effect (no cycle to avoid for a shielded triple)
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z'])
        edgePairs = [('X', 'Y'), ('Y', 'X'), ('X', 'Z'), ('Y', 'Z')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        didOrient = EdgeOrientation.applyCycleAvoidance(partiallyDirectedGraph)
        self.assertFalse(didOrient)
        self.assertEqual(4, len(partiallyDirectedGraph.edges()))


    def testMR3(self):
        # should orient X->Y
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z', 'W'])
        edgePairs = [('X', 'Y'), ('Y', 'X'), ('X', 'Z'), ('Z', 'X'), ('X', 'W'), ('W', 'X'), ('Z', 'Y'), ('W', 'Y')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        didOrient = EdgeOrientation.applyMR3(partiallyDirectedGraph)
        self.assertTrue(didOrient)
        self.assertEqual(7, len(partiallyDirectedGraph.edges()))

        # should orient no edges
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z', 'W'])
        edgePairs = [('X', 'Y'), ('Y', 'X'), ('X', 'Z'), ('Z', 'X'), ('Y', 'Z'), ('Y', 'W'), ('Z', 'W'),
                     ('X', 'W'), ('W', 'X')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        didOrient = EdgeOrientation.applyMR3(partiallyDirectedGraph)
        self.assertFalse(didOrient)
        self.assertEqual(9, len(partiallyDirectedGraph.edges()))

        # should orient no edges
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z', 'W', 'V'])
        edgePairs = [('X', 'Y'), ('Y', 'X'), ('X', 'Z'), ('Z', 'X'), ('Y', 'Z'), ('Y', 'W'), ('Z', 'W'), ('V', 'W'),
                     ('X', 'W'), ('W', 'X')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        didOrient = EdgeOrientation.applyMR3(partiallyDirectedGraph)
        self.assertFalse(didOrient)
        self.assertEqual(10, len(partiallyDirectedGraph.edges()))

        # should orient no edges because already oriented X->Y
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z', 'W'])
        edgePairs = [('X', 'Y'), ('X', 'Z'), ('Z', 'X'), ('X', 'W'), ('W', 'X'), ('Z', 'Y'), ('W', 'Y')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        didOrient = EdgeOrientation.applyMR3(partiallyDirectedGraph)
        self.assertFalse(didOrient)
        self.assertEqual(7, len(partiallyDirectedGraph.edges()))

        # should not orient because X-Z is not undirected
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z', 'W'])
        edgePairs = [('X', 'Y'), ('Y', 'X'), ('X', 'Z'), ('X', 'W'), ('W', 'X'), ('Z', 'Y'), ('W', 'Y')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        didOrient = EdgeOrientation.applyMR3(partiallyDirectedGraph)
        self.assertFalse(didOrient)
        self.assertEqual(7, len(partiallyDirectedGraph.edges()))

        # should not orient because X-W is not undirected
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z', 'W'])
        edgePairs = [('X', 'Y'), ('Y', 'X'), ('X', 'Z'), ('Z', 'X'), ('W', 'X'), ('Z', 'Y'), ('W', 'Y')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        didOrient = EdgeOrientation.applyMR3(partiallyDirectedGraph)
        self.assertFalse(didOrient)
        self.assertEqual(7, len(partiallyDirectedGraph.edges()))

        # should not orient because both X-Z and X-W are not undirected
        partiallyDirectedGraph = nx.DiGraph()
        partiallyDirectedGraph.add_nodes_from(['X', 'Y', 'Z', 'W'])
        edgePairs = [('X', 'Y'), ('Y', 'X'), ('Z', 'X'), ('W', 'X'), ('Z', 'Y'), ('W', 'Y')]
        partiallyDirectedGraph.add_edges_from(edgePairs)
        didOrient = EdgeOrientation.applyMR3(partiallyDirectedGraph)
        self.assertFalse(didOrient)
        self.assertEqual(6, len(partiallyDirectedGraph.edges()))


if __name__ == '__main__':
    unittest.main()
