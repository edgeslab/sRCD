import pdb
import logging
import unittest

import networkx as nx

from mock import MagicMock
from mock import PropertyMock

from causality.learning import FCIEdgeOrientation
from causality.model.RelationalDependency import RelationalDependency
from causality.model.Schema import Schema
from causality.model.Model import Model
from causality.citest.CITest import Oracle
from causality.learning.RelFCI import RelFCI

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)



class TestFCIEdgeOrientation(unittest.TestCase):


    def _makeSingleEntityAGGWithDeps(self, relDeps, skeleton=True, orient=True, returnAlgo=False):
        schema = Schema()
        schema.addEntity('A')

        attNames = set()
        for relDep in relDeps:
            a,b = relDep.split('->')
            attNames.add(a.strip().split('.')[-1])
            attNames.add(b.strip().split('.')[-1])

        schema.addAttributes('A', attNames)
        model = Model(schema, relDeps)
        mockOracle = MagicMock(wraps=Oracle(model))
        mockModelProperty = PropertyMock()
        type(mockOracle).model = mockModelProperty
        relfci = RelFCI(schema, mockOracle, hopThreshold=2, depth=0)
        if skeleton:
            relfci.identifyUndirectedDependencies()
            relfci.finalizeUndirectedDependencies()
        if orient:
            relfci.orientDependencies()
        if returnAlgo:
            return relfci
        return relfci.perspectiveToAgg['A']


    def _orient(self, aggs, x, y, full=False, isEffect=True):
        toMark = RelationalDependency.TAIL_MARK_RIGHT_ARROW if isEffect else RelationalDependency.TAIL_MARK_EMPTY
        fromMark = RelationalDependency.TAIL_MARK_LEFT_ARROW if isEffect else RelationalDependency.TAIL_MARK_EMPTY

        relDepXY = FCIEdgeOrientation._getRelDep(aggs['A'], x, y)
        FCIEdgeOrientation.propagateEdgeOrientation(aggs, relDepXY, recurse=True, isTo=isEffect, mark=toMark)
        
        relDepYX = FCIEdgeOrientation._getRelDep(aggs['A'], y, x)
        FCIEdgeOrientation.propagateEdgeOrientation(aggs, relDepYX, recurse=True, isTo=not isEffect, mark=fromMark)


    def _orientFull(self, aggs, x, y):
        relDepXY = FCIEdgeOrientation._getRelDep(aggs['A'], x, y)
        FCIEdgeOrientation.propagateEdgeOrientation(aggs, relDepXY, recurse=True)
        relDepXY = FCIEdgeOrientation._getRelDep(aggs['A'], x, y)
        FCIEdgeOrientation.propagateEdgeOrientation(aggs, relDepXY, recurse=True, isTo=False, mark=RelationalDependency.TAIL_MARK_EMPTY)

        relDepYX = FCIEdgeOrientation._getRelDep(aggs['A'], y, x)
        FCIEdgeOrientation.propagateEdgeRemoval(aggs, relDepYX, recurse=True)


    def _orientUndirected(self, aggs, x, y):
        relDepXY = FCIEdgeOrientation._getRelDep(aggs['A'], x, y)
        FCIEdgeOrientation.propagateEdgeOrientation(aggs, relDepXY, recurse=True, isTo=True, mark=RelationalDependency.TAIL_MARK_EMPTY)
        
        relDepYX = FCIEdgeOrientation._getRelDep(aggs['A'], y, x)
        FCIEdgeOrientation.propagateEdgeOrientation(aggs, relDepYX, recurse=True, isTo=False, mark=RelationalDependency.TAIL_MARK_EMPTY)
        
        relDepXY = FCIEdgeOrientation._getRelDep(aggs['A'], x, y)
        FCIEdgeOrientation.propagateEdgeOrientation(aggs, relDepXY, recurse=True, isTo=False, mark=RelationalDependency.TAIL_MARK_EMPTY)
        
        relDepYX = FCIEdgeOrientation._getRelDep(aggs['A'], y, x)
        FCIEdgeOrientation.propagateEdgeOrientation(aggs, relDepYX, recurse=True, isTo=True, mark=RelationalDependency.TAIL_MARK_EMPTY)
        
    
    def testIsUncoveredPath(self):
        graph = nx.DiGraph()
        edgePairs = [(0, 1), (1, 2), (2, 3), (3, 4)]
        graph.add_edges_from(edgePairs)
        
        self.assertTrue(FCIEdgeOrientation._isUncoveredPath(graph, [0, 1]))
        self.assertTrue(FCIEdgeOrientation._isUncoveredPath(graph, [0, 1, 2]))
        self.assertTrue(FCIEdgeOrientation._isUncoveredPath(graph, [0, 1, 2, 3]))
        self.assertTrue(FCIEdgeOrientation._isUncoveredPath(graph, [0, 1, 2, 3, 4]))

        # make shielded tripple forward
        graph.add_edges_from([(0, 2)])

        self.assertFalse(FCIEdgeOrientation._isUncoveredPath(graph, [0, 1, 2]))
        self.assertTrue(FCIEdgeOrientation._isUncoveredPath(graph, [1, 2, 3, 4]))

        # make shielded tripple forward
        graph.add_edges_from([(4, 2)])
        
        self.assertFalse(FCIEdgeOrientation._isUncoveredPath(graph, [2, 3, 4]))
        self.assertTrue(FCIEdgeOrientation._isUncoveredPath(graph, [1, 2, 3]))


    def testIsPotentiatllyDirectedPath(self):
        #Regular PD path
        agg = self._makeSingleEntityAGGWithDeps(['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].Z -> [A].W'])
        x, y, z, w = agg.getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        self.assertTrue(FCIEdgeOrientation._isPotentiatllyDirectedPath(agg, [x, y]))
        self.assertTrue(FCIEdgeOrientation._isPotentiatllyDirectedPath(agg, [x, y, z, w]))

        #PD path with shielded triple
        agg = self._makeSingleEntityAGGWithDeps(['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].X -> [A].Z', '[A].Z -> [A].W'])
        x, y, z, w = agg.getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        self.assertTrue(FCIEdgeOrientation._isPotentiatllyDirectedPath(agg, [x, y]))
        self.assertTrue(FCIEdgeOrientation._isPotentiatllyDirectedPath(agg, [x, y, z, w]))

        #Not a PD path
        agg = self._makeSingleEntityAGGWithDeps(['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].Z -> [A].W', '[A].W -> [A].Z',])
        x, y, z, w = agg.getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])        
        self.assertTrue(FCIEdgeOrientation._isPotentiatllyDirectedPath(agg, [x, y, z]))
        self.assertFalse(FCIEdgeOrientation._isPotentiatllyDirectedPath(agg, [x, y, z, w]))

        agg = self._makeSingleEntityAGGWithDeps(['[A].X -> [A].Y', '[A].Z -> [A].Y', '[A].Z -> [A].W'])
        x, y, z, w = agg.getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        self.assertFalse(FCIEdgeOrientation._isPotentiatllyDirectedPath(agg, [x, y, z]))


    def testIsCirclePath(self):
        #Regular PD path but not circle path
        agg = self._makeSingleEntityAGGWithDeps(['[A].X -> [A].Y', '[A].Z -> [A].Y', '[A].Y -> [A].W'])
        x, y, z, w = agg.getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        self.assertFalse(FCIEdgeOrientation._isCirclePath(agg, [x, y, w]))

        #Not a PD path and not circle path
        self.assertFalse(FCIEdgeOrientation._isCirclePath(agg, [x, y, z, w]))

        #Regular circle path
        agg = self._makeSingleEntityAGGWithDeps(['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].Z -> [A].W'])
        x, y, z, w = agg.getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        self.assertTrue(FCIEdgeOrientation._isCirclePath(agg, [x, y, z]))


    def testIsDiscriminatingPath(self):
        # Regular PD path but not discriminating path
        agg = self._makeSingleEntityAGGWithDeps(['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].Z -> [A].W'])
        x, y, z, w = agg.getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        self.assertFalse(FCIEdgeOrientation._isDiscriminatingPath(agg, [x, y, z, w], z))

        #discriminating path
        agg = self._makeSingleEntityAGGWithDeps(['[A].X -> [A].Y', '[A].Z -> [A].Y', '[A].Z -> [A].W', '[A].Y -> [A].W'])
        x, y, z, w = agg.getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        
        relDepXW = FCIEdgeOrientation._getRelDep(agg, x, w)
        FCIEdgeOrientation.propagateEdgeRemoval({'A': agg}, relDepXW, recurse=True)
        relDepWX = FCIEdgeOrientation._getRelDep(agg, w, x)
        FCIEdgeOrientation.propagateEdgeRemoval({'A': agg}, relDepWX, recurse=True)
        self._orient({'A': agg}, y, w)

        self.assertTrue(FCIEdgeOrientation._isDiscriminatingPath(agg, [x, y, z, w], z))

        #almost discriminating path - 1
        agg = self._makeSingleEntityAGGWithDeps(['[A].X -> [A].Y', '[A].Z -> [A].Y', '[A].Z -> [A].W', '[A].Y -> [A].W', '[A].X -> [A].W'])
        x, y, z, w = agg.getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        self.assertFalse(FCIEdgeOrientation._isDiscriminatingPath(agg, [x, y, z, w], z))

        #almost discriminating path - 2
        agg = self._makeSingleEntityAGGWithDeps(['[A].X -> [A].Y', '[A].Z -> [A].Y', '[A].Z -> [A].W', '[A].Y -> [A].W'])
        x, y, z, w = agg.getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        self.assertFalse(FCIEdgeOrientation._isDiscriminatingPath(agg, [x, y, z, w], z))


    def testColliderDetection(self):
        # should orient X->Z<-Y
        relDeps = ['[A].X -> [A].Z', '[A].Y -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=False, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        algo.setSepsets({})
        didOrient = FCIEdgeOrientation.applyColliderDetection(algo)
        self.assertTrue(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(4, len(algo.orientedDependencies))

        # should orient X->Z<-Y, even though X and Y have a sepset (W), it doesn't include Z
        relDeps = ['[A].X -> [A].Z', '[A].Y -> [A].Z', '[A].Z -> [A].W']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=False, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        algo.setSepsets({('[A].X', '[A].Y'): {'[A].W'}, ('[A].Y', '[A].X'): {'[A].W'}})
        didOrient = FCIEdgeOrientation.applyColliderDetection(algo)
        self.assertTrue(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(4, len(algo.orientedDependencies))

        # should orient no edges (Z is in sepset(X, Y))
        relDeps = ['[A].X -> [A].Z', '[A].Y -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=False, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        algo.setSepsets({('[A].X', '[A].Y'): {'[A].Z'}, ('[A].Y', '[A].X'): {'[A].Z'}})
        didOrient = FCIEdgeOrientation.applyColliderDetection(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(0, len(algo.orientedDependencies))

        # should orient no edges because X -> Y
        relDeps = ['[A].X -> [A].Z', '[A].Y -> [A].Z', '[A].X -> [A].Y']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=False, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        algo.setSepsets({})
        didOrient = FCIEdgeOrientation.applyColliderDetection(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(0, len(algo.orientedDependencies))

        # should orient no edges because X <- Y
        relDeps = ['[A].X -> [A].Z', '[A].Y -> [A].Z', '[A].Y -> [A].X']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=False, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        algo.setSepsets({})
        didOrient = FCIEdgeOrientation.applyColliderDetection(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(0, len(algo.orientedDependencies))

        # should orient no edges because already oriented
        relDeps = ['[A].X -> [A].Z', '[A].Y -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=False, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        algo.setSepsets({})
        FCIEdgeOrientation.applyColliderDetection(algo)
        didOrient = FCIEdgeOrientation.applyColliderDetection(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(4, len(algo.orientedDependencies))

        # should orient X->Z even though Z<-Y already oriented
        relDeps = ['[A].X -> [A].Z', '[A].Y -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=False, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        algo.setSepsets({})
        
        y, z = algo.perspectiveToAgg['A'].getNodesByName(['[A].Y', '[A].Z'])
        self._orient(algo.perspectiveToAgg, y, z)
        
        didOrient = FCIEdgeOrientation.applyColliderDetection(algo)
        self.assertTrue(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(4, len(algo.orientedDependencies))

        # should orient X->Z<-Y and Y->Z<-W
        relDeps = ['[A].X -> [A].Z', '[A].Y -> [A].Z', '[A].W -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=False, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        algo.setSepsets({})
        didOrient = FCIEdgeOrientation.applyColliderDetection(algo)
        self.assertTrue(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(6, len(algo.orientedDependencies))


    def testKnownNonColliders(self):
        # should orient X->Y->Z
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=False, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)

        x,y = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y'])
        self._orient(algo.perspectiveToAgg, x, y)

        didOrient = FCIEdgeOrientation.applyKnownNonColliders(algo)
        self.assertTrue(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(3, len(algo.orientedDependencies))

        # should orient no edges because X and Y have an undirected edge
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=False, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        didOrient = FCIEdgeOrientation.applyKnownNonColliders(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(0, len(algo.orientedDependencies))

        # should orient no edges because X -> Z
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].X -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=False, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)

        x,y = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y'])
        self._orient(algo.perspectiveToAgg, x, y)

        didOrient = FCIEdgeOrientation.applyKnownNonColliders(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(2, len(algo.orientedDependencies))
    

        # should orient no edges because X <- Z
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].Z -> [A].X']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=False, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)

        x,y = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y'])
        self._orient(algo.perspectiveToAgg, x, y)

        didOrient = FCIEdgeOrientation.applyKnownNonColliders(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(2, len(algo.orientedDependencies))

        # should orient no edges because it already oriented Y->Z
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=False, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)

        x,y = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y'])
        self._orient(algo.perspectiveToAgg, x, y)

        y,z = algo.perspectiveToAgg['A'].getNodesByName(['[A].Y', '[A].Z'])
        self._orientFull(algo.perspectiveToAgg, y, z)

        didOrient = FCIEdgeOrientation.applyKnownNonColliders(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(3, len(algo.orientedDependencies))


    def testCycleAvoidance(self):
        # should orient X->Z
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].X -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=False, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)

        x,y,z = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z'])
        self._orientFull(algo.perspectiveToAgg, x, y)
        self._orient(algo.perspectiveToAgg, y, z)

        didOrient = FCIEdgeOrientation.applyCycleAvoidance(algo)
        self.assertTrue(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(5, len(algo.orientedDependencies))

        # should orient no edges if two are unoriented in triple
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].X -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=False, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)

        x,y,z = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z'])
        self._orient(algo.perspectiveToAgg, x, y)
        self._orient(algo.perspectiveToAgg, y, z)

        didOrient = FCIEdgeOrientation.applyCycleAvoidance(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(4, len(algo.orientedDependencies))

        # should orient no edges if X and Z are not adjacent (unshieled triple)
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=False, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)

        x,y,z = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z'])
        self._orientFull(algo.perspectiveToAgg, x, y)
        self._orient(algo.perspectiveToAgg, y, z)

        didOrient = FCIEdgeOrientation.applyCycleAvoidance(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(3, len(algo.orientedDependencies))

        # should orient no edges because X already oriented to Z
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].X -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=False, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)

        x,y,z = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z'])
        self._orientFull(algo.perspectiveToAgg, x, y)
        self._orient(algo.perspectiveToAgg, y, z)
        self._orient(algo.perspectiveToAgg, x, z)

        didOrient = FCIEdgeOrientation.applyCycleAvoidance(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(5, len(algo.orientedDependencies))

        # should orient no edges because common cause (no cycle to avoid for a shielded triple)
        relDeps = ['[A].Y -> [A].X', '[A].Y -> [A].Z', '[A].X -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=False, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)

        x,y,z = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z'])
        self._orientFull(algo.perspectiveToAgg, y, x)
        self._orient(algo.perspectiveToAgg, y, z)

        didOrient = FCIEdgeOrientation.applyCycleAvoidance(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(3, len(algo.orientedDependencies))

        # should orient no edges because common effect (no cycle to avoid for a shielded triple)
        relDeps = ['[A].X -> [A].Y', '[A].Z -> [A].Y', '[A].X -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=False, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)

        x,y,z = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z'])
        self._orientFull(algo.perspectiveToAgg, x, y)
        self._orient(algo.perspectiveToAgg, z, y)

        didOrient = FCIEdgeOrientation.applyCycleAvoidance(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(3, len(algo.orientedDependencies))


    def testMR3(self):
        # should orient W->Y
        relDeps = ['[A].X -> [A].Y', '[A].Z -> [A].Y', '[A].X -> [A].W', '[A].Z -> [A].W', '[A].W -> [A].Y']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=False, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)

        x,y,z = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z'])
        self._orient(algo.perspectiveToAgg, x, y)
        self._orient(algo.perspectiveToAgg, z, y)

        didOrient = FCIEdgeOrientation.applyMR3(algo)
        self.assertTrue(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(6, len(algo.orientedDependencies))

        # should orient no edges
        relDeps = ['[A].X -> [A].W', '[A].Z -> [A].W', '[A].X -> [A].Z', '[A].X -> [A].Y', '[A].Z -> [A].Y']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=False, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)

        x,y,z = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z'])
        self._orient(algo.perspectiveToAgg, x, y)
        self._orient(algo.perspectiveToAgg, z, y)

        didOrient = FCIEdgeOrientation.applyMR3(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(4, len(algo.orientedDependencies))

        # should orient no edges because already oriented W->Y
        relDeps = ['[A].X -> [A].Y', '[A].Z -> [A].Y', '[A].X -> [A].W', '[A].Z -> [A].W', '[A].W -> [A].Y']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=False, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)

        x,y,z,w = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        self._orient(algo.perspectiveToAgg, x, y)
        self._orient(algo.perspectiveToAgg, z, y)
        self._orient(algo.perspectiveToAgg, w, y)

        didOrient = FCIEdgeOrientation.applyMR3(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(6, len(algo.orientedDependencies))

        # should orient no edges because X-W not undirected
        relDeps = ['[A].X -> [A].Y', '[A].Z -> [A].Y', '[A].X -> [A].W', '[A].Z -> [A].W', '[A].W -> [A].Y']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=False, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)

        x,y,z,w = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        self._orient(algo.perspectiveToAgg, x, y)
        self._orient(algo.perspectiveToAgg, z, y)
        self._orient(algo.perspectiveToAgg, x, w)

        didOrient = FCIEdgeOrientation.applyMR3(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(6, len(algo.orientedDependencies))

        # should orient no edges because Z-W not undirected
        relDeps = ['[A].X -> [A].Y', '[A].Z -> [A].Y', '[A].X -> [A].W', '[A].Z -> [A].W', '[A].W -> [A].Y']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=False, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)

        x,y,z,w = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        self._orient(algo.perspectiveToAgg, x, y)
        self._orient(algo.perspectiveToAgg, z, y)
        self._orient(algo.perspectiveToAgg, z, w)

        didOrient = FCIEdgeOrientation.applyMR3(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(6, len(algo.orientedDependencies))

        # should orient no edges because X-W and Z-W not undirected
        relDeps = ['[A].X -> [A].Y', '[A].Z -> [A].Y', '[A].X -> [A].W', '[A].Z -> [A].W', '[A].W -> [A].Y']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=False, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)

        x,y,z,w = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        self._orient(algo.perspectiveToAgg, x, y)
        self._orient(algo.perspectiveToAgg, z, y)
        self._orient(algo.perspectiveToAgg, x, w)
        self._orient(algo.perspectiveToAgg, z, w)

        didOrient = FCIEdgeOrientation.applyMR3(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(8, len(algo.orientedDependencies))
    

    def testFCIR4(self):
        # should orient Z->W
        relDeps = ['[A].X -> [A].Y', '[A].Z -> [A].Y', '[A].Z -> [A].W', '[A].Y -> [A].W']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)

        x, y, z, w = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        algo.setSepsets({('[A].X', '[A].W'): {'[A].Z'}, ('[A].W', '[A].X'): {'[A].Z'}})
        self._orient(algo.perspectiveToAgg, x, y)
        self._orient(algo.perspectiveToAgg, z, y)
        self._orient(algo.perspectiveToAgg, z, w)
        self._orient(algo.perspectiveToAgg, y, w)

        didOrient = FCIEdgeOrientation.applyFCIR4(algo)
        self.assertTrue(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(7, len(algo.orientedDependencies))


        # should not orient Z->W since Z not in sepset(X,W)
        relDeps = ['[A].X -> [A].Y', '[A].Z -> [A].Y', '[A].Z -> [A].W', '[A].Y -> [A].W']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)

        x, y, z, w = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        # algo.setSepsets({('[A].X', '[A].W'): {'[A].Z'}, ('[A].W', '[A].X'): {'[A].Z'}})
        self._orient(algo.perspectiveToAgg, x, y)
        self._orient(algo.perspectiveToAgg, z, y)
        self._orient(algo.perspectiveToAgg, z, w)
        self._orient(algo.perspectiveToAgg, y, w)

        didOrient = FCIEdgeOrientation.applyFCIR4(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(8, len(algo.orientedDependencies))


        # should not orient Z->W since Y not a parent of W
        relDeps = ['[A].X -> [A].Y', '[A].Z -> [A].Y', '[A].Z -> [A].W']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)

        x, y, z, w = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        algo.setSepsets({('[A].X', '[A].W'): {'[A].Z'}, ('[A].W', '[A].X'): {'[A].Z'}})
        self._orient(algo.perspectiveToAgg, x, y)
        self._orient(algo.perspectiveToAgg, z, y)
        self._orient(algo.perspectiveToAgg, z, w)

        didOrient = FCIEdgeOrientation.applyFCIR4(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(6, len(algo.orientedDependencies))


        # should not orient Z->W since it is already oriented
        relDeps = ['[A].X -> [A].Y', '[A].Z -> [A].Y', '[A].Z -> [A].W', '[A].Y -> [A].W']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)

        x, y, z, w = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        algo.setSepsets({('[A].X', '[A].W'): {'[A].Z'}, ('[A].W', '[A].X'): {'[A].Z'}})
        self._orient(algo.perspectiveToAgg, x, y)
        self._orient(algo.perspectiveToAgg, z, y)
        self._orientFull(algo.perspectiveToAgg, z, w)
        self._orient(algo.perspectiveToAgg, y, w)

        didOrient = FCIEdgeOrientation.applyFCIR4(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(7, len(algo.orientedDependencies))


        # should not orient Z->W since Y not a collider
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].Z -> [A].W', '[A].Y -> [A].W']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)

        x, y, z, w = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        algo.setSepsets({('[A].X', '[A].W'): {'[A].Z'}, ('[A].W', '[A].X'): {'[A].Z'}})
        self._orient(algo.perspectiveToAgg, x, y)
        self._orient(algo.perspectiveToAgg, y, z)
        self._orient(algo.perspectiveToAgg, z, w)
        self._orient(algo.perspectiveToAgg, y, w)

        didOrient = FCIEdgeOrientation.applyFCIR4(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(8, len(algo.orientedDependencies))

    
    def testFCIR5(self):
        # should orient X-Y, Y-Z, Z-W, X-W
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].Z -> [A].W', '[A].X -> [A].W']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        didOrient = FCIEdgeOrientation.applyFCIR5(algo)
        self.assertTrue(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(8, len(algo.orientedDependencies))

        # should not orient any edges since X-W not adjacent
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].Z -> [A].W']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        didOrient = FCIEdgeOrientation.applyFCIR5(algo)
        self.assertFalse(didOrient)

        # should not orient any edges since X-Y-Z-W not circle path, but uncovered
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].Z -> [A].W']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        x, y, z, w = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        self._orient(algo.perspectiveToAgg, y, z)
        didOrient = FCIEdgeOrientation.applyFCIR5(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(2, len(algo.orientedDependencies))

        # should not orient any edges since X-Y-Z-W not uncovered, but circle path
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].Z -> [A].W', '[A].X -> [A].W', '[A].X -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        didOrient = FCIEdgeOrientation.applyFCIR5(algo)
        self.assertFalse(didOrient)

        # should not orient any edges since X->W already oriented
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].Z -> [A].W', '[A].X -> [A].W', '[A].X -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        x, y, z, w = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        self._orient(algo.perspectiveToAgg, x, w)
        didOrient = FCIEdgeOrientation.applyFCIR5(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(2, len(algo.orientedDependencies))


    def testFCIR6(self):
        # should orient Y-*Z
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        x, y, z = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z'])
        self._orientUndirected(algo.perspectiveToAgg, x, y)
        didOrient = FCIEdgeOrientation.applyFCIR6(algo)
        self.assertTrue(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(4, len(algo.orientedDependencies))

        # should orient Y-*Z even if X-Z adjacent
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].X -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        x, y, z = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z'])
        self._orientUndirected(algo.perspectiveToAgg, x, y)
        didOrient = FCIEdgeOrientation.applyFCIR6(algo)
        self.assertTrue(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(6, len(algo.orientedDependencies))

        # should orient Y-*Z even if Yo->Z oriented
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        x, y, z = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z'])
        self._orientUndirected(algo.perspectiveToAgg, x, y)
        self._orient(algo.perspectiveToAgg, y, z)
        didOrient = FCIEdgeOrientation.applyFCIR6(algo)
        self.assertTrue(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(4, len(algo.orientedDependencies))

        # should not orient Y-*Z since X-Y not undirected
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        x, y, z = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z'])
        self._orient(algo.perspectiveToAgg, x, y)
        didOrient = FCIEdgeOrientation.applyFCIR6(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(2, len(algo.orientedDependencies))

        # should not orient Y-*Z since Y->Z already oriented
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        x, y, z = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z'])
        self._orient(algo.perspectiveToAgg, x, y)
        self._orientFull(algo.perspectiveToAgg, y, z)
        didOrient = FCIEdgeOrientation.applyFCIR6(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(3, len(algo.orientedDependencies))


    def testFCIR7(self):
        # should orient Y-*Z
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        x, y, z = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z'])
        self._orient(algo.perspectiveToAgg, x, y, isEffect=False)
        didOrient = FCIEdgeOrientation.applyFCIR7(algo)
        self.assertTrue(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(4, len(algo.orientedDependencies))

        # should not orient Y-*Z since X-Z adjacent
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].X -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        x, y, z = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z'])
        self._orient(algo.perspectiveToAgg, x, y, isEffect=False)
        didOrient = FCIEdgeOrientation.applyFCIR7(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(2, len(algo.orientedDependencies))

        # should not orient Y-*Z since X-Y undirected
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].X -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        didOrient = FCIEdgeOrientation.applyFCIR7(algo)
        self.assertFalse(didOrient)

        # should not orient Y-*Z since Y->Z already oriented
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].X -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        x, y, z = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z'])
        self._orientFull(algo.perspectiveToAgg, x, y)
        didOrient = FCIEdgeOrientation.applyFCIR7(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(1, len(algo.orientedDependencies))


    def testFCIR8(self):
        # should orient X->Z
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].X -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        x, y, z = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z'])
        self._orientFull(algo.perspectiveToAgg, x, y)
        self._orientFull(algo.perspectiveToAgg, y, z)
        self._orient(algo.perspectiveToAgg, x, z)
        didOrient = FCIEdgeOrientation.applyFCIR8(algo)
        self.assertTrue(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(3, len(algo.orientedDependencies))

        # should orient X->Z even if X-oY not fully oriented
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].X -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        x, y, z = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z'])
        self._orient(algo.perspectiveToAgg, x, y, isEffect=False)
        self._orientFull(algo.perspectiveToAgg, y, z)
        self._orient(algo.perspectiveToAgg, x, z)
        didOrient = FCIEdgeOrientation.applyFCIR8(algo)
        self.assertTrue(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(4, len(algo.orientedDependencies))

        # should not orient X->Z since Yo->Z not fully oriented
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].X -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        x, y, z = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z'])
        self._orientFull(algo.perspectiveToAgg, x, y)
        self._orient(algo.perspectiveToAgg, y, z)
        self._orient(algo.perspectiveToAgg, x, z)
        didOrient = FCIEdgeOrientation.applyFCIR8(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(5, len(algo.orientedDependencies))

        # should not orient X->Z since Yo-oZ not fully oriented
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].X -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        x, y, z = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z'])
        self._orientFull(algo.perspectiveToAgg, x, y)
        self._orient(algo.perspectiveToAgg, x, z)
        didOrient = FCIEdgeOrientation.applyFCIR8(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(3, len(algo.orientedDependencies))

        # should not orient X->Z since Xo->Y not fully oriented
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].X -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        x, y, z = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z'])
        self._orient(algo.perspectiveToAgg, x, y)
        self._orientFull(algo.perspectiveToAgg, y, z)
        self._orient(algo.perspectiveToAgg, x, z)
        didOrient = FCIEdgeOrientation.applyFCIR8(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(5, len(algo.orientedDependencies))

        # should not orient X->Z since X->Z already oriented
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].X -> [A].Z']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        x, y, z = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z'])
        self._orientFull(algo.perspectiveToAgg, x, y)
        self._orientFull(algo.perspectiveToAgg, y, z)
        self._orientFull(algo.perspectiveToAgg, x, z)
        didOrient = FCIEdgeOrientation.applyFCIR8(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(3, len(algo.orientedDependencies))


    def testFCIR9(self):
        # should orient X->W
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].Z -> [A].W', '[A].X -> [A].W']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        x, y, z, w = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        self._orient(algo.perspectiveToAgg, x, w)
        didOrient = FCIEdgeOrientation.applyFCIR9(algo)
        self.assertTrue(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(1, len(algo.orientedDependencies))

        # should orient X->W even if all the other edges are fully oriented
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].Z -> [A].W', '[A].X -> [A].W']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        x, y, z, w = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        self._orientFull(algo.perspectiveToAgg, x, y)
        self._orientFull(algo.perspectiveToAgg, y, z)
        self._orientFull(algo.perspectiveToAgg, z, w)
        self._orient(algo.perspectiveToAgg, x, w)
        didOrient = FCIEdgeOrientation.applyFCIR9(algo)
        self.assertTrue(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(4, len(algo.orientedDependencies))

        # should not orient X->W since not a pd path, but uncovered
        relDeps = ['[A].X -> [A].Y', '[A].Z -> [A].Y', '[A].Z -> [A].W', '[A].X -> [A].W']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        x, y, z, w = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        self._orient(algo.perspectiveToAgg, x, w)
        self._orient(algo.perspectiveToAgg, z, y)
        didOrient = FCIEdgeOrientation.applyFCIR9(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(4, len(algo.orientedDependencies))

        # should not orient X->W since not a uncovered path, but pd path
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].Z -> [A].W', '[A].X -> [A].W', '[A].Y -> [A].W']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        x, y, z, w = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        self._orient(algo.perspectiveToAgg, x, w)
        didOrient = FCIEdgeOrientation.applyFCIR9(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(2, len(algo.orientedDependencies))

        # should orient X->W since already oriented
        relDeps = ['[A].X -> [A].Y', '[A].Y -> [A].Z', '[A].Z -> [A].W', '[A].X -> [A].W']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        x, y, z, w = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        self._orientFull(algo.perspectiveToAgg, x, w)
        didOrient = FCIEdgeOrientation.applyFCIR9(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(1, len(algo.orientedDependencies))

    
    def testFCIR10(self):
        # should orient X->Z
        relDeps = ['[A].X -> [A].Z', '[A].Y -> [A].Z', '[A].W -> [A].Z', '[A].X -> [A].Y', '[A].X -> [A].W']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        x, y, z, w = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        self._orient(algo.perspectiveToAgg, x, z)
        self._orientFull(algo.perspectiveToAgg, y, z)
        self._orientFull(algo.perspectiveToAgg, w, z)
        didOrient = FCIEdgeOrientation.applyFCIR10(algo)
        self.assertTrue(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(3, len(algo.orientedDependencies))

        # should orient X->Z with longer p1/p2 paths
        relDeps = ['[A].X -> [A].Z', '[A].Y -> [A].Z', '[A].W -> [A].Z', '[A].X -> [A].P1', '[A].P1 -> [A].Y', '[A].X -> [A].P2', '[A].P2 -> [A].W']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        x, y, z, w = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        self._orient(algo.perspectiveToAgg, x, z)
        self._orientFull(algo.perspectiveToAgg, y, z)
        self._orientFull(algo.perspectiveToAgg, w, z)
        didOrient = FCIEdgeOrientation.applyFCIR10(algo)
        self.assertTrue(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(3, len(algo.orientedDependencies))

        # should not orient X->Z since p1 not a pd path 
        relDeps = ['[A].X -> [A].Z', '[A].Y -> [A].Z', '[A].W -> [A].Z', '[A].Y -> [A].X', '[A].X -> [A].W']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        x, y, z, w = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        self._orient(algo.perspectiveToAgg, x, z)
        self._orientFull(algo.perspectiveToAgg, y, z)
        self._orientFull(algo.perspectiveToAgg, w, z)
        self._orientFull(algo.perspectiveToAgg, y, x)
        didOrient = FCIEdgeOrientation.applyFCIR10(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(5, len(algo.orientedDependencies))

        # should not orient X->Z since p2 not a pd path 
        relDeps = ['[A].X -> [A].Z', '[A].Y -> [A].Z', '[A].W -> [A].Z', '[A].X -> [A].Y', '[A].W -> [A].X']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        x, y, z, w = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        self._orient(algo.perspectiveToAgg, x, z)
        self._orientFull(algo.perspectiveToAgg, y, z)
        self._orientFull(algo.perspectiveToAgg, w, z)
        self._orientFull(algo.perspectiveToAgg, w, x)
        didOrient = FCIEdgeOrientation.applyFCIR10(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(5, len(algo.orientedDependencies))

        # should not orient X->Z since already oriented
        relDeps = ['[A].X -> [A].Z', '[A].Y -> [A].Z', '[A].W -> [A].Z', '[A].X -> [A].Y', '[A].X -> [A].W']
        algo = self._makeSingleEntityAGGWithDeps(relDeps, skeleton=True, orient=False, returnAlgo=True)
        algo.setUndirectedDependencies(relDeps)
        x, y, z, w = algo.perspectiveToAgg['A'].getNodesByName(['[A].X', '[A].Y', '[A].Z', '[A].W'])
        self._orientFull(algo.perspectiveToAgg, x, z)
        self._orientFull(algo.perspectiveToAgg, y, z)
        self._orientFull(algo.perspectiveToAgg, w, z)
        didOrient = FCIEdgeOrientation.applyFCIR10(algo)
        self.assertFalse(didOrient)
        algo.updateOrientedDependencies()
        self.assertEqual(3, len(algo.orientedDependencies))
    



if __name__ == '__main__':
    unittest.main()
