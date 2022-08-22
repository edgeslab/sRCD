import unittest
from causality.model.Schema import Attribute
from causality.model.Schema import Schema
from causality.test import TestUtil
from causality.model.Aggregator import IdentityAggregator, AverageAggregator, CountAggregator, NonRandomChoiceAggregator, CaseAggregator
from causality.test.testDataStoreSuite import skipOrInstantiateDsSubclass

class TestDataStoreRelationalPath(unittest.TestCase):

    dsSubclassInitTuple = None    # DataStore implementation. set by suite runner. None if called directly via IntelliJ


    def setUp(self):
        self.dataStore = skipOrInstantiateDsSubclass(self)
        self.schema = Schema()


    def tearDown(self):
        self.dataStore.close()


    def testSingletonRelationalPath(self):
        self.schema.addEntity('A')
        self.dataStore.addTable(self.schema, 'A')
        self.dataStore.insertRowsWithValues('A', ['id'], [[1]])
        aIds = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'A', [IdentityAggregator('[A].exists')]))
        self.assertEqual([(1, [1])], aIds)

        self.dataStore.insertRowsWithValues('A', ['id'], [[2]])
        aIds = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'A', [IdentityAggregator('[A].exists')]))
        aIds = [(idVal, tuple(val for val in vals)) for idVal, vals in aIds]
        TestUtil.assertUnorderedListEqual(self, [(1, (1,)), (2, (2,))], aIds)

        self.schema.addEntity('B')
        self.dataStore.addTable(self.schema, 'B')
        self.schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        self.dataStore.addTable(self.schema, 'AB')
        self.dataStore.insertRowsWithValues('AB', ['id'], [[0]])
        abIds = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'AB', [IdentityAggregator('[AB].exists')]))
        self.assertEqual([(0, [0])], abIds)

        self.dataStore.insertRowsWithValues('AB', ['id'], [[1]])
        abIds = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'AB', [IdentityAggregator('[AB].exists')]))
        abIds = [(idVal, tuple(val for val in vals)) for idVal, vals in abIds]
        TestUtil.assertUnorderedListEqual(self, [(0, (0,)), (1, (1,))], abIds)


    def testPairRelationalPath(self):
        self.schema.addEntity('A')
        self.schema.addEntity('B')
        self.schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        self.dataStore.addTable(self.schema, 'A')
        self.dataStore.insertRowsWithValues('A', ['id'], [[1], [2]])
        self.dataStore.addTable(self.schema, 'B')
        self.dataStore.addTable(self.schema, 'AB')
        self.dataStore.insertRowsWithValues('AB', ['id', 'A_id'], [[0, 1], [1, 2]])

        abIds = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'A', [IdentityAggregator('[A, AB].exists')]))
        abIds = [(idVal, tuple(val for val in vals)) for idVal, vals in abIds]
        TestUtil.assertUnorderedListEqual(self, [(1, (0,)), (2, (1,))], abIds)

        aIds = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'AB', [IdentityAggregator('[AB, A].exists')]))
        aIds = [(idVal, tuple(val for val in vals)) for idVal, vals in aIds]
        TestUtil.assertUnorderedListEqual(self, [(0, (1,)), (1, (2,))], aIds)


    def testTripleRelationalPath(self):
        self.schema.addEntity('A')
        self.schema.addEntity('B')
        self.schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        self.dataStore.addTable(self.schema, 'A')
        self.dataStore.insertRowsWithValues('A', ['id'], [[1], [2], [3]])
        self.dataStore.addTable(self.schema, 'B')
        self.dataStore.insertRowsWithValues('B', ['id'], [[4], [5], [6]])
        self.dataStore.addTable(self.schema, 'AB')
        self.dataStore.insertRowsWithValues('AB', ['id', 'A_id', 'B_id'], [[0, 1, 4], [1, 2, 6], [2, 3, 5]])

        bIds = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'A', [IdentityAggregator('[A, AB, B].exists')]))
        bIds = [(idVal, tuple(val for val in vals)) for idVal, vals in bIds]
        TestUtil.assertUnorderedListEqual(self, [(1, (4,)), (2, (6,)), (3, (5,))], bIds)

        aIds = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'B', [IdentityAggregator('[B, AB, A].exists')]))
        aIds = [(idVal, tuple(val for val in vals)) for idVal, vals in aIds]
        TestUtil.assertUnorderedListEqual(self, [(4, (1,)), (5, (3,)), (6, (2,))], aIds)


    def testManyTripleRelationalPath(self):
        self.schema.addEntity('A')
        self.schema.addEntity('B')
        self.schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.MANY))
        self.dataStore.addTable(self.schema, 'A')
        self.dataStore.insertRowsWithValues('A', ['id'], [[1], [2], [3]])
        self.dataStore.addTable(self.schema, 'B')
        self.dataStore.insertRowsWithValues('B', ['id'], [[4], [5], [6]])
        self.dataStore.addTable(self.schema, 'AB')
        self.dataStore.insertRowsWithValues('AB', ['id', 'A_id', 'B_id'], [[0, 1, 4], [1, 1, 5], [2, 2, 4], [3, 2, 6], [4, 3, 5]])

        bIds = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'A', [IdentityAggregator('[A, AB, B].exists')]))
        bIds = [(idVal, tuple(val for val in vals)) for idVal, vals in bIds]
        TestUtil.assertUnorderedListEqual(self, [(1, (4,)), (1, (5,)), (2, (4,)), (2, (6,)), (3, (5,))], bIds)

        aIds = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'B', [IdentityAggregator('[B, AB, A].exists')]))
        aIds = [(idVal, tuple(val for val in vals)) for idVal, vals in aIds]
        TestUtil.assertUnorderedListEqual(self, [(4, (1,)), (4, (2,)), (5, (1,)), (5, (3,)), (6, (2,))], aIds)


    def testBadEROrder(self):
        self.schema.addEntity('A')
        self.schema.addEntity('B')
        self.schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        self.dataStore.addTable(self.schema, 'A')
        self.dataStore.addTable(self.schema, 'B')
        self.dataStore.addTable(self.schema, 'AB')
        self.dataStore.insertRowsWithValues('A', ['id'], [[1]])
        self.dataStore.insertRowsWithValues('AB', ['id'], [[1]])

        method = self.dataStore.getValuesForRelVarAggrs(self.schema, 'A', [IdentityAggregator('[A, A].exists')])
        TestUtil.assertRaisesMessage(self, Exception, "invalid item1Name 'A' and item2Name 'A' in relational path: types must alternate between entities and relationships",
             method.__next__)

        method = self.dataStore.getValuesForRelVarAggrs(self.schema, 'AB', [IdentityAggregator('[AB, AB].exists')])
        TestUtil.assertRaisesMessage(self, Exception, "invalid item1Name 'AB' and item2Name 'AB' in relational path: types must alternate between entities and relationships",
             method.__next__)

        method = self.dataStore.getValuesForRelVarAggrs(self.schema, 'A', [IdentityAggregator('[A, AB, AB].exists')])
        TestUtil.assertRaisesMessage(self, Exception, "invalid item1Name 'AB' and item2Name 'AB' in relational path: types must alternate between entities and relationships",
             method.__next__)

        method = self.dataStore.getValuesForRelVarAggrs(self.schema, 'A', [IdentityAggregator('[A, AB, B, B].exists')])
        TestUtil.assertRaisesMessage(self, Exception, "invalid item1Name 'B' and item2Name 'B' in relational path: types must alternate between entities and relationships",
             method.__next__)


    def testNonExistingTables(self):
        method = self.dataStore.getValuesForRelVarAggrs(self.schema, 'A', [])
        TestUtil.assertRaisesMessage(self, Exception, "No table found for base item 'A'",
             method.__next__)

        method = self.dataStore.getValuesForRelVarAggrs(self.schema, 'B', [])
        TestUtil.assertRaisesMessage(self, Exception, "No table found for base item 'B'",
             method.__next__)

        self.schema.addEntity('A')
        self.dataStore.addTable(self.schema, 'A')
        self.dataStore.insertRowsWithValues('A', ['id'], [[1]])
        method = self.dataStore.getValuesForRelVarAggrs(self.schema, 'A', [IdentityAggregator('[A, AB].exists')])
        TestUtil.assertRaisesMessage(self, Exception, "No table found for item 'AB' in path ['A', 'AB']",
             method.__next__)


    def testBurningBridges(self):
        self.schema.addEntity('A')
        self.schema.addEntity('B')
        self.schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        self.dataStore.addTable(self.schema, 'A')
        self.dataStore.insertRowsWithValues('A', ['id'], [[1], [2], [3]])
        self.dataStore.addTable(self.schema, 'B')
        self.dataStore.insertRowsWithValues('B', ['id'], [[4], [5], [6]])
        self.dataStore.addTable(self.schema, 'AB')
        self.dataStore.insertRowsWithValues('AB', ['id', 'A_id', 'B_id'], [[0, 1, 4], [1, 1, 5], [2, 2, 4],
            [3, 2, 6], [4, 3, 5]])

        abIds = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'A', [IdentityAggregator('[A, AB, B, AB].exists')]))
        abIds = [(idVal, tuple(val for val in vals)) for idVal, vals in abIds]
        TestUtil.assertUnorderedListEqual(self, [(1, (2,)), (1, (4,)), (2, (0,)), (3, (1,))], abIds)

        abIds = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'B', [IdentityAggregator('[B, AB, A, AB].exists')]))
        abIds = [(idVal, tuple(val for val in vals)) for idVal, vals in abIds]
        TestUtil.assertUnorderedListEqual(self, [(4, (1,)), (4, (3,)), (5, (0,)), (6, (2,))], abIds)

        aIds = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'A', [IdentityAggregator('[A, AB, B, AB, A].exists')]))
        aIds = [(idVal, tuple(val for val in vals)) for idVal, vals in aIds]
        TestUtil.assertUnorderedListEqual(self, [(1, (2,)), (1, (3,)), (2, (1,)), (3, (1,))], aIds)

        bIds = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'B', [IdentityAggregator('[B, AB, A, AB, B].exists')]))
        bIds = [(idVal, tuple(val for val in vals)) for idVal, vals in bIds]
        TestUtil.assertUnorderedListEqual(self, [(4, (5,)), (4, (6,)), (5, (4,)), (6, (4,))], bIds)

        # test to make sure entity ids are burnt by using two different relationship classes
        self.schema.addRelationship('AB2', ('A', Schema.ONE), ('B', Schema.ONE))
        self.dataStore.addTable(self.schema, 'AB2')
        self.dataStore.insertRowsWithValues('AB2', ['id', 'A_id', 'B_id'], [[0, 1, 4], [1, 2, 4]])

        aIds = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'A', [IdentityAggregator('[A, AB, B, AB2, A].exists')]))
        aIds = [(idVal, tuple(val for val in vals)) for idVal, vals in aIds]
        TestUtil.assertUnorderedListEqual(self, [(1, (2,)), (2, (1,)), (3, (None,))], aIds)


    def testOneToOneRelationalVariableInstance(self):
        self.schema.addEntity('A')
        self.schema.addEntity('B')
        self.schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        self.schema.addAttribute('A', 'X', Attribute.FLOAT)
        self.schema.addAttribute('B', 'Y', Attribute.STRING)
        self.schema.addAttribute('AB', 'XY', Attribute.INTEGER)
        self.dataStore.addTable(self.schema, 'A')
        self.dataStore.insertRowsWithValues('A', ['id', 'X'], [[1, 2.5], [2, 1.5], [3, 1.0]])
        self.dataStore.addTable(self.schema, 'B')
        self.dataStore.insertRowsWithValues('B', ['id', 'Y'], [[4, 'red'], [5, 'yellow'], [6, 'blue']])
        self.dataStore.addTable(self.schema, 'AB')
        self.dataStore.insertRowsWithValues('AB', ['id', 'A_id', 'B_id', 'XY'],
            [[0, 1, 4, 2], [1, 2, 6, 9], [2, 3, 5, 2]])

        xVals = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'A', [IdentityAggregator('[A].X')]))
        xVals = [(idVal, tuple(val for val in vals)) for idVal, vals in xVals]
        TestUtil.assertUnorderedListEqual(self, [(1, (2.5,)), (2, (1.5,)), (3, (1.0,))], xVals)

        xyVals = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'AB', [IdentityAggregator('[AB].XY')]))
        xyVals = [(idVal, tuple(val for val in vals)) for idVal, vals in xyVals]
        TestUtil.assertUnorderedListEqual(self, [(0, (2,)), (1, (9,)), (2, (2,))], xyVals)

        yVals = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'A', [IdentityAggregator('[A, AB, B].Y')]))
        yVals = [(idVal, tuple(val for val in vals)) for idVal, vals in yVals]
        TestUtil.assertUnorderedListEqual(self, [(1, ('red',)), (2, ('blue',)), (3, ('yellow',))], yVals)


    def testManyToManyRelationalVariableInstance(self):
        self.schema.addEntity('A')
        self.schema.addEntity('B')
        self.schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        self.schema.addAttribute('B', 'Y', Attribute.INTEGER)
        self.dataStore.addTable(self.schema, 'A')
        self.dataStore.insertRowsWithValues('A', ['id'], [[1], [2], [3]])
        self.dataStore.addTable(self.schema, 'B')
        self.dataStore.insertRowsWithValues('B', ['id', 'Y'], [[4, 3], [5, 5], [6, 8], [7, 3]])
        self.dataStore.addTable(self.schema, 'AB')
        self.dataStore.insertRowsWithValues('AB', ['id', 'A_id', 'B_id'],
            [[0, 1, 4], [1, 1, 5], [2, 2, 4], [3, 2, 6], [4, 3, 5], [5, 2, 7]])

        yVals = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'A', [IdentityAggregator('[A, AB, B].Y')]))
        yVals = [(idVal, tuple(val for val in vals)) for idVal, vals in yVals]
        TestUtil.assertUnorderedListEqual(self, [(1, (3,)), (1, (5,)), (2, (3,)), (2, (8,)), (2, (3,)), (3, (5,))], yVals)


    def testMissingAttrForRelationalVariable(self):
        # entity or relationship class doesn't have attribute defined --> raise exception (e.g., [A].Y)
        self.schema.addEntity('A')
        self.dataStore.addTable(self.schema, 'A')
        self.dataStore.insertRowsWithValues('A', ['id'], [[1]])
        method = self.dataStore.getValuesForRelVarAggrs(self.schema, 'A', [IdentityAggregator('[A].Y')])
        TestUtil.assertRaisesMessage(self, Exception, "Attribute 'Y' does not exist for item 'A'",
             method.__next__)

        self.schema.addEntity('B')
        self.dataStore.addTable(self.schema, 'B')
        self.dataStore.insertRowsWithValues('B', ['id'], [[1]])
        self.schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        self.dataStore.addTable(self.schema, 'AB')
        self.dataStore.insertRowsWithValues('AB', ['id'], [[1]])
        method = self.dataStore.getValuesForRelVarAggrs(self.schema, 'AB', [IdentityAggregator('[AB].YZ')])
        TestUtil.assertRaisesMessage(self, Exception, "Attribute 'YZ' does not exist for item 'AB'",
            method.__next__)

        method = self.dataStore.getValuesForRelVarAggrs(self.schema, 'B', [IdentityAggregator('[B].X')])
        TestUtil.assertRaisesMessage(self, Exception, "Attribute 'X' does not exist for item 'B'",
             method.__next__)


    def testMissingAttrValsForRelationalVariable(self):
        # if entity or relationship instances don't have values for given attribute, then prune out missing values
        # from the returned terminal set. If all values are missing, then return None.
        self.schema.addEntity('A')
        self.schema.addAttribute('A', 'X', Attribute.INTEGER)
        self.dataStore.addTable(self.schema, 'A')
        self.dataStore.insertRowsWithValues('A', ['id'], [[1]])

        xVals = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'A', [IdentityAggregator('[A].X')]))
        xVals = [(idVal, tuple(val for val in vals)) for idVal, vals in xVals]
        TestUtil.assertUnorderedListEqual(self, [(1, (None,))], xVals)

        self.schema.addEntity('B')
        self.schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.ONE))
        self.schema.addAttribute('B', 'Y', Attribute.STRING)
        self.dataStore.addTable(self.schema, 'B')
        self.dataStore.insertRowsWithValues('B', ['id', 'Y'], [[1, 'red'], [3, 'blue']])
        self.dataStore.insertRowsWithValues('B', ['id'], [[2]])
        self.dataStore.addTable(self.schema, 'AB')
        self.dataStore.insertRowsWithValues('AB', ['id', 'A_id', 'B_id'], [[0, 1, 1], [1, 1, 2], [2, 1, 3]])

        yVals = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'A', [IdentityAggregator('[A, AB, B].Y')]))
        yVals = [(idVal, tuple(val for val in vals)) for idVal, vals in yVals]
        TestUtil.assertUnorderedListEqual(self, [(1, ('red',)), (1, ('blue',))], yVals)


    def testSingleNonIdentityAggregator(self):
        self.schema.addEntity('A')
        self.schema.addEntity('B')
        self.schema.addAttribute('B', 'Y', Attribute.INTEGER)
        self.schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.ONE))
        self.dataStore.addTable(self.schema, 'A')
        self.dataStore.insertRowsWithValues('A', ['id'], [[1], [2]])
        self.dataStore.addTable(self.schema, 'B')
        self.dataStore.insertRowsWithValues('B', ['id', 'Y'], [[3, 5], [4, 7], [5, 1], [6, 2], [7, 3]])
        self.dataStore.addTable(self.schema, 'AB')
        self.dataStore.insertRowsWithValues('AB', ['id', 'A_id', 'B_id'], [[0, 1, 3], [1, 1, 4], [2, 2, 5], [3, 2, 6], [4, 2, 7]])

        yVals = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'A', [AverageAggregator('[A, AB, B].Y')]))
        yVals = [(idVal, tuple(val for val in vals)) for idVal, vals in yVals]
        TestUtil.assertUnorderedListEqual(self, [(1, (6,)), (2, (2,))], yVals)

        # do one for count
        yVals = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'A', [CountAggregator('[A, AB, B].Y')]))
        yVals = [(idVal, tuple(val for val in vals)) for idVal, vals in yVals]
        TestUtil.assertUnorderedListEqual(self, [(1, (2,)), (2, (3,))], yVals)

        # count shouldn't depend on variable
        yVals = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'A', [CountAggregator('[A, AB, B].exists')]))
        yVals = [(idVal, tuple(val for val in vals)) for idVal, vals in yVals]
        TestUtil.assertUnorderedListEqual(self, [(1, (2,)), (2, (3,))], yVals)

        # do one for random choice (change first aggregator to random choice)
        self.schema = Schema()
        self.schema.addEntity('A')
        self.schema.addEntity('B')
        self.schema.addAttribute('B', 'Y', Attribute.STRING)
        self.schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.ONE))
        self.dataStore = skipOrInstantiateDsSubclass(self)
        self.dataStore.addTable(self.schema, 'A')
        self.dataStore.insertRowsWithValues('A', ['id'], [[1], [2]])
        self.dataStore.addTable(self.schema, 'B')
        self.dataStore.insertRowsWithValues('B', ['id', 'Y'], [[3, 'red'], [4, 'red'], [5, 'blue'], [6, 'blue'], [7, 'blue']])
        self.dataStore.addTable(self.schema, 'AB')
        self.dataStore.insertRowsWithValues('AB', ['id', 'A_id', 'B_id'], [[0, 1, 3], [1, 1, 4], [2, 2, 5], [3, 2, 6], [4, 2, 7]])

        yVals = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'A', [NonRandomChoiceAggregator('[A, AB, B].Y')]))
        yVals = [(idVal, tuple(val for val in vals)) for idVal, vals in yVals]
        TestUtil.assertUnorderedListEqual(self, [(1, ('red',)), (2, ('blue',))], yVals)

        # do one for case aggregator
        self.schema = Schema()
        self.schema.addEntity('A')
        self.schema.addEntity('B')
        self.schema.addAttribute('B', 'Y', Attribute.INTEGER)
        self.schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.ONE))
        self.dataStore = skipOrInstantiateDsSubclass(self)
        self.dataStore.addTable(self.schema, 'A')
        self.dataStore.insertRowsWithValues('A', ['id'], [[1], [2]])
        self.dataStore.addTable(self.schema, 'B')
        self.dataStore.insertRowsWithValues('B', ['id', 'Y'], [[3, 5], [4, 7], [5, 1], [6, 2], [7, 3]])
        self.dataStore.addTable(self.schema, 'AB')
        self.dataStore.insertRowsWithValues('AB', ['id', 'A_id', 'B_id'], [[0, 1, 3], [1, 1, 4], [2, 2, 5], [3, 2, 6], [4, 2, 7]])

        yVals = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'A',
            [CaseAggregator(AverageAggregator('[A, AB, B].Y'), [(CaseAggregator.GREATER_THAN, 5, 'yellow')], elseValue='green')]))
        yVals = [(idVal, tuple(val for val in vals)) for idVal, vals in yVals]
        TestUtil.assertUnorderedListEqual(self, [(1, ('yellow',)), (2, ('green',))], yVals)


    def testMultipleNonIdentityAggregators(self):
        self.schema.addEntity('A')
        self.schema.addEntity('B')
        self.schema.addAttribute('B', 'Y', Attribute.INTEGER)
        self.schema.addAttribute('B', 'Z', Attribute.STRING)
        self.schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.ONE))
        self.dataStore.addTable(self.schema, 'A')
        self.dataStore.insertRowsWithValues('A', ['id'], [[1], [2]])
        self.dataStore.addTable(self.schema, 'B')
        self.dataStore.insertRowsWithValues('B', ['id', 'Y', 'Z'], [[3, 5, 'red'], [4, 7, 'red'], [5, 1, 'blue'], [6, 2, 'blue'], [7, 3, 'blue']])
        self.dataStore.addTable(self.schema, 'AB')
        self.dataStore.insertRowsWithValues('AB', ['id', 'A_id', 'B_id'], [[0, 1, 3], [1, 1, 4], [2, 2, 5], [3, 2, 6], [4, 2, 7]])

        yVals = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'A',
            [AverageAggregator('[A, AB, B].Y'), CountAggregator('[A, AB, B].exists'), NonRandomChoiceAggregator('[A, AB, B].Z')]))
        yVals = [(idVal, tuple(val for val in vals)) for idVal, vals in yVals]
        TestUtil.assertUnorderedListEqual(self, [(1, (6, 2, 'red')), (2, (2, 3, 'blue'))], yVals)


    def testSingleAggrWithNoneValues(self):
        # Relational skeleton looks like this:
        # A1 -- B4
        # A2 -- B5
        # A3 --
        self.schema.addEntity('A')
        self.schema.addEntity('B')
        self.schema.addAttribute('B', 'Y', Attribute.STRING)
        self.schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        self.dataStore.addTable(self.schema, 'A')
        self.dataStore.insertRowsWithValues('A', ['id'], [[1], [2], [3]])
        self.dataStore.addTable(self.schema, 'B')
        self.dataStore.insertRowsWithValues('B', ['id', 'Y'], [[4, 'red'], [5, 'blue']])
        self.dataStore.addTable(self.schema, 'AB')
        self.dataStore.insertRowsWithValues('AB', ['id', 'A_id', 'B_id'], [[0, 1, 4], [1, 2, 5]])

        yVals = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'A', [NonRandomChoiceAggregator('[A, AB, B].Y')]))
        yVals = [(idVal, tuple(val for val in vals)) for idVal, vals in yVals]
        TestUtil.assertUnorderedListEqual(self, [(1, ('red',)), (2, ('blue',)), (3, (None,))], yVals)


    def testMultipleAggrWithNoneValues(self):
        # Relational skeleton looks like this:
        # A1 -- B3 -- C7
        # A2 -- B4
        #       B5 -- C8
        #       B6
        self.schema.addEntity('A')
        self.schema.addEntity('B')
        self.schema.addEntity('C')
        self.schema.addAttribute('A', 'X', Attribute.STRING)
        self.schema.addAttribute('C', 'Z', Attribute.FLOAT)
        self.schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        self.schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.ONE))
        self.dataStore.addTable(self.schema, 'A')
        self.dataStore.insertRowsWithValues('A', ['id', 'X'], [[1, 'red'], [2, 'blue']])
        self.dataStore.addTable(self.schema, 'B')
        self.dataStore.insertRowsWithValues('B', ['id'], [[3], [4], [5], [6]])
        self.dataStore.addTable(self.schema, 'AB')
        self.dataStore.insertRowsWithValues('AB', ['id', 'A_id', 'B_id'], [[0, 1, 3], [1, 2, 4]])
        self.dataStore.addTable(self.schema, 'C')
        self.dataStore.insertRowsWithValues('C', ['id', 'Z'], [[7, 1.5], [8, 2.5]])
        self.dataStore.addTable(self.schema, 'BC')
        self.dataStore.insertRowsWithValues('BC', ['id', 'B_id', 'C_id'], [[0, 3, 7], [1, 5, 8]])

        yVals = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'B',
            [NonRandomChoiceAggregator('[B, AB, A].X'), NonRandomChoiceAggregator('[B, BC, C].Z')]))
        yVals = [(idVal, tuple(val for val in vals)) for idVal, vals in yVals]
        TestUtil.assertUnorderedListEqual(self, [(3, ('red', 1.5)), (4, ('blue', None)), (5, (None, 2.5)), (6, (None, None))], yVals)


    def testCountAggrWithZeroInstances(self):
        self.schema.addEntity('A')
        self.schema.addEntity('B')
        self.schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        self.dataStore.addTable(self.schema, 'A')
        self.dataStore.insertRowsWithValues('A', ['id'], [[i] for i in range(5)])
        self.dataStore.addTable(self.schema, 'B')
        self.dataStore.insertRowsWithValues('B', ['id'], [[i] for i in range(5, 8)])
        self.dataStore.addTable(self.schema, 'AB')
        self.dataStore.insertRowsWithValues('AB', ['id', 'A_id', 'B_id'], [[0, 0, 5], [1, 2, 6], [2, 3, 7]])

        countVals = list(self.dataStore.getValuesForRelVarAggrs(self.schema, 'A', [CountAggregator('[A, AB].exists')]))
        countVals = [(idVal, tuple(val for val in vals)) for idVal, vals in countVals]
        TestUtil.assertUnorderedListEqual(self, [(0, (1,)), (1, (0,)), (2, (1,)), (3, (1,)), (4, (0,))], countVals)


if __name__ == '__main__':
    unittest.main()
