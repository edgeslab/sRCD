import unittest
from causality.model.Schema import Attribute
from causality.model.Schema import Schema
from causality.test import TestUtil
from causality.test.testDataStoreSuite import skipOrInstantiateDsSubclass
from causality.datastore.SqlDataStore import SqlDataStore


class TestDataStoreBasicOperations(unittest.TestCase):

    dsSubclassInitTuple = None    # DataStore implementation. set by suite runner. None if called directly via IntelliJ (which we skip), rather than DataStoreTestSuite


    def setUp(self):
        self.dataStore = skipOrInstantiateDsSubclass(self)
        self.schema = Schema()


    def tearDown(self):
        self.dataStore.close()


    def testAttributeDataTypeToSqlType(self):
        expectedColumnTypes = {Attribute.INTEGER: 'INTEGER', Attribute.STRING: 'TEXT', Attribute.FLOAT: 'REAL'}
        if isinstance(self.dataStore, SqlDataStore):
            for dataType in Attribute.DATA_TYPES.values():
                self.assertEqual(expectedColumnTypes[dataType], self.dataStore.getColumnTypeForAttributeType(dataType))


    def testInsertRowsWithValues(self):
        self.schema.addEntity('A')
        self.dataStore.addTable(self.schema, 'A')
        self.dataStore.insertRowsWithValues('A', ['id'], [[1]])
        actualRowsA = self.dataStore.getAllRows(self.schema, 'A', ['id'])
        self.assertEqual([(1,)], actualRowsA)

        self.dataStore.insertRowsWithValues('A', ['id'], [[2], [4]])
        actualRowsA = self.dataStore.getAllRows(self.schema, 'A', ['id'])
        self.assertEqual([(1,), (2,), (4,)], actualRowsA)

        self.schema.addEntity('B')
        self.dataStore.addTable(self.schema, 'B')
        self.dataStore.insertRowsWithValues('B', ['id'], [[3], [5]])
        actualRowsB = self.dataStore.getAllRows(self.schema, 'B', ['id'])
        self.assertEqual([(3,), (5,)], actualRowsB)

        # test adding a relationship schema item. Should automatically add column support for primary and foreign keys
        self.schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.ONE))
        self.dataStore.addTable(self.schema, 'AB')
        self.dataStore.insertRowsWithValues('AB', ['id', 'A_id', 'B_id'], [[1, 1, 3], [2, 4, 5]])
        actualRowsB = self.dataStore.getAllRows(self.schema, 'AB', ['id', 'A_id', 'B_id'])
        self.assertEqual([(1, 1, 3), (2, 4, 5)], actualRowsB)

        # test additional columns
        self.schema.addEntity('C')
        self.schema.addAttribute('C', 'X', Attribute.STRING)
        self.dataStore.addTable(self.schema, 'C')
        self.dataStore.insertRowsWithValues('C', ['id', 'X'], [[0, 'red'], [1, 'blue']])
        actualRowsC = self.dataStore.getAllRows(self.schema, 'C', ['id', 'X'])
        self.assertEqual([(0, 'red'), (1, 'blue')], actualRowsC)

        self.dataStore.insertRowsWithValues('C', ['X', 'id'], [['yellow', 2], ['blue', 3]])
        actualRowsC = self.dataStore.getAllRows(self.schema, 'C', ['id', 'X'])
        self.assertEqual([(0, 'red'), (1, 'blue'), (2, 'yellow'), (3, 'blue')], actualRowsC)


    def testUpdateRowValue(self):
        self.schema.addEntity('A')
        self.schema.addAttribute('A', 'X', Attribute.STRING)
        self.dataStore.addTable(self.schema, 'A')
        self.dataStore.insertRowsWithValues('A', ['id'], [[1]])
        self.dataStore.updateRowValue('A', 1, 'X', 'red')
        actualRowsA = self.dataStore.getAllRows(self.schema, 'A', ['id', 'X'])
        self.assertEqual([(1, 'red')], list(actualRowsA))

        # update should overwrite previous values
        self.dataStore.updateRowValue('A', 1, 'X', 'blue')
        actualRowsA = self.dataStore.getAllRows(self.schema, 'A', ['id', 'X'])
        self.assertEqual([(1, 'blue')], list(actualRowsA))

        # update row that doesn't exist, should not update
        self.dataStore.updateRowValue('A', 2, 'X', 'blue')
        actualRowsA = self.dataStore.getAllRows(self.schema, 'A', ['id', 'X'])
        self.assertEqual([(1, 'blue')], list(actualRowsA))


    def testDeleteRow(self):
        self.schema.addEntity('A')
        self.schema.addAttribute('A', 'X', Attribute.INTEGER)
        self.dataStore.addTable(self.schema, 'A')
        self.dataStore.insertRowsWithValues('A', ['id'], [[1]])
        self.dataStore.deleteRow('A', 1)
        actualRowsA = self.dataStore.getAllRows(self.schema, 'A', ['id', 'X'])
        self.assertEqual(0, len(actualRowsA))

        # delete a row that doesn't exist. should do nothing, and not error
        self.dataStore.deleteRow('A', 2)


    def testAddDropTable(self):
        self.assertEqual([], self.dataStore.getTableNames())

        self.schema.addEntity('A')
        self.schema.addEntity('B')
        self.dataStore.addTable(self.schema, 'A')
        self.dataStore.addTable(self.schema, 'B')
        TestUtil.assertUnorderedListEqual(self, list(map(str.lower, ['a', 'b'])), self.dataStore.getTableNames())

        self.dataStore.dropAllTables()
        self.assertEqual([], self.dataStore.getTableNames())


    def testBadInsertRows(self):
        self.schema.addEntity('A')
        self.schema.addAttribute('A', 'X', Attribute.INTEGER)
        # test duplicate row ids
        self.dataStore.addTable(self.schema, 'A')
        self.dataStore.insertRowsWithValues('A', ['id'], [[1]])
        TestUtil.assertRaisesMessage(self, Exception, "Row id 1 already exists.",
            self.dataStore.insertRowsWithValues, 'A', ['id'], [[1]])

        self.dataStore.rollback()
        self.dataStore.insertRowsWithValues('A', ['id'], [[2]])
        TestUtil.assertRaisesMessage(self, Exception, "Row id 2 already exists.",
            self.dataStore.insertRowsWithValues, 'A', ['id'], [[2]])

        # columnNames can't be empty
        self.dataStore.rollback()
        TestUtil.assertRaisesMessage(self, Exception, "Column names are empty.",
            self.dataStore.insertRowsWithValues, 'A', [], [[2]])

        # columnNames must include 'id'
        self.dataStore.rollback()
        TestUtil.assertRaisesMessage(self, Exception, "Column name 'id' not found.",
            self.dataStore.insertRowsWithValues, 'A', ['X'], [[2]])

        # columnNames can't have any duplicate names
        self.dataStore.rollback()
        TestUtil.assertRaisesMessage(self, Exception, "Duplicate column name found.",
            self.dataStore.insertRowsWithValues, 'A', ['id', 'id'], [[2, 2]])

        # each row must have the same number of values as the columnNames
        self.dataStore.rollback()
        TestUtil.assertRaisesMessage(self, Exception, "Column count and row count differ: [], ['id']",
            self.dataStore.insertRowsWithValues, 'A', ['id'], [[]])

        self.dataStore.rollback()
        TestUtil.assertRaisesMessage(self, Exception, "Column count and row count differ: [1, 11], ['id']",
            self.dataStore.insertRowsWithValues, 'A', ['id'], [[1, 11]])

        self.dataStore.rollback()
        TestUtil.assertRaisesMessage(self, Exception, "Column count and row count differ: [4, 44, 2.5], ['id', 'X']",
            self.dataStore.insertRowsWithValues, 'A', ['id', 'X'], [[3, 33], [4, 44, 2.5]])


    def testBadAddTable(self):
        # schema doesn't contain schema item
        TestUtil.assertRaisesMessage(self, Exception, "Schema does not contain schema item 'A'.",
            self.dataStore.addTable, self.schema, 'A')

        # duplicate table
        self.schema.addEntity('A')
        self.dataStore.addTable(self.schema, 'A')
        TestUtil.assertRaisesMessage(self, Exception, "Table 'A' already exists.",
            self.dataStore.addTable, self.schema, 'A')


    def testBadTableRowOrCol(self):
        TestUtil.assertRaisesMessage(self, Exception, "Table 'A' does not exist yet.",
            self.dataStore.insertRowsWithValues, 'A', ['id'], [[1]])
        TestUtil.assertRaisesMessage(self, Exception, "Table 'B' does not exist yet.",
            self.dataStore.insertRowsWithValues, 'B', ['id'], [[1]])

        TestUtil.assertRaisesMessage(self, Exception, "Table 'A' does not exist yet.",
            self.dataStore.getAllRows, self.schema, 'A', [])
        TestUtil.assertRaisesMessage(self, Exception, "Table 'B' does not exist yet.",
            self.dataStore.getAllRows, self.schema, 'B', [])


    def testBadUpdateOrDeleteRowValue(self):
        TestUtil.assertRaisesMessage(self, Exception, "Table 'A' does not exist yet.",
            self.dataStore.updateRowValue, 'A', 1, 'X', None)
        TestUtil.assertRaisesMessage(self, Exception, "Table 'B' does not exist yet.",
            self.dataStore.updateRowValue, 'B', 1, 'X', None)

        TestUtil.assertRaisesMessage(self, Exception, "Table 'A' does not exist yet.",
            self.dataStore.deleteRow, 'A', 1)
        TestUtil.assertRaisesMessage(self, Exception, "Table 'B' does not exist yet.",
            self.dataStore.deleteRow, 'B', 1)


    def testInterleavedGetAllRows(self):
        self.schema.addEntity('A')
        self.schema.addAttribute('A', 'X', Attribute.STRING)
        self.dataStore.addTable(self.schema, 'A')

        self.assertEqual([], self.dataStore.getAllRows(self.schema, 'A', ['id']))
        self.assertEqual([], self.dataStore.getAllRows(self.schema, 'A', ['id', 'X']))

        self.dataStore.insertRowsWithValues('A', ['id', 'X'], [[1, 'red']])
        self.assertEqual([(1,)], self.dataStore.getAllRows(self.schema, 'A', ['id']))
        self.assertEqual([(1, 'red')], self.dataStore.getAllRows(self.schema, 'A', ['id', 'X']))

        self.dataStore.insertRowsWithValues('A', ['id'], [[2]])
        self.assertEqual([(1,), (2,)], self.dataStore.getAllRows(self.schema, 'A', ['id']))
        self.assertEqual([(1, 'red'), (2, None)], self.dataStore.getAllRows(self.schema, 'A', ['id', 'X']))


    def testBadColumnNamesForGetAllRows(self):
        self.schema.addEntity('A')
        self.dataStore.addTable(self.schema, 'A')
        TestUtil.assertRaisesMessage(self, Exception, "Column name 'X' does not exist in table 'A'",
            self.dataStore.getAllRows, self.schema, 'A', ['id', 'X'])

        self.dataStore = skipOrInstantiateDsSubclass(self)
        self.schema.addAttribute('A', 'X', Attribute.INTEGER)
        self.dataStore.addTable(self.schema, 'A')
        TestUtil.assertRaisesMessage(self, Exception, "Column name 'Y' does not exist in table 'A'",
            self.dataStore.getAllRows, self.schema, 'A', ['id', 'X', 'Y'])


if __name__ == '__main__':
    unittest.main()
