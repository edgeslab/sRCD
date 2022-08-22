import unittest
from causality.test import TestUtil
from causality.model.Schema import Attribute
from causality.model.Schema import Schema
from causality.citest.CITest import CITest
from causality.citest.CITest import LinearCITest
from causality.datastore.InMemoryDataStore import InMemoryDataStore
from causality.datastore.Sqlite3DataStore import Sqlite3DataStore

class TestLinearCITest(unittest.TestCase):

    def testCITest(self):
        citest = CITest()
        TestUtil.assertRaisesMessage(self, NotImplementedError, None,
                 citest.isConditionallyIndependent, None, None, None)


    def testLinearCITestWithPropositional(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X1', Attribute.FLOAT)
        schema.addAttribute('A', 'X2', Attribute.FLOAT)
        schema.addAttribute('A', 'X3', Attribute.FLOAT)
        schema.addAttribute('A', 'X4', Attribute.FLOAT)

        dataStore = Sqlite3DataStore()
        dataStore.addTable(schema, 'A')
        aFile = open('causality/test/citest/propositional-data.csv', 'r')
        dataStore.loadCsvFile('A', ['id', 'X1', 'X2', 'X3', 'X4'], 'causality/test/citest/propositional-data.csv', aFile, ',', 5)

        linearCITest = LinearCITest(schema, dataStore)
        self.assertTrue(isinstance(linearCITest, CITest))
        self.assertEqual(0.05, linearCITest.alpha) # default for alpha
        self.assertEqual(0.01, linearCITest.soeThreshold) # default for soeThreshold
        self.assertTrue(linearCITest.isConditionallyIndependent('[A].X1', '[A].X2', []))
        self.assertFalse(linearCITest.isConditionallyIndependent('[A].X1', '[A].X2', ['[A].X3']))
        self.assertFalse(linearCITest.isConditionallyIndependent('[A].X1', '[A].X2', ['[A].X4'])) # tests conditioning on descendant of a collider
        self.assertFalse(linearCITest.isConditionallyIndependent('[A].X1', '[A].X2', ['[A].X3', '[A].X4']))

        # test ability to pass in alpha or soeThreshold
        linearCITest = LinearCITest(schema, dataStore, alpha=0.32)
        self.assertFalse(linearCITest.isConditionallyIndependent('[A].X1', '[A].X2', [])) # p-value=0.3113290, soe=0.01045787

        linearCITest = LinearCITest(schema, dataStore, soeThreshold=0.9)
        self.assertEqual(0.9, linearCITest.soeThreshold)
        self.assertTrue(linearCITest.isConditionallyIndependent('[A].X1', '[A].X2', ['[A].X3'])) # p-value<<10^-50, soe=0.897906
        aFile.close()

        # true model has multiple paths
        dataStore = Sqlite3DataStore()
        dataStore.addTable(schema, 'A')
        aFile = open('causality/test/citest/propositional-data-2.csv', 'r')
        dataStore.loadCsvFile('A', ['id', 'X1', 'X2', 'X3', 'X4'], 'causality/test/citest/propositional-data-2.csv', aFile, ',', 5)
        linearCITest = LinearCITest(schema, dataStore)
        self.assertFalse(linearCITest.isConditionallyIndependent('[A].X1', '[A].X4', []))
        self.assertTrue(linearCITest.isConditionallyIndependent('[A].X1', '[A].X4', ['[A].X2']))

        linearCITest = LinearCITest(schema, dataStore, soeThreshold=0.001, alpha=0.75)
        self.assertFalse(linearCITest.isConditionallyIndependent('[A].X1', '[A].X4', ['[A].X2', '[A].X3'])) # p-value=0.72122, soe=0.00133

        linearCITest = LinearCITest(schema, dataStore)
        self.assertFalse(linearCITest.isConditionallyIndependent('[A].X2', '[A].X3', []))
        self.assertFalse(linearCITest.isConditionallyIndependent('[A].X2', '[A].X3', ['[A].X1']))
        self.assertFalse(linearCITest.isConditionallyIndependent('[A].X2', '[A].X3', ['[A].X4']))

        linearCITest = LinearCITest(schema, dataStore, soeThreshold=0.08, alpha=0.005)
        self.assertTrue(linearCITest.isConditionallyIndependent('[A].X2', '[A].X3', ['[A].X1', '[A].X4'])) # p-value=0.0062, soe=0.075
        aFile.close()


    def testBadInput(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X1')
        schema.addAttribute('A', 'X2')
        dataStore = InMemoryDataStore()
        linearCITest = LinearCITest(schema, dataStore)
        TestUtil.assertRaisesMessage(self, Exception, "relVar1Str must be a parseable RelationalVariable string",
             linearCITest.isConditionallyIndependent, None, '[A].X2', [])
        TestUtil.assertRaisesMessage(self, Exception, "relVar2Str must be a parseable RelationalVariable string",
             linearCITest.isConditionallyIndependent, '[A].X1', None, [])
        TestUtil.assertRaisesMessage(self, Exception, "condRelVarStrs must be a sequence of parseable RelationalVariable strings",
             linearCITest.isConditionallyIndependent, '[A].X1', '[A].X2', None)
        TestUtil.assertRaisesMessage(self, Exception, "condRelVarStrs must be a sequence of parseable RelationalVariable strings",
             linearCITest.isConditionallyIndependent, '[A].X1', '[A].X2', '[A].X2')

        # relVar2Str MUST be singleton
        schema.addEntity('B')
        schema.addAttribute('B', 'Y', Attribute.FLOAT)
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.MANY))
        TestUtil.assertRaisesMessage(self, Exception, "At least one of relVar1Str or relVar2Str must have a singleton path",
             linearCITest.isConditionallyIndependent, '[A, AB, B, AB, A].X1', '[A, AB, B].Y', [])


    def testLinearCITestWithThreeEntityRDSJMLRExample(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X', Attribute.FLOAT)
        schema.addEntity('B')
        schema.addAttribute('B', 'Y', Attribute.FLOAT)
        schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.MANY))
        schema.addEntity('C')
        schema.addAttribute('C', 'Z', Attribute.FLOAT)
        schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.MANY))

        dataStore = Sqlite3DataStore()
        dataStore.addTable(schema, 'A')
        dataStore.addTable(schema, 'B')
        dataStore.addTable(schema, 'C')
        dataStore.addTable(schema, 'AB')
        dataStore.addTable(schema, 'BC')
        aFile = open('causality/test/citest/relational-data-a.csv', 'r')
        dataStore.loadCsvFile('A', ['id', 'X'], 'causality/test/citest/relational-data-a.csv', aFile, ',', 2)
        aFile.close()
        bFile = open('causality/test/citest/relational-data-b.csv', 'r')
        dataStore.loadCsvFile('B', ['id', 'Y'], 'causality/test/citest/relational-data-b.csv', bFile, ',', 2)
        bFile.close()
        cFile = open('causality/test/citest/relational-data-c.csv', 'r')
        dataStore.loadCsvFile('C', ['id', 'Z'], 'causality/test/citest/relational-data-c.csv', cFile, ',', 2)
        cFile.close()
        abFile = open('causality/test/citest/relational-data-ab.csv', 'r')
        dataStore.loadCsvFile('AB', ['id', 'a_id', 'b_id'], 'causality/test/citest/relational-data-ab.csv', abFile, ',', 3)
        abFile.close()
        bcFile = open('causality/test/citest/relational-data-bc.csv', 'r')
        dataStore.loadCsvFile('BC', ['id', 'b_id', 'c_id'], 'causality/test/citest/relational-data-bc.csv', bcFile, ',', 3)
        bcFile.close()

        linearCITest = LinearCITest(schema, dataStore)
        self.assertFalse(linearCITest.isConditionallyIndependent('[A, AB, B, BC, C].Z', '[A].X', []))
        self.assertTrue(linearCITest.isConditionallyIndependent('[A, AB, B, BC, C].Z', '[A].X', ['[A, AB, B].Y']))
        self.assertTrue(linearCITest.isConditionallyIndependent('[A, AB, B, BC, C].Z', '[A].X',
                                                                ['[A, AB, B].Y', '[A, AB, B, AB, A].X']))

        linearCITest = LinearCITest(schema, dataStore, alpha=0.4, soeThreshold=0.001)
        self.assertFalse(linearCITest.isConditionallyIndependent('[A, AB, B, BC, C].Z', '[A].X', ['[A, AB, B].Y']))

if __name__ == '__main__':
    unittest.main()
