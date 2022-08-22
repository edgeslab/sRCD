import unittest
from causality.model.RelationalDependency import RelationalVariable
from causality.model.RelationalDependency import RelationalDependency
from causality.model import ParserUtil
from causality.test import TestUtil

class ParseUtilTest(unittest.TestCase):

    def testParseRelationalVariable(self):
        relVarStr = '[A].X'
        actualRelVar = ParserUtil.parseRelVar(relVarStr)
        self.assertTrue(isinstance(actualRelVar, RelationalVariable))
        self.assertEqual(['A'], actualRelVar.path)
        self.assertEqual('X', actualRelVar.attrName)
        self.assertFalse(actualRelVar.isExistence())
        self.assertEqual(relVarStr, str(actualRelVar))

        relVarStr = '[A].exists'
        actualRelVar = ParserUtil.parseRelVar(relVarStr)
        self.assertTrue(isinstance(actualRelVar, RelationalVariable))
        self.assertEqual(['A'], actualRelVar.path)
        self.assertEqual('exists', actualRelVar.attrName)
        self.assertTrue(actualRelVar.isExistence())
        self.assertEqual(relVarStr, str(actualRelVar))

        relVarStr = '[B].Y'
        actualRelVar = ParserUtil.parseRelVar(relVarStr)
        self.assertTrue(isinstance(actualRelVar, RelationalVariable))
        self.assertEqual(['B'], actualRelVar.path)
        self.assertEqual('Y', actualRelVar.attrName)
        self.assertFalse(actualRelVar.isExistence())
        self.assertEqual(relVarStr, str(actualRelVar))

        relVarStr = '[AB].XY'
        actualRelVar = ParserUtil.parseRelVar(relVarStr)
        self.assertTrue(isinstance(actualRelVar, RelationalVariable))
        self.assertEqual(['AB'], actualRelVar.path)
        self.assertEqual('XY', actualRelVar.attrName)
        self.assertFalse(actualRelVar.isExistence())
        self.assertEqual(relVarStr, str(actualRelVar))

        relVarStr = '[A, AB].XY'
        actualRelVar = ParserUtil.parseRelVar(relVarStr)
        self.assertTrue(isinstance(actualRelVar, RelationalVariable))
        self.assertEqual(['A', 'AB'], actualRelVar.path)
        self.assertEqual('XY', actualRelVar.attrName)
        self.assertFalse(actualRelVar.isExistence())
        self.assertEqual(relVarStr, str(actualRelVar))

        relVarStr = '[A, AB, B].Y'
        actualRelVar = ParserUtil.parseRelVar(relVarStr)
        self.assertTrue(isinstance(actualRelVar, RelationalVariable))
        self.assertEqual(['A', 'AB', 'B'], actualRelVar.path)
        self.assertEqual('Y', actualRelVar.attrName)
        self.assertFalse(actualRelVar.isExistence())
        self.assertEqual(relVarStr, str(actualRelVar))

        relVarStr = '[A, AB, B].exists'
        actualRelVar = ParserUtil.parseRelVar(relVarStr)
        self.assertTrue(isinstance(actualRelVar, RelationalVariable))
        self.assertEqual(['A', 'AB', 'B'], actualRelVar.path)
        self.assertEqual('exists', actualRelVar.attrName)
        self.assertTrue(actualRelVar.isExistence())
        self.assertEqual(relVarStr, str(actualRelVar))

        # testing that spaces are ignored after commas between item names
        relVarStr = '[A,AB,B].Y'
        actualRelVar = ParserUtil.parseRelVar(relVarStr)
        self.assertTrue(isinstance(actualRelVar, RelationalVariable))
        self.assertEqual(['A', 'AB', 'B'], actualRelVar.path)
        self.assertEqual('Y', actualRelVar.attrName)
        self.assertFalse(actualRelVar.isExistence())
        self.assertEqual('[A, AB, B].Y', str(actualRelVar))

        # testing that spaces are ignored in attribute names
        relVarStr = '[A,AB,B].Y '
        actualRelVar = ParserUtil.parseRelVar(relVarStr)
        self.assertTrue(isinstance(actualRelVar, RelationalVariable))
        self.assertEqual(['A', 'AB', 'B'], actualRelVar.path)
        self.assertEqual('Y', actualRelVar.attrName)
        self.assertFalse(actualRelVar.isExistence())
        self.assertEqual('[A, AB, B].Y', str(actualRelVar))


    def testRelVarInputFormat(self):
        # input not a string
        relVarStr = None
        TestUtil.assertRaisesMessage(self, Exception, "relVarStr 'None' is not a string or RelationalVariable Object",
                    ParserUtil.parseRelVar, relVarStr)

        # not exactly one '.'
        relVarStr = '[AB]XY'
        TestUtil.assertRaisesMessage(self, Exception, "relVarStr '[AB]XY' did not have exactly one dot",
            ParserUtil.parseRelVar, relVarStr)

        relVarStr = '[.AB].XY'
        TestUtil.assertRaisesMessage(self, Exception, "relVarStr '[.AB].XY' did not have exactly one dot",
            ParserUtil.parseRelVar, relVarStr)

        relVarStr = '[[AB].XY'
        TestUtil.assertRaisesMessage(self, Exception, "relVarStr '[[AB].XY' did not have exactly one left square bracket",
            ParserUtil.parseRelVar, relVarStr)

        relVarStr = '[AB].XY]'
        TestUtil.assertRaisesMessage(self, Exception, "relVarStr '[AB].XY]' did not have exactly one right square bracket",
            ParserUtil.parseRelVar, relVarStr)

        # no [] at start and end
        relVarStr = 'A[].Z'
        TestUtil.assertRaisesMessage(self, Exception, "pathStr 'A[]' did not start and end with square brackets",
            ParserUtil.parseRelVar, relVarStr)

        relVarStr = '[A.Z]'
        TestUtil.assertRaisesMessage(self, Exception, "pathStr '[A' did not start and end with square brackets",
            ParserUtil.parseRelVar, relVarStr)


    def testParseRelationalDependency(self):
        relDepStr = '[A].X -> [A].Y'
        actualRelDep = ParserUtil.parseRelDep(relDepStr)
        self.assertTrue(isinstance(actualRelDep, RelationalDependency))
        self.assertEqual('[A].X', str(actualRelDep.relVar1))
        self.assertEqual('[A].Y', str(actualRelDep.relVar2))
        self.assertEqual(relDepStr, str(actualRelDep))

        relDepStr = '[A, AB, B].Y -> [A, AB].XY'
        actualRelDep = ParserUtil.parseRelDep(relDepStr)
        self.assertTrue(isinstance(actualRelDep, RelationalDependency))
        self.assertEqual('[A, AB, B].Y', str(actualRelDep.relVar1))
        self.assertEqual('[A, AB].XY', str(actualRelDep.relVar2))
        self.assertEqual(relDepStr, str(actualRelDep))

        relDepStr = '[A, AB, B].Y -> [A, AB].XY'
        actualRelDepIn = ParserUtil.parseRelDep(relDepStr)
        actualRelDepOut = ParserUtil.parseRelDep(actualRelDepIn)
        self.assertEqual(actualRelDepIn, actualRelDepOut)

        TestUtil.assertRaisesMessage(self, Exception, "relDepStr is not a string or RelationalDependency object",
            ParserUtil.parseRelDep, None)


    def testRelDepInputFormat(self):
        # no ->
        relDepStr = '[A].X xx [A].Y'
        TestUtil.assertRaisesMessage(self, Exception, "relDepStr '[A].X xx [A].Y' did not have exactly one '->' arrow",
            ParserUtil.parseRelDep, relDepStr)

        # more than one ->
        relDepStr = '[A].X ->-> [A].Y'
        TestUtil.assertRaisesMessage(self, Exception, "relDepStr '[A].X ->-> [A].Y' did not have exactly one '->' arrow",
            ParserUtil.parseRelDep, relDepStr)


    def testRelDepConstructorBadInput(self):
        TestUtil.assertRaisesMessage(self, Exception, "RelationalDependency expects two RelationalVariable objects",
            RelationalDependency, None, None)


if __name__ == '__main__':
    unittest.main()
