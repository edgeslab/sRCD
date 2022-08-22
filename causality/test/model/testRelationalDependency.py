import unittest
from causality.model import ParserUtil

class TestRelationalDependency(unittest.TestCase):

    def testReverseDependency(self):
        relDep = ParserUtil.parseRelDep('[A].X -> [A].Y')
        self.assertEqual(ParserUtil.parseRelDep('[A].Y -> [A].X'), relDep.reverse())

        relDep = ParserUtil.parseRelDep('[B].Y -> [B].X')
        self.assertEqual(ParserUtil.parseRelDep('[B].X -> [B].Y'), relDep.reverse())

        relDep = ParserUtil.parseRelDep('[B, AB, A].X -> [B].Y')
        self.assertEqual(ParserUtil.parseRelDep('[A, AB, B].Y -> [A].X'), relDep.reverse())

        relDep = ParserUtil.parseRelDep('[B, AB, A, AB, B].Y2 -> [B].Y1')
        self.assertEqual(ParserUtil.parseRelDep('[B, AB, A, AB, B].Y1 -> [B].Y2'), relDep.reverse())


if __name__ == '__main__':
    unittest.main()
