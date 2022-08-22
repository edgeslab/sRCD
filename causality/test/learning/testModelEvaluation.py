import unittest
from causality.test import TestUtil
from causality.model.Model import Model
from causality.model.Schema import Schema
from causality.learning import ModelEvaluation

class TestModelEvaluation(unittest.TestCase):

    def testComputePrecision(self):
        trueValues = []
        learnedValues = []
        self.assertEqual(1.0, ModelEvaluation.precision(trueValues, learnedValues))

        trueValues = [1]
        learnedValues = []
        self.assertEqual(1.0, ModelEvaluation.precision(trueValues, learnedValues))

        trueValues = []
        learnedValues = [1]
        self.assertEqual(0.0, ModelEvaluation.precision(trueValues, learnedValues))

        trueValues = [1, 2]
        learnedValues = [2]
        self.assertEqual(1.0, ModelEvaluation.precision(trueValues, learnedValues))

        trueValues = [1, 2]
        learnedValues = [2, 3]
        self.assertEqual(0.5, ModelEvaluation.precision(trueValues, learnedValues))

        trueValues = [1, 2]
        learnedValues = [2, 3, 4]
        self.assertEqual(1/3, ModelEvaluation.precision(trueValues, learnedValues))


    def testComputeRecall(self):
        trueValues = []
        learnedValues = []
        self.assertEqual(1.0, ModelEvaluation.recall(trueValues, learnedValues))

        trueValues = [1]
        learnedValues = []
        self.assertEqual(0.0, ModelEvaluation.recall(trueValues, learnedValues))

        trueValues = []
        learnedValues = [1]
        self.assertEqual(1.0, ModelEvaluation.recall(trueValues, learnedValues))

        trueValues = [1, 2]
        learnedValues = [2]
        self.assertEqual(0.5, ModelEvaluation.recall(trueValues, learnedValues))

        trueValues = [1, 2]
        learnedValues = [2, 3]
        self.assertEqual(0.5, ModelEvaluation.recall(trueValues, learnedValues))

        trueValues = [1, 2]
        learnedValues = [2, 3, 4]
        self.assertEqual(0.5, ModelEvaluation.recall(trueValues, learnedValues))

        trueValues = [1, 2, 3]
        learnedValues = [2, 3, 4]
        self.assertEqual(2/3, ModelEvaluation.recall(trueValues, learnedValues))


    def testSkeletonPrecision(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X1')
        schema.addAttribute('A', 'X2')
        schema.addAttribute('A', 'X3')
        schema.addAttribute('A', 'X4')
        dependencies = []
        model = Model(schema, dependencies)

        learnedDependencies = []
        self.assertEqual(1.0, ModelEvaluation.skeletonPrecision(model, learnedDependencies))

        learnedDependencies = ['[A].X1 -> [A].X2']
        self.assertEqual(0.0, ModelEvaluation.skeletonPrecision(model, learnedDependencies))

        # true model has one dependency
        dependencies = ['[A].X1 -> [A].X2']
        model = Model(schema, dependencies)
        learnedDependencies = []
        self.assertEqual(1.0, ModelEvaluation.skeletonPrecision(model, learnedDependencies))

        learnedDependencies = ['[A].X1 -> [A].X2']
        self.assertEqual(1.0, ModelEvaluation.skeletonPrecision(model, learnedDependencies))

        learnedDependencies = ['[A].X1 -> [A].X2', '[A].X2 -> [A].X1']
        self.assertEqual(1.0, ModelEvaluation.skeletonPrecision(model, learnedDependencies))

        # true model has two dependencies
        dependencies = ['[A].X1 -> [A].X2', '[A].X3 -> [A].X2']
        model = Model(schema, dependencies)
        learnedDependencies = ['[A].X3 -> [A].X2']
        self.assertEqual(1.0, ModelEvaluation.skeletonPrecision(model, learnedDependencies))

        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X2 -> [A].X3']
        self.assertEqual(1.0, ModelEvaluation.skeletonPrecision(model, learnedDependencies))

        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X2 -> [A].X3', '[A].X2 -> [A].X4']
        self.assertEqual(0.5, ModelEvaluation.skeletonPrecision(model, learnedDependencies))

        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X2 -> [A].X3', '[A].X2 -> [A].X4', '[A].X4 -> [A].X2']
        self.assertEqual(0.5, ModelEvaluation.skeletonPrecision(model, learnedDependencies))

        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X2 -> [A].X3', '[A].X2 -> [A].X4', '[A].X4 -> [A].X2',
                               '[A].X1 -> [A].X4']
        self.assertEqual(1/3, ModelEvaluation.skeletonPrecision(model, learnedDependencies))

        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X2 -> [A].X3', '[A].X2 -> [A].X4', '[A].X4 -> [A].X2',
                               '[A].X1 -> [A].X4', '[A].X4 -> [A].X1']
        self.assertEqual(1/3, ModelEvaluation.skeletonPrecision(model, learnedDependencies))


    def testSkeletonRecall(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X1')
        schema.addAttribute('A', 'X2')
        schema.addAttribute('A', 'X3')
        schema.addAttribute('A', 'X4')
        dependencies = []
        model = Model(schema, dependencies)

        learnedDependencies = []
        self.assertEqual(1.0, ModelEvaluation.skeletonRecall(model, learnedDependencies))
        learnedDependencies = ['[A].X1 -> [A].X2']
        self.assertEqual(1.0, ModelEvaluation.skeletonRecall(model, learnedDependencies))
        learnedDependencies = ['[A].X1 -> [A].X2', '[A].X2 -> [A].X1']
        self.assertEqual(1.0, ModelEvaluation.skeletonRecall(model, learnedDependencies))

        # true model has one dependency
        dependencies = ['[A].X1 -> [A].X2']
        model = Model(schema, dependencies)
        learnedDependencies = []
        self.assertEqual(0.0, ModelEvaluation.skeletonRecall(model, learnedDependencies))

        # true model has two dependencies
        dependencies = ['[A].X1 -> [A].X2', '[A].X3 -> [A].X2']
        model = Model(schema, dependencies)
        learnedDependencies = ['[A].X3 -> [A].X2']
        self.assertEqual(0.5, ModelEvaluation.skeletonRecall(model, learnedDependencies))

        learnedDependencies = ['[A].X2 -> [A].X3']
        self.assertEqual(0.5, ModelEvaluation.skeletonRecall(model, learnedDependencies))

        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X2 -> [A].X3']
        self.assertEqual(0.5, ModelEvaluation.skeletonRecall(model, learnedDependencies))

        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X2 -> [A].X3', '[A].X4 -> [A].X1']
        self.assertEqual(0.5, ModelEvaluation.skeletonRecall(model, learnedDependencies))

        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X2 -> [A].X3', '[A].X4 -> [A].X1', '[A].X1 -> [A].X4']
        self.assertEqual(0.5, ModelEvaluation.skeletonRecall(model, learnedDependencies))

        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X2 -> [A].X3', '[A].X4 -> [A].X1',
                               '[A].X1 -> [A].X4', '[A].X1 -> [A].X3']
        self.assertEqual(0.5, ModelEvaluation.skeletonRecall(model, learnedDependencies))

        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X2 -> [A].X3', '[A].X4 -> [A].X1',
                               '[A].X1 -> [A].X4', '[A].X1 -> [A].X3', '[A].X2 -> [A].X1']
        self.assertEqual(1.0, ModelEvaluation.skeletonRecall(model, learnedDependencies))

        # true model has three dependencies
        dependencies = ['[A].X1 -> [A].X2', '[A].X3 -> [A].X2', '[A].X2 -> [A].X4']
        model = Model(schema, dependencies)
        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X2 -> [A].X4', '[A].X1 -> [A].X4']
        self.assertEqual(2/3, ModelEvaluation.skeletonRecall(model, learnedDependencies))

        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X2 -> [A].X4', '[A].X4 -> [A].X2',
                               '[A].X1 -> [A].X4', '[A].X4 -> [A].X1']
        self.assertEqual(2/3, ModelEvaluation.skeletonRecall(model, learnedDependencies))


    def testOrientedPrecision(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X1')
        schema.addAttribute('A', 'X2')
        schema.addAttribute('A', 'X3')
        schema.addAttribute('A', 'X4')
        dependencies = []
        model = Model(schema, dependencies)

        learnedDependencies = []
        self.assertEqual(1.0, ModelEvaluation.orientedPrecision(model, learnedDependencies))

        learnedDependencies = ['[A].X1 -> [A].X2']
        self.assertEqual(0.0, ModelEvaluation.orientedPrecision(model, learnedDependencies))

        # only oriented dependencies count
        learnedDependencies = ['[A].X1 -> [A].X2', '[A].X2 -> [A].X1']
        self.assertEqual(1.0, ModelEvaluation.orientedPrecision(model, learnedDependencies))

        learnedDependencies = ['[A].X1 -> [A].X2', '[A].X2 -> [A].X1', '[A].X3 -> [A].X2']
        self.assertEqual(0.0, ModelEvaluation.orientedPrecision(model, learnedDependencies))

        learnedDependencies = ['[A].X1 -> [A].X2', '[A].X2 -> [A].X1', '[A].X3 -> [A].X2', '[A].X2 -> [A].X3']
        self.assertEqual(1.0, ModelEvaluation.orientedPrecision(model, learnedDependencies))

        # true model has one dependency
        dependencies = ['[A].X1 -> [A].X2']
        model = Model(schema, dependencies)
        learnedDependencies = []
        self.assertEqual(1.0, ModelEvaluation.orientedPrecision(model, learnedDependencies))

        learnedDependencies = ['[A].X1 -> [A].X2']
        self.assertEqual(1.0, ModelEvaluation.orientedPrecision(model, learnedDependencies))

        learnedDependencies = ['[A].X1 -> [A].X2', '[A].X2 -> [A].X1']
        self.assertEqual(1.0, ModelEvaluation.orientedPrecision(model, learnedDependencies))

        learnedDependencies = ['[A].X1 -> [A].X2', '[A].X3 -> [A].X2']
        self.assertEqual(0.5, ModelEvaluation.orientedPrecision(model, learnedDependencies))

        learnedDependencies = ['[A].X1 -> [A].X2', '[A].X3 -> [A].X2', '[A].X2 -> [A].X3']
        self.assertEqual(1.0, ModelEvaluation.orientedPrecision(model, learnedDependencies))

        # true model has two dependency
        dependencies = ['[A].X1 -> [A].X2', '[A].X3 -> [A].X2']
        model = Model(schema, dependencies)
        learnedDependencies = ['[A].X3 -> [A].X2']
        self.assertEqual(1.0, ModelEvaluation.orientedPrecision(model, learnedDependencies))

        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X2 -> [A].X3']
        self.assertEqual(1.0, ModelEvaluation.orientedPrecision(model, learnedDependencies))

        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X2 -> [A].X3', '[A].X2 -> [A].X4']
        self.assertEqual(0.0, ModelEvaluation.orientedPrecision(model, learnedDependencies))

        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X2 -> [A].X4']
        self.assertEqual(0.5, ModelEvaluation.orientedPrecision(model, learnedDependencies))

        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X2 -> [A].X4', '[A].X4 -> [A].X2',
                               '[A].X1 -> [A].X2']
        self.assertEqual(1.0, ModelEvaluation.orientedPrecision(model, learnedDependencies))

        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X2 -> [A].X4', '[A].X1 -> [A].X4']
        self.assertEqual(1/3, ModelEvaluation.orientedPrecision(model, learnedDependencies))

        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X2 -> [A].X4', '[A].X1 -> [A].X2']
        self.assertEqual(2/3, ModelEvaluation.orientedPrecision(model, learnedDependencies))


    def testOrientedRecall(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X1')
        schema.addAttribute('A', 'X2')
        schema.addAttribute('A', 'X3')
        schema.addAttribute('A', 'X4')
        dependencies = []
        model = Model(schema, dependencies)

        learnedDependencies = []
        self.assertEqual(1.0, ModelEvaluation.orientedRecall(model, learnedDependencies))

        learnedDependencies = ['[A].X1 -> [A].X2']
        self.assertEqual(1.0, ModelEvaluation.orientedRecall(model, learnedDependencies))

        learnedDependencies = ['[A].X1 -> [A].X2', '[A].X2 -> [A].X1']
        self.assertEqual(1.0, ModelEvaluation.orientedRecall(model, learnedDependencies))

        learnedDependencies = ['[A].X1 -> [A].X2', '[A].X2 -> [A].X3']
        self.assertEqual(1.0, ModelEvaluation.orientedRecall(model, learnedDependencies))

        # true model has one dependency
        dependencies = ['[A].X1 -> [A].X2']
        model = Model(schema, dependencies)
        learnedDependencies = []
        self.assertEqual(0.0, ModelEvaluation.orientedRecall(model, learnedDependencies))

        learnedDependencies = ['[A].X1 -> [A].X2', '[A].X2 -> [A].X1']
        self.assertEqual(0.0, ModelEvaluation.orientedRecall(model, learnedDependencies))

        learnedDependencies = ['[A].X2 -> [A].X1']
        self.assertEqual(0.0, ModelEvaluation.orientedRecall(model, learnedDependencies))

        learnedDependencies = ['[A].X1 -> [A].X2', '[A].X2 -> [A].X3']
        self.assertEqual(1.0, ModelEvaluation.orientedRecall(model, learnedDependencies))

        learnedDependencies = ['[A].X1 -> [A].X2', '[A].X2 -> [A].X3', '[A].X3 -> [A].X2']
        self.assertEqual(1.0, ModelEvaluation.orientedRecall(model, learnedDependencies))

        # true model has two dependencies
        dependencies = ['[A].X1 -> [A].X2', '[A].X3 -> [A].X2']
        model = Model(schema, dependencies)
        learnedDependencies = ['[A].X3 -> [A].X2']
        self.assertEqual(0.5, ModelEvaluation.orientedRecall(model, learnedDependencies))

        learnedDependencies = ['[A].X2 -> [A].X3']
        self.assertEqual(0.0, ModelEvaluation.orientedRecall(model, learnedDependencies))

        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X2 -> [A].X3']
        self.assertEqual(0.0, ModelEvaluation.orientedRecall(model, learnedDependencies))

        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X4 -> [A].X1']
        self.assertEqual(0.5, ModelEvaluation.orientedRecall(model, learnedDependencies))

        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X2 -> [A].X3', '[A].X4 -> [A].X1', '[A].X1 -> [A].X4']
        self.assertEqual(0.0, ModelEvaluation.orientedRecall(model, learnedDependencies))

        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X4 -> [A].X1', '[A].X1 -> [A].X4']
        self.assertEqual(0.5, ModelEvaluation.orientedRecall(model, learnedDependencies))

        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X4 -> [A].X1',
                               '[A].X1 -> [A].X4', '[A].X1 -> [A].X2']
        self.assertEqual(1.0, ModelEvaluation.orientedRecall(model, learnedDependencies))

        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X2 -> [A].X3', '[A].X4 -> [A].X1',
                               '[A].X1 -> [A].X4', '[A].X1 -> [A].X3', '[A].X2 -> [A].X1']
        self.assertEqual(0.0, ModelEvaluation.orientedRecall(model, learnedDependencies))

        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X2 -> [A].X3', '[A].X4 -> [A].X1',
                               '[A].X1 -> [A].X4', '[A].X1 -> [A].X3', '[A].X1 -> [A].X2']
        self.assertEqual(0.5, ModelEvaluation.orientedRecall(model, learnedDependencies))

        # true model has three dependencies
        dependencies = ['[A].X1 -> [A].X2', '[A].X3 -> [A].X2', '[A].X2 -> [A].X4']
        model = Model(schema, dependencies)
        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X2 -> [A].X4', '[A].X1 -> [A].X4']
        self.assertEqual(2/3, ModelEvaluation.orientedRecall(model, learnedDependencies))

        learnedDependencies = ['[A].X3 -> [A].X2', '[A].X2 -> [A].X4', '[A].X4 -> [A].X2',
                               '[A].X1 -> [A].X4', '[A].X4 -> [A].X1']
        self.assertEqual(1/3, ModelEvaluation.orientedRecall(model, learnedDependencies))


    def testBadInput(self):
        model = Model(Schema(), [])
        for method in [ModelEvaluation.skeletonPrecision, ModelEvaluation.skeletonRecall,
                       ModelEvaluation.orientedPrecision,
                       ModelEvaluation.orientedRecall]:

            TestUtil.assertRaisesMessage(self, Exception, "learnedDependencies must be a list of RelationalDependencies "
                                                          "or parseable RelationalDependency strings",
                 method, model, None)

            TestUtil.assertRaisesMessage(self, Exception, "learnedDependencies must be a list of RelationalDependencies "
                                                          "or parseable RelationalDependency strings",
                 method, model, 'dependency')


if __name__ == '__main__':
    unittest.main()
