import unittest
from causality.modelspace import RelationalSpace
from causality.test import TestUtil
from causality.model.Model import Model
from causality.model.Schema import Schema
from causality.modelspace import ModelGenerator

class TestRandomModelGenerator(unittest.TestCase):

    def testOneEntity(self):
        schema = Schema()
        schema.addEntity('A')
        model = ModelGenerator.generateModel(schema, 0, 0)
        self.assertIsInstance(model, Model)
        TestUtil.assertUnorderedListEqual(self, [], model.dependencies)

        schema.addAttribute('A', 'X1')
        schema.addAttribute('A', 'X2')
        dependencies = ['[A].X1 -> [A].X2', '[A].X2 -> [A].X1']
        model = ModelGenerator.generateModel(schema, 0, 1, dependencies=dependencies,
            randomPicker=lambda depsList, _: [depsList[0]])
        self.assertIsInstance(model, Model)
        TestUtil.assertUnorderedListEqual(self, ['[A].X1 -> [A].X2'], [str(dep) for dep in model.dependencies])

        model = ModelGenerator.generateModel(schema, 0, 1, dependencies=dependencies,
            randomPicker=lambda depsList, _: [depsList[-1]])
        self.assertIsInstance(model, Model)
        TestUtil.assertUnorderedListEqual(self, ['[A].X2 -> [A].X1'], [str(dep) for dep in model.dependencies])

        schema.addAttribute('A', 'X3')
        dependencies = ['[A].X1 -> [A].X2', '[A].X2 -> [A].X3', '[A].X2 -> [A].X1', '[A].X3 -> [A].X2',
                        '[A].X1 -> [A].X3', '[A].X3 -> [A].X1']
        model = ModelGenerator.generateModel(schema, 0, 2, dependencies=dependencies,
            randomPicker=lambda depsList, _: [depsList[0]])
        self.assertIsInstance(model, Model)
        TestUtil.assertUnorderedListEqual(self, ['[A].X1 -> [A].X2', '[A].X2 -> [A].X3'],
            [str(dep) for dep in model.dependencies])

        # tests that model iteratively attempts to add dependencies, throwing out those that create conflicts
        dependencies = ['[A].X1 -> [A].X2', '[A].X2 -> [A].X1', '[A].X2 -> [A].X3', '[A].X3 -> [A].X2',
                       '[A].X1 -> [A].X3', '[A].X3 -> [A].X1']
        model = ModelGenerator.generateModel(schema, 0, 2, dependencies=dependencies,
            randomPicker=lambda depsList, _: [depsList[0]])
        self.assertIsInstance(model, Model)
        TestUtil.assertUnorderedListEqual(self, ['[A].X1 -> [A].X2', '[A].X2 -> [A].X3'],
            [str(dep) for dep in model.dependencies])


    def testTwoEntities(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.MANY))
        schema.addAttribute('A', 'X')
        schema.addAttribute('B', 'Y')
        schema.addAttribute('AB', 'XY')
        dependencies = ['[A, AB].XY -> [A].X', '[A, AB].exists -> [A].X', '[AB, A].X -> [AB].XY',
                        '[AB, A].X -> [AB].exists', '[AB, B].Y -> [AB].XY', '[AB, B].Y -> [AB].exists',
                        '[B, AB].XY -> [B].Y', '[B, AB].exists -> [B].Y', '[A, AB, B].Y -> [A].X',
                        '[AB, B, AB].exists -> [AB].XY', '[B, AB, A].X -> [B].Y']
        model = ModelGenerator.generateModel(schema, 2, 3, dependencies=dependencies,
            randomPicker=lambda depsList, _: [depsList[0]])
        self.assertIsInstance(model, Model)
        TestUtil.assertUnorderedListEqual(self, ['[A, AB].XY -> [A].X', '[A, AB].exists -> [A].X', '[AB, B].Y -> [AB].XY'],
            [str(dep) for dep in model.dependencies])


    def testChoseTooFewDependencies(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X1')
        schema.addAttribute('A', 'X2')
        schema.addAttribute('A', 'X3')
        dependencies = ['[A].X3 -> [A].X1', '[A].X1 -> [A].X2', '[A].X2 -> [A].X3', '[A].X2 -> [A].X1',
                        '[A].X1 -> [A].X3', '[A].X3 -> [A].X2']
        TestUtil.assertRaisesMessage(self, Exception, "Could not generate a model: failed to find a model with 4 dependenc[y|ies]",
                    ModelGenerator.generateModel, schema, 0, 4, dependencies=dependencies,
                    randomPicker=lambda depsList, _: [depsList[0]])


    def testTooManyDependencies(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X1')
        TestUtil.assertRaisesMessage(self, Exception, "Could not generate a model: not enough dependencies to draw from",
            ModelGenerator.generateModel, schema, 0, 1)

        schema.addAttribute('A', 'X2')
        TestUtil.assertRaisesMessage(self, Exception, "Could not generate a model: failed to find a model with 2 dependenc[y|ies]",
                    ModelGenerator.generateModel, schema, 0, 2)

        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.MANY))
        schema.addAttribute('A', 'X')
        schema.addAttribute('B', 'Y')
        schema.addAttribute('AB', 'XY')
        TestUtil.assertRaisesMessage(self, Exception, "Could not generate a model: failed to find a model with 7 dependenc[y|ies]",
                    ModelGenerator.generateModel, schema, 2, 7,
                    dependencies=RelationalSpace.getRelationalDependencies(schema, 1, includeExistence=True))


    def testMaximumNumParentsArgument(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addAttribute('A', 'X1')
        schema.addAttribute('A', 'X2')
        schema.addAttribute('A', 'X3')
        dependencies = ['[A].X1 -> [A].X3', '[A].X2 -> [A].X3']
        TestUtil.assertRaisesMessage(self, Exception, "Could not generate a model: failed to find a model with 2 dependenc[y|ies]",
                                     ModelGenerator.generateModel, schema, 0, 2, maxNumParents=1, dependencies=dependencies)

        schema.addAttribute('A', 'X4')
        dependencies = ['[A].X1 -> [A].X3', '[A].X2 -> [A].X3', '[A].X4 -> [A].X3']
        TestUtil.assertRaisesMessage(self, Exception, "Could not generate a model: failed to find a model with 3 dependenc[y|ies]",
                                     ModelGenerator.generateModel, schema, 0, 3, maxNumParents=2, dependencies=dependencies)



if __name__ == '__main__':
    unittest.main()
