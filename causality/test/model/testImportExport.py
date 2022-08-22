import unittest
from causality.model.Model import Model
from causality.test import TestUtil
from causality.model.Schema import Attribute
from causality.model.Schema import Schema
from causality.model.Schema import Entity
from causality.model.Schema import Relationship

class TestImportExport(unittest.TestCase):

    def testEntityToJsonDict(self):
        a = Entity('A')
        jsonDict = a.toJsonDict()
        self.assertEqual({'name': 'A', 'attributes': []}, jsonDict)

        b = Entity('B')
        jsonDict = b.toJsonDict()
        self.assertEqual({'name': 'B', 'attributes': []}, jsonDict)

        a.addAttribute('X1', Attribute.INTEGER)
        jsonDict = a.toJsonDict()
        self.assertEqual('A', jsonDict['name'])

        attrList = jsonDict['attributes']
        self.assertEqual(1, len(attrList))
        self.assertEqual(attrList[0], {'name': 'X1', 'dataType': Attribute.INTEGER})

        a.addAttribute('X2', Attribute.INTEGER)
        jsonDict = a.toJsonDict()
        self.assertEqual('A', jsonDict['name'])

        attrList = jsonDict['attributes']
        self.assertEqual(2, len(attrList))
        self.assertEqual(attrList[0], {'name': 'X1', 'dataType': Attribute.INTEGER})
        self.assertEqual(attrList[1], {'name': 'X2', 'dataType': Attribute.INTEGER})


    def testRelationshipsToJsonDict(self):
        ab = Relationship('AB', ('A', Schema.MANY), ('B', Schema.ONE))
        jsonDict = ab.toJsonDict()
        self.assertEqual({'name': 'AB', 'attributes': [],
                          'cardinalities': {'entity1Name': 'A', 'entity1Card': Schema.MANY,
                                            'entity2Name': 'B', 'entity2Card': Schema.ONE}}, jsonDict)

        bc = Relationship('BC', ('B', Schema.ONE), ('C', Schema.MANY))
        bc.addAttribute('YZ1', Attribute.INTEGER)
        bc.addAttribute('YZ2', Attribute.INTEGER)
        bc.addAttribute('YZ3', Attribute.INTEGER)
        jsonDict = bc.toJsonDict()
        self.assertEqual('BC', jsonDict['name'])

        attrList = jsonDict['attributes']
        self.assertEqual(3, len(attrList))
        self.assertEqual(attrList[0], {'name': 'YZ1', 'dataType': Attribute.INTEGER})
        self.assertEqual(attrList[1], {'name': 'YZ2', 'dataType': Attribute.INTEGER})
        self.assertEqual(attrList[2], {'name': 'YZ3', 'dataType': Attribute.INTEGER})

        self.assertEqual({'entity1Name': 'B', 'entity1Card': Schema.ONE, 'entity2Name': 'C', 'entity2Card': Schema.MANY},
                                                jsonDict['cardinalities'])


    def testSchemaToJsonDict(self):
        schema = Schema()
        jsonDict = schema.toJsonDict()
        self.assertEqual({'entities': [], 'relationships': []}, jsonDict)

        schema.addEntity('A')
        schema.addEntity('B')
        schema.addEntity('C')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.MANY))
        schema.addRelationship('BC', ('B', Schema.MANY), ('C', Schema.MANY))
        schema.addAttribute('A', 'X', Attribute.INTEGER)
        schema.addAttribute('B', 'Y', Attribute.INTEGER)
        schema.addAttribute('C', 'Z', Attribute.INTEGER)
        schema.addAttribute('AB', 'XY1', Attribute.INTEGER)
        schema.addAttribute('AB', 'XY2', Attribute.INTEGER)
        schema.addAttribute('BC', 'YZ', Attribute.INTEGER)
        
        jsonDict = schema.toJsonDict()
        TestUtil.assertUnorderedListEqual(self, ['entities', 'relationships'], list(jsonDict.keys()))
        for entJsonDict in jsonDict['entities']:
            TestUtil.assertUnorderedListEqual(self, ['name', 'attributes'], list(entJsonDict.keys()))
            if entJsonDict['name'] == 'A':
                entJsonAttrList = entJsonDict['attributes']
                self.assertEqual(1, len(entJsonAttrList))
                self.assertEqual({'name': 'X', 'dataType': Attribute.INTEGER}, entJsonAttrList[0])
            elif entJsonDict['name'] == 'B':
                entJsonAttrList = entJsonDict['attributes']
                self.assertEqual(1, len(entJsonAttrList))
                self.assertEqual({'name': 'Y', 'dataType': Attribute.INTEGER}, entJsonAttrList[0])
            elif entJsonDict['name'] == 'C':
                entJsonAttrList = entJsonDict['attributes']
                self.assertEqual(1, len(entJsonAttrList))
                self.assertEqual({'name': 'Z', 'dataType': Attribute.INTEGER}, entJsonAttrList[0])
            else:
                self.fail("Unknown entity name")

        for relJsonDict in jsonDict['relationships']:
            TestUtil.assertUnorderedListEqual(self, ['name', 'attributes', 'cardinalities'], list(relJsonDict.keys()))
            if relJsonDict['name'] == 'AB':
                relJsonAttrList = relJsonDict['attributes']
                self.assertEqual(2, len(relJsonAttrList))
                self.assertEqual({'name': 'XY1', 'dataType': Attribute.INTEGER}, relJsonAttrList[0])
                self.assertEqual({'name': 'XY2', 'dataType': Attribute.INTEGER}, relJsonAttrList[1])
                self.assertEqual(schema.getRelationship('AB').toJsonDict()['cardinalities'], relJsonDict['cardinalities'])
            elif relJsonDict['name'] == 'BC':
                relJsonAttrList = relJsonDict['attributes']
                self.assertEqual(1, len(relJsonAttrList))
                self.assertEqual({'name': 'YZ', 'dataType': Attribute.INTEGER}, relJsonAttrList[0])
                self.assertEqual(schema.getRelationship('BC').toJsonDict()['cardinalities'], relJsonDict['cardinalities'])
            else:
                self.fail("Unknown relationship name")


    def testSchemaFileIO(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addEntity('C')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.MANY))
        schema.addRelationship('BC', ('B', Schema.MANY), ('C', Schema.MANY))
        schema.addAttribute('A', 'X', Attribute.INTEGER)
        schema.addAttribute('B', 'Y', Attribute.INTEGER)
        schema.addAttribute('C', 'Z', Attribute.INTEGER)
        schema.addAttribute('AB', 'XY1', Attribute.INTEGER)
        schema.addAttribute('AB', 'XY2', Attribute.INTEGER)
        schema.addAttribute('BC', 'YZ', Attribute.INTEGER)

        schema.toFile('schema.json')
        loadedSchema = Schema.fromFile('schema.json')
        self.assertEqual(schema, loadedSchema)


    def testModelFileIO(self):
        schema = Schema()
        schema.addEntity('A')
        schema.addEntity('B')
        schema.addEntity('C')
        schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.MANY))
        schema.addRelationship('BC', ('B', Schema.MANY), ('C', Schema.MANY))
        schema.addAttribute('A', 'X', Attribute.INTEGER)
        schema.addAttribute('B', 'Y', Attribute.INTEGER)
        schema.addAttribute('C', 'Z', Attribute.INTEGER)
        schema.addAttribute('AB', 'XY1', Attribute.INTEGER)
        schema.addAttribute('AB', 'XY2', Attribute.INTEGER)
        schema.addAttribute('BC', 'YZ', Attribute.INTEGER)

        schema.toFile('schema.json')

        model = Model(schema, [])
        model.toFile('model.json')
        loadedModel = Model.fromFile('schema.json', 'model.json')
        self.assertEqual(model, loadedModel)

        model = Model(schema, ['[A, AB, B].Y -> [A].X'])
        model.toFile('model.json')
        loadedModel = Model.fromFile('schema.json', 'model.json')
        self.assertEqual(model, loadedModel)

        model = Model(schema, ['[A, AB, B].Y -> [A].X', '[AB, B, BC, C].Z -> [AB].exists'])
        model.toFile('model.json')
        loadedModel = Model.fromFile('schema.json', 'model.json')
        self.assertEqual(model, loadedModel)