import json
from ..model import RelationalValidity
from ..model.Schema import Schema
from ..model.Schema import SchemaItem
from ..model.RelationalDependency import RelationalVariable
from ..model import ParserUtil
import networkx as nx

class Model(object):
    """
    schema: the underlying schema for this model. Used to check consistency of entities/relationships/attributes
        involved in the model's dependencies.

    dependencies: a list of relational dependencies in shorthand, e.g., '[A, AB, B].Y -> [A].X' (see ParserUtil).
        If any variables have no parents, then they won't show up in dependencies, but will automatically be put into
        attrToParents internally.
    """
    def __init__(self, schema, dependencies, allowCycles=True, relationalDependencyChecker=RelationalValidity.checkRelationalDependencyValidity):

        # Internal consistency of model: dependencies and distribution parents must match
        if len(dependencies) != len(set(dependencies)):
            raise Exception("Found duplicate dependency")

        # Consistency of model against schema. Make sure all entities, relationships, and attributes in
        # schema are represented in model.
        self.dependencies = [ParserUtil.parseRelDep(relDepStr) for relDepStr in dependencies]
        for relDep in self.dependencies:
            relationalDependencyChecker(schema, relDep)

        # Dependencies are consistent with schema, pull off children attributes
        attrRelVars = set([relDep.relVar2 for relDep in self.dependencies])

        # Consistency of schema against model. Make sure all attributes in schema are represented in model.
        for item in schema.getSchemaItems():
            for attrName in [attr.name for attr in item.attributes]:
                attrRelVar = RelationalVariable([item.name], attrName)
                if attrRelVar not in attrRelVars:
                    attrRelVars.add(attrRelVar)

        self.dag = nx.DiGraph()
        # add dag nodes for implicit existence attributes for each entity
        for entity in schema.getEntities():
            self.dag.add_node(RelationalVariable([entity.name], SchemaItem.EXISTS_ATTR_NAME))
        # add dag node for implicit existence attributes for each relationship
        # add precedence edges from each entity existence to the existence of all relationships it participates in
        for relationship in schema.getRelationships():
            relExistNode = RelationalVariable([relationship.name], SchemaItem.EXISTS_ATTR_NAME)
            self.dag.add_node(relExistNode)
            self.dag.add_edge(RelationalVariable([relationship.entity1Name], SchemaItem.EXISTS_ATTR_NAME), relExistNode)
            self.dag.add_edge(RelationalVariable([relationship.entity2Name], SchemaItem.EXISTS_ATTR_NAME), relExistNode)
        # add dag nodes for each attribute and add precedence edges from the attribute's schema item's existence node
        for attrRelVar in attrRelVars:
            self.dag.add_node(attrRelVar)
            if not attrRelVar.isExistence(): # otherwise, creates edge with same from and to
                self.dag.add_edge(RelationalVariable([attrRelVar.getBaseItemName()], SchemaItem.EXISTS_ATTR_NAME), attrRelVar)
        # add dag edges for each dependency and precendence edges for every schema item's existence that shows up
        # in the parent relational path
        for relDep in self.dependencies:
            parentAttrRelVar = RelationalVariable([relDep.relVar1.getTerminalItemName()], relDep.relVar1.attrName)
            self.dag.add_edge(parentAttrRelVar, relDep.relVar2)
            for schemaItemName in relDep.relVar1.path[1:]: # skip base item, already in dag
                self.dag.add_edge(RelationalVariable([schemaItemName], SchemaItem.EXISTS_ATTR_NAME), relDep.relVar2)

        if not allowCycles and not nx.algorithms.dag.is_directed_acyclic_graph(self.dag):
            raise Exception("dependencies encodes a cycle among relational variables")

        self.schema = schema


    def __key(self):
        return self.schema._Schema__key(), tuple(sorted([str(relDep) for relDep in self.dependencies])) # prepending attribute __key with class name to fix namespace issue


    def __eq__(self, other):
        return isinstance(other, Model) and self.__key() == other.__key()


    def __hash__(self):
        return hash(self.__key())


    def getItemRelVars(self):
        return [itemRelVar for itemRelVar in self.dag.nodes() if itemRelVar.isExistence()]


    def getAttrRelVars(self):
        return [attrRelVar for attrRelVar in self.dag.nodes() if not attrRelVar.isExistence()]


    def getNextItemOrAttribute(self):
        """
        Returns the relational variable (read: attributes) in topological order of the DAG.
        Useful for making sure parents are generated before their children.
        """
        for relVar in nx.algorithms.dag.topological_sort(self.dag):
            yield relVar


    def toFile(self, fileName):
        """
        Model export does not save the schema, just the dependencies.
        """
        with open(fileName, 'w') as file:
            json.dump({'dependencies': [str(relDep) for relDep in self.dependencies]}, file, indent='\t')


    @classmethod
    def fromFile(cls, schemaFileName, modelFileName):
        schema = Schema.fromFile(schemaFileName)
        with open(modelFileName, 'r') as file:
            modelDict = json.load(file)
        return Model(schema, modelDict['dependencies'])