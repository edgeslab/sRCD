import collections
import itertools
from ..model.Aggregator import IdentityAggregator, NonRandomChoiceAggregator, AverageAggregator, CountAggregator, MaxAggregator, CaseAggregator
from ..datastore.DataStore import DataStore


class InMemoryDataStore(DataStore):

    def __init__(self):
        super().__init__()
        self.tableToRows = {}     # { tableName -> { rowId -> {'id': rowId, 'colName': colVal, ...} } }


    #
    # Basic operations
    #

    def addTable(self, schema, schemaItemName):
        if not schema.hasSchemaItem(schemaItemName):
            raise Exception("Schema does not contain schema item {!r}.".format(schemaItemName))
        if schemaItemName in self.tableToRows:
            raise Exception("Table {!r} already exists.".format(schemaItemName))
        self.tableToRows[schemaItemName] = {}


    def getAllRows(self, schema, schemaItemName, columnNames):
        if schemaItemName not in self.tableToRows:
            raise Exception("Table {!r} does not exist yet.".format(schemaItemName))

        schemaItem = schema.getSchemaItem(schemaItemName)
        schemaItemAttrNames = [attr.name for attr in schemaItem.attributes]
        for columnName in columnNames:
            if not columnName.endswith('id'):
                if not columnName in schemaItemAttrNames:
                    raise Exception("Column name {!r} does not exist in table {!r}".format(columnName, schemaItemName))

        rows = []
        for row in self.tableToRows[schemaItemName].values():
            vals = []
            for columnName in columnNames:
                if columnName in row:
                    vals.append(row[columnName])
                else:
                    vals.append(None)
            rows.append(tuple(vals))
        return rows


    def insertRowsWithValues(self, tableName, columnNames, rowsWithValues):
        """
        columnNames: column names in insert order
        rowsWithValues: list of lists with values for each columnName in specified order
        """
        if tableName not in self.tableToRows:
            raise Exception("Table {!r} does not exist yet.".format(tableName))
        if not columnNames:
            raise Exception("Column names are empty.")
        if 'id' not in columnNames:
            raise Exception("Column name 'id' not found.")
        if len(columnNames) != len(set(columnNames)):
            raise Exception("Duplicate column name found.")

        idColumnIdx = columnNames.index('id')
        for rowWithValue in rowsWithValues:
            if len(rowWithValue) != len(columnNames):
                raise Exception("Column count and row count differ: {}, {}".format(rowWithValue, columnNames))
            rowId = rowWithValue[idColumnIdx]
            if rowId in self.tableToRows[tableName]:
                raise Exception("Row id {} already exists.".format(rowId))
            self.tableToRows[tableName][rowId] = {}
            for columnIdx, columnName in enumerate(columnNames):
                self.tableToRows[tableName][rowId][columnName] = rowWithValue[columnIdx]


    def updateRowValue(self, tableName, rowId, colName, colValue):
        if tableName not in self.tableToRows:
            raise Exception("Table {!r} does not exist yet.".format(tableName))
        if rowId in self.tableToRows[tableName]:
            self.tableToRows[tableName][rowId][colName] = colValue


    def deleteRow(self, tableName, rowId):
        if tableName not in self.tableToRows:
            raise Exception("Table {!r} does not exist yet.".format(tableName))
        if rowId in self.tableToRows[tableName]:
            del(self.tableToRows[tableName][rowId])


    def _getTableNames(self):
        return list(self.tableToRows.keys())


    def dropAllTables(self):
        """
        WARNING! Deletes all of my tables.
        """
        self.tableToRows = {}


    def rollback(self):
        pass    # no-op


    #
    # Aggregator methods
    #
    def aggregate(self, aggregator, valuesList):
        if isinstance(aggregator, IdentityAggregator):
            return None if not valuesList else valuesList
        elif isinstance(aggregator, NonRandomChoiceAggregator):
            return None if not valuesList else valuesList[0]
        elif isinstance(aggregator, AverageAggregator):
            return None if not valuesList else sum(valuesList) / len(valuesList)
        elif isinstance(aggregator, CountAggregator):
            return len(valuesList)
        elif isinstance(aggregator, MaxAggregator):
            return None if not valuesList else max(valuesList)
        elif isinstance(aggregator, CaseAggregator):
            nestedVal = self.aggregate(aggregator.agg, valuesList)
            for caseStatement in aggregator.caseStatements:
                if caseStatement[0] == CaseAggregator.LESS_THAN:
                    if nestedVal < caseStatement[1]:
                        return caseStatement[2]
                elif caseStatement[0] == CaseAggregator.LESS_THAN_OR_EQUAL:
                    if nestedVal <= caseStatement[1]:
                        return caseStatement[2]
                elif caseStatement[0] == CaseAggregator.GREATER_THAN:
                    if nestedVal > caseStatement[1]:
                        return caseStatement[2]
                elif caseStatement[0] == CaseAggregator.GREATER_THAN_OR_EQUAL:
                    if nestedVal >= caseStatement[1]:
                        return caseStatement[2]
                elif caseStatement[0] == CaseAggregator.EQUALS:
                    if nestedVal == caseStatement[1]:
                        return caseStatement[2]
                elif caseStatement[0] == CaseAggregator.NOT_EQUALS:
                    if nestedVal != caseStatement[1]:
                        return caseStatement[2]
                else:
                    continue
            return aggregator.elseValue
        else:
            raise Exception("Unknown aggregator {!r}".format(str(aggregator)))

    #
    # Higher-level relational path/var method.
    #

    def getValuesForRelVarAggrs(self, schema, baseItemName, relVarAggrs):
        if baseItemName not in self.tableToRows:
            raise Exception("No table found for base item {!r}".format(baseItemName))

        for initialInstanceId in self.tableToRows[baseItemName].keys():
            values = []
            for relVarAggr in relVarAggrs:
                values.append(self.aggregate(relVarAggr, self._getTerminalSetForRelVar(schema, relVarAggr.relVar, initialInstanceId)))
            # If any values are iterable (other than a single string) of multiple values, then they need to be replicated
            # We wrap non-iterable values as tuples in order to work with itertools.product (the Cartesian product)
            wrappedValues = []
            for value in values:
                if isinstance(value, str) or not isinstance(value, collections.Iterable):
                    wrappedValues.append(tuple([value]))
                else:
                    wrappedValues.append(value)
            for values in itertools.product(*wrappedValues):
                yield initialInstanceId, list(values)


    #
    # Private supporting methods for getValuesForRelVarAggrs() implementation.
    #

    def _getTerminalSetForRelPath(self, schema, relPath, initialInstanceId):
        """
        Returns the terminal set for relPath starting at initialInstanceId. relPath is an alternating sequence
        of entity and relationship names. initialInstanceId is an instance of the first item on relPath, and the returned
        terminal set is a set of instance ids from the final (i.e., 'terminal') item on the path.
        """
        # singleton relational paths
        if len(relPath) == 1:
            return {initialInstanceId}

        currentIds = {initialInstanceId}
        visitedIdMap = {relPath[0]: {initialInstanceId}}
        nextCurrentIds = set()      # alternates between entity ids and relationship ids
        for item1Name, item2Name in zip(relPath[:-1], relPath[1:]):
            visitedIdMap.setdefault(item2Name, set())
            if item2Name in [rel.name for rel in schema.getRelationships()]:    # item1 is an entity
                entName, relName = item1Name, item2Name
                for relRow in self.getAllRows(schema, relName, ['id', entName + '_id']):
                    if relRow[1] in currentIds:
                        nextRelId = relRow[0]
                        if nextRelId not in visitedIdMap[relName]:
                            nextCurrentIds.add(nextRelId)
                            visitedIdMap[relName].add(nextRelId)
            else:   # item1 is a relationship
                entName, relName = item2Name, item1Name
                for relRow in self.getAllRows(schema, relName, ['id', entName + '_id']):
                    if relRow[0] in currentIds:
                        nextEntId = relRow[1]
                        if nextEntId not in visitedIdMap[entName]:
                            nextCurrentIds.add(nextEntId)
                            visitedIdMap[entName].add(nextEntId)
            currentIds = nextCurrentIds
            nextCurrentIds = set()
        return currentIds


    def _getTerminalSetForRelVar(self, schema, relVar, initialInstanceId):
        """
        Gets variable values of the ids reached from initialInstanceId in relationalVariable, but prunes
        out any missing values.
        """
        relPath = relVar.path
        self._validateRelPath(schema, relPath, self.getTableNames())
        self._verifyRelVarAttrsExist(schema, relVar)

        if relVar.isExistence():
            return self._getTerminalSetForRelPath(schema, relPath, initialInstanceId)

        variableName = relVar.attrName
        variableValues = []
        for terminalId in self._getTerminalSetForRelPath(schema, relPath, initialInstanceId):
            try:
                variableValues.append(self.tableToRows[relPath[-1]][terminalId][variableName])
            except Exception:
                pass
        return variableValues