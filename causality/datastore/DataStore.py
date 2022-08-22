class DataStore(object):
    """
    Abstract class
    """

    def __init__(self):
        self.aggrToHandler = {}

    #
    # Basic operations
    #

    def addTable(self, schema, schemaItemName):
        """
        Adds the table named schemaItemName. NB: The handling of SQL identifier case is RDBMS-dependent! While case in
        queries doesn't seem to matter, table name case does vary. Tips: 1) Do NOT use quoted identifiers, esp. table
        names, e.g., this is bad: CREATE TABLE "Foo" (id INTEGER); Better is: CREATE TABLE foo. 2) Testing of table
        names should lowercase or upcase result of below getTableNames() for consistency.
        """
        raise NotImplementedError()


    def close(self):
        """
        Does whatever resource cleanup might be necessary. Default is nothing.
        """
        pass


    def getAllRows(self, schema, tableName, columnNames):
        """
        Returns a list of tuples, ordered by columnNames.
        """
        raise NotImplementedError()


    def insertRowsWithValues(self, tableName, columnNames, rowsWithValues):
        raise NotImplementedError()


    def updateRowValue(self, tableName, rowId, colName, colValue):
        raise NotImplementedError()


    def deleteRow(self, tableName, rowId):
        raise NotImplementedError()


    def getTableNames(self):
        """
        Returns a list of all my tables. NB: See addTable() docs above for notes on case sensitivity re: returned
        table names, which varies by RDBMS. So this method converts class-specific result to lower case.
        """
        return list(map(str.lower, self._getTableNames()))


    def _getTableNames(self):
        """
        Internal method called by above getTableNames(). Returned case does not matter - coverted by same.
        """
        raise NotImplementedError()


    def dropAllTables(self):
        """
        WARNING! Deletes all of my tables.
        """
        raise NotImplementedError()


    def rollback(self):
        """
        Used by SqlDataStore.
        """
        raise NotImplementedError()


    #
    # Higher-level relational path/var methods.
    #
    def getValuesForRelVarAggrs(self, schema, baseItemName, relVarAggrs):
        """
        Gets the values for a list of aggregators over relational variables for _all_ base item instances including
        those that might have no value for some aggregator
        baseItemName is the perspective of every relVarAggr in relVarAggrs
        relVarAggrs is a list of aggregators over relational variables
        """
        raise NotImplementedError()


    def _validateRelPath(self, schema, relPath, tableNames):
        # check that all tables on relPath exist
        for item in relPath:
            if item.lower() not in tableNames:
                raise Exception("No table found for item {!r} in path {}".format(item, relPath))

        # check that path items alternate between entities and relationships
        for item1Name, item2Name in zip(relPath[:-1], relPath[1:]):
            if type(schema.getSchemaItem(item1Name)) == type(schema.getSchemaItem(item2Name)):
                raise Exception("invalid item1Name {!r} and item2Name {!r} in relational path: types must alternate "
                                "between entities and relationships".format(item1Name, item2Name))


    def _verifyRelVarAttrsExist(self, schema, relVar):
        relPath = relVar.path
        variableName = 'id' if relVar.isExistence() else relVar.attrName
        if variableName not in [attr.name for attr in
                                schema.getSchemaItem(relPath[-1]).attributes] and variableName != 'id':
            raise Exception("Attribute {!r} does not exist for item {!r}".format(variableName, relPath[-1]))


