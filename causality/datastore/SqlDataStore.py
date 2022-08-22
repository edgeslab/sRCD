import csv
import logging
import re
from ..model.Aggregator import IdentityAggregator, AverageAggregator, CountAggregator, \
    NonRandomChoiceAggregator, CaseAggregator, MaxAggregator
from ..model.Schema import Attribute
from ..datastore.DataStore import DataStore


logger = logging.getLogger(__name__)


class SqlDataStore(DataStore):
    """
    Abstract SQL DataStore. NB: It is very important that 1) all queries that pass values should use the db2-api
    'pyformat' paramstyle (which is easiest to translate to other styles), and 2) all users of this class execute
    those queries via this class's execute() and executemany() methods, and *not* via direct access to my _connection.
    This is because my methods do dialect translation, including paramstyle changes, on the passed query. Also, it is
    recommended you call close() when done with instances. Note that we handle transactions manually, rather than
    using autocommit. This relies on connections' default being autocommit = off.
    """
    OPERATOR_TO_SQL = {CaseAggregator.LESS_THAN: '<', CaseAggregator.LESS_THAN_OR_EQUAL: '<=',
                       CaseAggregator.GREATER_THAN: '>', CaseAggregator.GREATER_THAN_OR_EQUAL: '>=',
                       CaseAggregator.EQUALS: '=', CaseAggregator.NOT_EQUALS: '<>'}

    ATTRIBUTE_TYPE_TO_COLUMN_TYPE = {Attribute.INTEGER: 'INTEGER', Attribute.STRING: 'TEXT', Attribute.FLOAT: 'REAL'}


    def __init__(self, connection, executeParamstyle, executeManyParamstyle):
        """
        connection: an open connection to a database
        executeParamstyle: the destination paramstyle ( http://www.python.org/dev/peps/pep-0249/#paramstyle ) to
            translate queries to from the 'pyformat' style that's used everywhere in this code. used in execute() calls
        executeManyParamstyle: like executeParamstyle, but for executemany() calls where values are passed as lists
        """
        super().__init__()
        self._connection = connection
        self._executeParamstyle = executeParamstyle
        self._executeManyParamstyle = executeManyParamstyle


    #
    # Query execution methods. NB: Call these instead of direct _connection methods because these do dialect and
    # paramstyle translation.
    #

    def execute(self, query, paramDict=None, passedCursor=None):
        """
        Utility method for one-off command queries (i.e., those that don't return a value) that saves having to create
        a _connection. paramDict is the db2-api 'pyformat' paramstyle dictionary to look up :who-style parameters in
        query. Commits and debug logs the query to logger. Uses passedCursor if present, which is *not* closed here.
        Returns nothing.
        """
        query = self._translateQuery(query, self._executeParamstyle)
        cursor = passedCursor if passedCursor else self._connection.cursor()
        logger.debug('{};{}'.format(query, ' <- {}'.format(paramDict) if paramDict else ''))
        try:
            cursor.execute(query, paramDict if paramDict else {})
            self._connection.commit()
        except Exception as exception:
            self._connection.rollback()     # otherwise MySQL may deadlock (seen during unit tests, e.g., "Waiting for table metadata lock")
            raise exception                 # re-raise so others can handle
        finally:
            if not passedCursor:
                cursor.close()


    def _translateQuery(self, query, destinationParamstyle):
        """
        Translates query from the 'pyformat' style that's used everywhere in this code to destinationParamstyle. NB: As
        stated above, it is *essential* that this method be called on all executed queries. An example of translating
        from native 'pyformat' (PostgreSQL and MySQL) to 'named' (sqlite3):

        "DELETE FROM MemberOf_temp WHERE NOT(rank <> %(compareVal)s) OR NOT(rank < %(compareVal)s)"
          ->
        "DELETE FROM MemberOf_temp WHERE NOT(rank <> :compareVal) OR NOT(rank < :compareVal)"

        """
        pyformatRegex = """
            %\(         # start delimiter
                (       # group
                  .*?   # variable name (non-greedy)
                )
            \)s         # end delimiter
            """
        if destinationParamstyle == 'pyformat':
            return query    # native style!
        elif destinationParamstyle == 'named':
            return re.sub(pyformatRegex, r':\1', query, flags=re.VERBOSE)
        elif destinationParamstyle == 'format':
            return re.sub(pyformatRegex, r'%s', query, flags=re.VERBOSE)
        elif destinationParamstyle == 'qmark':
            return re.sub(pyformatRegex, r'?', query, flags=re.VERBOSE)
        else:
            raise Exception("Cannot translate from {!r} paramstyle to {!r}".format('pyformat', self._executeParamstyle))

    #
    # CSV methods
    #

    def loadCsvFile(self, tableName, columnNames, csvFileName, csvFile, delimiter, fieldCount):
        """
        Loads one csv file into tableName, reading rows one-at-a-time in chunks. This approach works for all RDBMSs, but
        is relatively slow compared to (non-standard) bulk import features.
        """
        valParams = ', '.join(['%(col_{})s'.format(colNum) for colNum in range(fieldCount)])    # we use 'named' paramstyle instead of 'qmark' because 'named' is more consistent with 'pyformat' as supported by PostgreSQL and MySQL - will help with syntatic-level dialect transformation
        query = 'INSERT INTO {tableName} ({colNames}) VALUES ({valParams})'.format(
            tableName=tableName,
            colNames=', '.join(columnNames),
            valParams=valParams)
        csvReader = csv.reader(csvFile, delimiter=delimiter)
        self.executeInsertFromCsvReader(query, csvReader)


    def executeInsertFromCsvReader(self, insertQuery, csvReader, rowsChunkSize=10000):  # NB: magic number
        """
        Executes insertQuery on csvReader, breaking the read-in rows into chunks of rowsChunkSize rows. NB: Works by
        committing after each chunk is read, rather than deferring the commit until the end of loadCsvFiles(). The
        latter was actually a little slower.
        """
        insertQuery = self._translateQuery(insertQuery, self._executeManyParamstyle)
        cursor = self._connection.cursor()
        logger.debug(insertQuery + ';')
        while True:
            rows = self.readRowsFromIterator(csvReader, rowsChunkSize)
            if not rows:
                break
            try:
                cursor.executemany(insertQuery, rows)
            except Exception as exception:
                self._connection.rollback()     # see comment above re: MySQL
                raise exception
            self._connection.commit()
        cursor.close()


    def readRowsFromIterator(self, iterator, numRowsToRead):
        """
        Utility that reads and returns numRowsToRead from iterator, returning [] if iterator runs out. Call repeatedly
        to read large iterators in chunks.
        """
        rows = []
        for i in range(numRowsToRead):
            try:
                rows.append(next(iterator))
            except StopIteration:
                break
        return rows


    #
    # Utility methods
    #

    def printTable(self, tableName):
        cursor = self._connection.cursor()
        query = 'SELECT * FROM {tableName}'.format(tableName=tableName)
        cursor.execute(query)
        rows = cursor.fetchall()
        self._connection.rollback()     # NB: without this, psycopg2 hangs if you're trying to drop the table you're querying from
        cursor.close()
        print(tableName, "contains:", '\n', rows)


    def selectRows(self, selectQuery):
        cursor = self._connection.cursor()
        cursor.execute(selectQuery)
        rows = cursor.fetchall()
        self._connection.rollback()     # NB: without this, psycopg2 hangs if you're trying to drop the table you're querying from
        cursor.close()
        return rows


    def replaceTable(self, oldTableName, newTableName):
        self.execute('DROP TABLE {oldTable}'.format(oldTable=oldTableName))
        self.execute('ALTER TABLE {newTable} RENAME TO {oldTable}'.format(
            newTable=newTableName,
            oldTable=oldTableName))


    #
    # SqlDataStore API methods
    #

    def getColumnTypeForAttributeType(self, dataType):
        return self.ATTRIBUTE_TYPE_TO_COLUMN_TYPE[dataType]


    def _getTableNamesQuery(self):
        raise NotImplementedError()


    def _getIntegrityErrorClass(self):
        """
        Returns the db-api http://www.python.org/dev/peps/pep-0249/#integrityerror class to use to check PRIMARY KEY
        constraint violations (see _isPrimaryKeyUniquenessViolation()).
        """
        raise NotImplementedError()


    def _isPrimaryKeyUniquenessViolation(self, integrityError):
        """
        Abstract method called by insertRowsWithValues() that returns true if the passed exception instance as returned
        by _getIntegrityErrorClass() indicates that an inserted row's primary key was not unique.
        """
        raise NotImplementedError()


    def _cascadeRequired(self):
        """
        Returns whether CASCADE should be included in DROP TABLE commands. See comment in dropAllTables()
        """
        raise NotImplementedError()


    def _getSqlForNonIntegerDelete(self, columnName):
        """
        Returns an SQL expression suitable for a DELETE FROM ... WHERE clause. Implementations might need to define
        a helper function ('is_integer(), say) if one doesn't exist, which should be created if necessary in the
        implementation's initializer. Note: the SQL should test for the /not/ case, i.e., 'not is_integer({col})'.
        """
        raise NotImplementedError()


    def _getSqlForRandomDeleteWhere(self, probability, paramDict):
        """
        Returns an SQL expression suitable for a DELETE FROM ... WHERE random clause.
        probability: xx
        paramDict: is modified in-place as described in Filter.getSqlForDeleteWhere()
        """
        raise NotImplementedError()


    def _getIndexPrefixLengthSpec(self):
        """
        Returns an optional prefix length expression for indexing TEXT columns. Supports MySqlDataStore, which does not
        like TEXT column indexes, in general. http://stackoverflow.com/questions/13710170/blob-text-column-bestilling-used-in-key-specification-without-a-key-length
        """
        return ''


    def getCastDataTypeForSqlType(self, sqlDataType):
        """
        A function that MYSQL can override.
        """
        return sqlDataType


    #
    # DataStore Basic operations
    #
    def addTable(self, schema, schemaItemName):
        if not schema.hasSchemaItem(schemaItemName):
            raise Exception("Schema does not contain schema item {!r}.".format(schemaItemName))
        if schemaItemName.lower() in self.getTableNames():
            raise Exception("Table {!r} already exists.".format(schemaItemName))

        query = 'CREATE TABLE {schemaItemName} (id INTEGER PRIMARY KEY'.format(schemaItemName=schemaItemName)

        # add columns for foreign keys if schemaItemName is a relationship
        if schema.hasRelationship(schemaItemName):
            relationship = schema.getRelationship(schemaItemName)
            foreignKey1 = relationship.entity1Name + '_id'
            foreignKey2 = relationship.entity2Name + '_id'
            query += ',\n\t{foreignKey1} INTEGER'.format(foreignKey1=foreignKey1)
            query += ',\n\t{foreignKey2} INTEGER'.format(foreignKey2=foreignKey2)

        # add columns for each non-key attribute
        for attr in schema.getSchemaItem(schemaItemName).attributes:
            attrName = attr.name
            query += ',\n\t{attrName} {dataType}'.format(
                attrName=attrName,
                dataType=self.getColumnTypeForAttributeType(attr.dataType))

        # add foreign key constraints if it's a relationship
        if schema.hasRelationship(schemaItemName):
            relationship = schema.getRelationship(schemaItemName)
            foreignKey1 = relationship.entity1Name + '_id'
            foreignKey2 = relationship.entity2Name + '_id'
            query += ',\n\tFOREIGN KEY({foreignKey1}) REFERENCES {ent1Name} (id)'.format(
                foreignKey1=foreignKey1,
                ent1Name=relationship.entity1Name)
            query += ',\n\tFOREIGN KEY({foreignKey2}) REFERENCES {ent2Name} (id)'.format(
                foreignKey2=foreignKey2,
                ent2Name=relationship.entity2Name)
        query += ')'
        self.execute(query)


    def close(self):
        self._connection.close()


    def getAllRows(self, schema, schemaItemName, columnNames):
        if schemaItemName.lower() not in self.getTableNames():
            raise Exception("Table {!r} does not exist yet.".format(schemaItemName))

        schemaItem = schema.getSchemaItem(schemaItemName)
        schemaItemAttrNames = [attr.name for attr in schemaItem.attributes]
        for columnName in columnNames:
            if not columnName.endswith('id'):
                if not columnName in schemaItemAttrNames:
                    raise Exception("Column name {!r} does not exist in table {!r}".format(columnName, schemaItemName))

        query = 'SELECT {columnNames}\n FROM {schemaItemName}'.format(
            columnNames=', '.join(columnNames),
            schemaItemName=schemaItemName)

        cursor = self._connection.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        self._connection.rollback()     # NB: without this, psycopg2 hangs if you're trying to drop the table you're querying from
        cursor.close()
        return rows


    def insertRowsWithValues(self, schemaItemName, columnNames, rowsWithValues):
        if schemaItemName.lower() not in self.getTableNames():
            raise Exception("Table {!r} does not exist yet.".format(schemaItemName))
        if not columnNames:
            raise Exception("Column names are empty.")
        if 'id' not in columnNames:
            raise Exception("Column name 'id' not found.")
        if len(columnNames) != len(set(columnNames)):
            raise Exception("Duplicate column name found.")

        cursor = self._connection.cursor()
        valParams = ['%({})s'.format(colName) for colName in columnNames]
        query = 'INSERT INTO {schemaItemName} ({columnNames}) VALUES ({valParams})'.format(
            schemaItemName=schemaItemName,
            columnNames=', '.join(columnNames),
            valParams=', '.join(valParams))
        for rowWithValue in rowsWithValues:
            if len(rowWithValue) != len(columnNames):
                cursor.close()
                raise Exception("Column count and row count differ: {}, {}".format(rowWithValue, columnNames))

            paramDict = {columnName: val for columnName, val in zip(columnNames, rowWithValue)}
            try:
                logger.debug(query + ';')
                self.execute(query, paramDict, cursor)
            except self._getIntegrityErrorClass() as ie:
                if self._isPrimaryKeyUniquenessViolation(ie):
                    idColumnIdx = columnNames.index('id')
                    cursor.close()
                    raise Exception("Row id {} already exists.".format(rowWithValue[idColumnIdx]))
        self._connection.commit()
        cursor.close()


    def updateRowValue(self, schemaItemName, rowId, colName, colValue):
        if schemaItemName.lower() not in self.getTableNames():
            raise Exception("Table {!r} does not exist yet.".format(schemaItemName))

        query = 'UPDATE {schemaItemName} \n' \
                'SET {colName} = %(colValue)s \n' \
                'WHERE id = %(rowId)s'.format(
            schemaItemName=schemaItemName,
            colName=colName)
        paramDict = {'colValue': colValue, 'rowId': rowId}
        self.execute(query, paramDict)


    def deleteRow(self, schemaItemName, rowId):
        if schemaItemName.lower() not in self.getTableNames():
            raise Exception("Table {!r} does not exist yet.".format(schemaItemName))

        query = 'DELETE FROM {schemaItemName} \n' \
                'WHERE id = %(rowId)s'.format(
            schemaItemName=schemaItemName)
        paramDict = {'rowId': rowId}
        self.execute(query, paramDict)


    def _getTableNames(self):
        query = self._getTableNamesQuery()
        cursor = self._connection.cursor()
        logger.debug(query + ';')
        cursor.execute(query)
        tableNames = [row[0] for row in cursor.fetchall()]
        self._connection.rollback()     # NB: without this, psycopg2 hangs if you're trying to drop the table you're querying from
        cursor.close()
        return tableNames


    def dropAllTables(self):
        for tableName in self.getTableNames():
            query = 'DROP TABLE {tableName} {shouldCascade}'.format(
                tableName=tableName,
                shouldCascade='CASCADE' if self._cascadeRequired() else '')    # CASCADE required to avoid PostgreSQL's error: psycopg2.InternalError: cannot drop table a because other objects depend on it. DETAIL:  constraint ab_a_id_fkey on table ab depends on table a. HINT:  Use DROP ... CASCADE to drop the dependent objects too.
            self.execute(query)


    def rollback(self):
        self._connection.rollback()


    #
    # DataStore Aggregator methods
    #
    def _getSqlForAggregator(self, aggregator, columnName, paramDict):
        # NB: For each new aggregator, write a new test in testDataStoreRelationalPath.testSingleNonIdentityAggregator
        if isinstance(aggregator, IdentityAggregator):
            return columnName
        elif isinstance(aggregator, NonRandomChoiceAggregator):
            return self._getSqlForNonRandomChoiceAggregator(columnName)
        elif isinstance(aggregator, AverageAggregator):
            return 'AVG({columnName})'.format(columnName=columnName)
        elif isinstance(aggregator, CountAggregator):
            return 'COUNT({columnName})'.format(columnName=columnName)
        elif isinstance(aggregator, MaxAggregator):
            return 'MAX({columnName})'.format(columnName=columnName)
        elif isinstance(aggregator, CaseAggregator):
            nestedAggStr = self._getSqlForAggregator(aggregator.agg, columnName, paramDict)
            caseSql = 'CASE'
            for index, caseStatement in enumerate(aggregator.caseStatements):
                compareValParam = 'compareVal_{index}'.format(index=index)
                returnValParam = 'returnVal_{index}'.format(index=index)
                caseSql += ' WHEN {nestedAggStr} {opSql} %({compareValParam})s THEN %({returnValParam})s'.format(
                    nestedAggStr=nestedAggStr,
                    opSql=self.OPERATOR_TO_SQL[caseStatement[0]],
                    compareValParam=compareValParam,
                    returnValParam=returnValParam)
                paramDict[compareValParam] = caseStatement[1]
                paramDict[returnValParam] = caseStatement[2]
            elseValParam = 'elseVal'
            caseSql += ' ELSE {elseVal} END'.format(
                elseVal='NULL' if not aggregator.elseValue else '%({elseValParam})s'.format(elseValParam='elseVal'))
            if aggregator.elseValue:
                paramDict[elseValParam] = aggregator.elseValue
            return caseSql
        else:
            raise Exception("Unknown aggregator {!r}".format(str(aggregator)))


    def _getSqlForNonRandomChoiceAggregator(self, columnName):
        return '{columnName}'.format(columnName=columnName)


    def _isGroupBy(self, aggregator):
        if isinstance(aggregator, IdentityAggregator):
            return False
        elif isinstance(aggregator, NonRandomChoiceAggregator):
            return True
        elif isinstance(aggregator, AverageAggregator):
            return True
        elif isinstance(aggregator, CountAggregator):
            return True
        elif isinstance(aggregator, MaxAggregator):
            return True
        elif isinstance(aggregator, CaseAggregator):
            return self._isGroupBy(aggregator.agg)
        else:
            raise Exception("Unknown aggregator {!r}".format(str(aggregator)))


    #
    # DataStore Higher-level relational path/var method.
    #
    def getValuesForRelVarAggrs(self, schema, baseItemName, relVarAggrs):
        if baseItemName.lower() not in self.getTableNames():
            raise Exception("No table found for base item {!r}".format(baseItemName))

        paramDict = {}  # used to collect parameter values in SQL for execute()
        if relVarAggrs:
            baseQuery = self._getSQLQueryForIds(baseItemName)
            relVarAggrQueries = [self._getSQLQueryForRelVarAggr(relVarAggr, schema, paramDict) for relVarAggr in
                                 relVarAggrs]
            selectClause = 'SELECT relVarTable0.id AS id'
            for relVarAggrIdx, relVarAggr in enumerate(relVarAggrs):
                if isinstance(relVarAggr, CountAggregator):
                    selectClause += ', CASE WHEN relVarTable{relVarAggrIdxPlusOne}.val IS NULL ' \
                                    'THEN 0 ELSE relVarTable{relVarAggrIdxPlusOne}.val ' \
                                    'END'.format(relVarAggrIdxPlusOne=relVarAggrIdx + 1)
                else:
                    selectClause += ', relVarTable{relVarAggrIdxPlusOne}.val'.format(
                        relVarAggrIdxPlusOne=relVarAggrIdx + 1)

            fromClause = '\nFROM ({baseQuery}) relVarTable0'.format(baseQuery=baseQuery)
            for relVarAggrIdx, relVarAggrQuery in enumerate(relVarAggrQueries):
                fromClause += '\nLEFT JOIN\n({relVarAggrQuery}) relVarTable{relVarAggrIdxPlusOne}\n\t ' \
                              'ON relVarTable0.id = relVarTable{relVarAggrIdxPlusOne}.id'.format(
                    relVarAggrQuery=relVarAggrQuery,
                    relVarAggrIdxPlusOne=relVarAggrIdx + 1)

            query = '{selectClause} {fromClause}'.format(selectClause=selectClause, fromClause=fromClause)
        else:
            query = self._getSQLQueryForIds(baseItemName)

        cursor = self._connection.cursor()
        query = self._translateQuery(query, self._executeParamstyle)
        logger.debug(query + ';')
        cursor.execute(query, paramDict)
        for row in cursor.fetchall():
            yield row[0], [val for val in row[1:]]
        self._connection.rollback()     # NB: without this, psycopg2 hangs if you're trying to drop the table you're querying from
        cursor.close()


    def _getSQLQueryForRelVarAggr(self, relVarAggr, schema, paramDict):
        relPath = relVarAggr.relVar.path
        self._validateRelPath(schema, relPath, self.getTableNames())
        self._verifyRelVarAttrsExist(schema, relVarAggr.relVar)

        # count frequency of each item so that the numeric suffix on mapped names will be like:
        # [A, AB, B, AB, A] -> [A1, AB1, B1, AB2, A2]
        mappedRelPath = []
        itemCountsUsed = {}
        for itemName in relPath:
            itemCountsUsed.setdefault(itemName, 0)
            itemCountsUsed[itemName] += 1
            mappedRelPath.append(itemName + str(itemCountsUsed[itemName]))

        # record all mapped names for each original name
        originalToMappedNames = {}
        for itemName, mappedItemName in zip(relPath, mappedRelPath):
            originalToMappedNames.setdefault(itemName, []).append(mappedItemName)

        # build up each part of query
        variableName = 'id' if relVarAggr.relVar.isExistence() else relVarAggr.relVar.attrName
        aggrSql = self._getSqlForAggregator(relVarAggr, '{mappedRelPathMinus1}.{varName}'.format(
            mappedRelPathMinus1=mappedRelPath[-1],
            varName=variableName),
                                            paramDict)
        selectClause = 'SELECT {mappedRelPath0}.id AS id, {aggrSql} AS val'.format(
            mappedRelPath0=mappedRelPath[0],
            aggrSql=aggrSql)
        fromClause = '\nFROM '
        for itemName, mappedItemName in zip(relPath, mappedRelPath):
            fromClause += '{itemName} {mappedItemName}, '.format(
                itemName=itemName,
                mappedItemName=mappedItemName)
        fromClause = fromClause[:-2]
        whereEqualities = []
        for relPathIdx, (item1Name, item2Name) in enumerate(zip(relPath[:-1], relPath[1:])):
            if item2Name in [rel.name for rel in schema.getRelationships()]:    # item1 is an entity
                entName = item1Name
                mappedEntName, mappedRelName = mappedRelPath[relPathIdx], mappedRelPath[relPathIdx + 1]
                whereEqualities.append('{mappedEntName}.id = {mappedRelName}.{entName}_id'.format(
                    mappedEntName=mappedEntName,
                    mappedRelName=mappedRelName,
                    entName=entName))
            else:   # item1 is a relationship
                entName = item2Name
                mappedEntName, mappedRelName = mappedRelPath[relPathIdx + 1], mappedRelPath[relPathIdx]
                whereEqualities.append('{mappedEntName}.id = {mappedRelName}.{entName}_id'.format(
                    mappedEntName=mappedEntName,
                    mappedRelName=mappedRelName,
                    entName=entName))
        whereClause = '\nWHERE '
        if len(whereEqualities) > 0:
            whereClause += ' AND '.join(whereEqualities)
        whereInequalities = []
        for itemName, mappedItemNames in originalToMappedNames.items():
            for mappedItemName1, mappedItemName2 in zip(mappedItemNames[:-1], mappedItemNames[1:]):
                whereInequalities.append('{mappedItemName1}.id <> {mappedItemName2}.id'.format(
                    mappedItemName1=mappedItemName1,
                    mappedItemName2=mappedItemName2))
        if len(whereInequalities) > 0:
            whereClause += ' AND {whereInequalities}'.format(whereInequalities=' AND '.join(whereInequalities))
        if len(whereEqualities) > 0 or len(whereInequalities) > 0:
            whereClause += ' AND '
        whereClause += '{mappedRelPathMinus1}.{varName} IS NOT NULL'.format(
            mappedRelPathMinus1=mappedRelPath[-1],
            varName=variableName)

        groupByClause = ''
        if self._isGroupBy(relVarAggr):
            groupByClause += '\nGROUP BY {mappedRelPath0}.id'.format(mappedRelPath0=mappedRelPath[0])
        queryString = '{selectClause} {fromClause} {whereClause} {groupByClause}'.format(
            selectClause=selectClause,
            fromClause=fromClause,
            whereClause=whereClause,
            groupByClause=groupByClause)
        return queryString


    def _getSQLQueryForIds(self, tableName):
        return "SELECT id FROM {tableName}".format(tableName=tableName)
