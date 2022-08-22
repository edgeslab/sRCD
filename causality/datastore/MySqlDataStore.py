import logging
import re
from ..datastore.SqlDataStore import SqlDataStore


logger = logging.getLogger(__name__)


class MySqlDataStore(SqlDataStore):
    """
    A MySQL-based DataStore. Requires MySQL Connector/Python module.
    """

    def __init__(self, **kwargs):
        """
        kwargs: Keywards to be passed to mysql.connector.connect() as documented in
            http://dev.mysql.com/doc/refman/5.6/en/myconnpy_example_connecting.html and
            http://dev.mysql.com/doc/refman/5.6/en/connector-python-connectargs.html
        example:
            database=test user=test password=test host=localhost port=3306
        """
        import mysql.connector  # import here as-needed so that tests can run without the module installed
        super().__init__(mysql.connector.connect(**kwargs), 'pyformat', 'format')
        self.defineIsIntegerFunction()


    def defineIsIntegerFunction(self):
        # define the is_integer() that _getSqlForNonIntegerDelete() uses
        self.execute('DROP FUNCTION IF EXISTS is_integer')
        query = '''
            CREATE FUNCTION is_integer (val TEXT) RETURNS INTEGER
            BEGIN
                DECLARE val_cast_as_int INT SIGNED;
                DECLARE CONTINUE HANDLER FOR SQLSTATE '22007' RETURN 0;
                SET val_cast_as_int = cast(val as SIGNED);
                RETURN 1;
            END
        '''
        self.execute(query)


    def dropAllTables(self):
        try:
            self.execute('SET FOREIGN_KEY_CHECKS=0')
            super().dropAllTables()
        finally:
            self.execute('SET FOREIGN_KEY_CHECKS=1')


    def _getTableNamesQuery(self):
        return "SHOW TABLES"


    def _getIntegrityErrorClass(self):
        import mysql.connector  # import here as-needed so that tests can run without the module installed
        return mysql.connector.IntegrityError


    def _isPrimaryKeyUniquenessViolation(self, integrityError):
        return re.findall(r'Duplicate entry.*for key', integrityError.msg)


    def _cascadeRequired(self):
        return False    #  RESTRICT and CASCADE are permitted to make porting easier. In MySQL 5.6, they do nothing. - http://dev.mysql.com/doc/refman/5.6/en/drop-table.html


    def _getSqlForNonIntegerDelete(self, columnName):
        return "NOT is_integer({column})".format(column=columnName)     # NB: requires user-defined function is_integer()


    def _getSqlForRandomDeleteWhere(self, probability, paramDict):
        paramDict['probVal'] = probability
        return 'NOT RAND() < %(probVal)s'


    def _getIndexPrefixLengthSpec(self):
        # avoid errors like: mysql.connector.errors.ProgrammingError: 1170 (42000): BLOB/TEXT column 'column_0' used in key specification without a key length
        # REF: http://stackoverflow.com/questions/13710170/blob-text-column-bestilling-used-in-key-specification-without-a-key-length
        return '(256)'


    def getCastDataTypeForSqlType(self, sqlDataType):
        # to have to work around a MySQL oddity: http://dev.mysql.com/doc/refman/5.6/en/cast-functions.html#function_cast
        sqlTypeToCastType = {'INTEGER': 'SIGNED', 'TEXT': 'CHAR', 'REAL': 'BINARY'}
        return sqlTypeToCastType[sqlDataType]


    def loadCsvFile(self, tableName, columnNames, csvFileAbsName, csvFile, delimiter, fieldCount):
        """
        Overrides default row-by-row loader to use bulk import. NB: Requires the user to have FILE permission on the
        database, e.g.,
            GRANT FILE ON *.* TO 'test'@'localhost';
        """
        cursor = self._connection.cursor()
        try:
            cursor.execute('''LOAD DATA INFILE '{csvFile}' INTO TABLE {tableName}
                              FIELDS TERMINATED BY '{delimiter}';'''.format(
                csvFile=csvFileAbsName, tableName=tableName, delimiter=delimiter))
            self._connection.commit()
        except Exception as exception:
            self._connection.rollback()     # otherwise MySQL may deadlock (seen during unit tests, e.g., "Waiting for table metadata lock")
            raise exception                 # re-raise so others can handle
        finally:
            cursor.close()
