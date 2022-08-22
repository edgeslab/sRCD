import logging
from ..datastore.SqlDataStore import SqlDataStore


logger = logging.getLogger(__name__)


class PostgreSqlDataStore(SqlDataStore):
    """
    A PostgreSQL-based DataStore. Requires psycopg2 module.
    """

    def __init__(self, **kwargs):
        """
        kwargs: Keywards to be passed to psycopg2.connect() as documented in
            http://initd.org/psycopg/docs/module.html#psycopg2.connect and
            http://www.postgresql.org/docs/current/static/libpq-connect.html#LIBPQ-CONNSTRING
        example:
            dbname=test user=test password=test host=localhost port=5432
        """
        import psycopg2     # import here as-needed so that tests can run without the module installed
        super().__init__(psycopg2.connect(**kwargs), 'pyformat', 'format')
        self.defineIsIntegerFunction()


    def defineIsIntegerFunction(self):
        # define the is_integer() that _getSqlForNonIntegerDelete() uses
        query = '''
            CREATE OR REPLACE FUNCTION is_integer(TEXT) RETURNS BOOLEAN AS $$
            BEGIN
                PERFORM cast($1 AS INTEGER);
                RETURN 1;
            EXCEPTION
                WHEN INVALID_TEXT_REPRESENTATION THEN
                    RETURN 0;
            END;
            $$ LANGUAGE PLPGSQL IMMUTABLE;
        '''
        self.execute(query)


    def _getTableNamesQuery(self):
        return "SELECT table_name FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog', 'information_schema')"


    def _getIntegrityErrorClass(self):
        import psycopg2     # import here as-needed so that tests can run without the module installed
        return psycopg2.IntegrityError


    def _isPrimaryKeyUniquenessViolation(self, integrityError):
        return integrityError.args[0].startswith('duplicate key value violates unique constraint')


    def _cascadeRequired(self):
        return True


    def _getSqlForNonRandomChoiceAggregator(self, columnName):
        return '(array_agg({columnName}))[1]'.format(columnName=columnName) # non-random choice, always choosing first item


    def _getSqlForNonIntegerDelete(self, columnName):
        return "NOT is_integer({column})".format(column=columnName)     # NB: requires user-defined function is_integer()


    def _getSqlForRandomDeleteWhere(self, probability, paramDict):
        paramDict['probVal'] = probability
        return 'NOT RANDOM() < %(probVal)s'


    def loadCsvFile(self, tableName, csvFile, delimiter):
        """
        Overrides default row-by-row loader to use bulk import.
        """
        cursor = self._connection.cursor()
        try:
            cursor.copy_from(csvFile, tableName, sep=delimiter)
            self._connection.commit()
        except Exception as exception:
            self._connection.rollback()     # otherwise MySQL may deadlock (seen during unit tests, e.g., "Waiting for table metadata lock")
            raise exception                 # re-raise so others can handle
        finally:
            cursor.close()
