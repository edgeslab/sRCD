import logging
import sqlite3
from ..datastore.SqlDataStore import SqlDataStore

logger = logging.getLogger(__name__)


class Sqlite3DataStore(SqlDataStore):
    """
    An sqlite3-based concrete DataStore. Requires sqlite3 module, and sqlite3 library 3.7 or better (earlier versions
    on Mac OS X timed out and hung).
    """

    def __init__(self, database=':memory:'):
        super().__init__(sqlite3.connect(database), 'named', 'qmark')


    def _getTableNamesQuery(self):
        return "SELECT tbl_name FROM sqlite_master WHERE type = 'table'"    # NB: does not include TEMP tables


    def _getIndexNamesForTableQuery(self, tableName):
        return "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = '{tableName}'".format(
            tableName=tableName)


    def _getIntegrityErrorClass(self):
        return sqlite3.IntegrityError


    def _isPrimaryKeyUniquenessViolation(self, integrityError):
        return integrityError.args[0] == "PRIMARY KEY must be unique"


    def _cascadeRequired(self):
        return False


    def _getSqlForNonIntegerDelete(self, columnName):
        return "cast({column} AS INTEGER) <> {column}".format(column=columnName)


    def _getSqlForRandomDeleteWhere(self, probability, paramDict):
        # sqlite implementation inspired by http://stackoverflow.com/questions/6037675/how-to-randomly-delete-20-of-the-rows-in-a-sqlite-table
        sqlite3BoundAbsVal = +9223372036854775808   # from http://www.sqlite.org/lang_corefunc.html
        probVal = -sqlite3BoundAbsVal + (probability * 2 * sqlite3BoundAbsVal)      # NB: this is wrong for the 1.0 case upper end, because it should be <=, not <. however, with so many values it's probably OK to ignore
        paramDict['probVal'] = probVal
        return 'NOT RANDOM() < %(probVal)s'

