import logging
import os
import unittest
from causality.test.tcunittest import TeamcityTestRunner
from causality.datastore.Sqlite3DataStore import Sqlite3DataStore
from causality.datastore.InMemoryDataStore import InMemoryDataStore

logger = logging.getLogger(__name__)


class DataStoreTestSuite(unittest.TestCase):
    # runs the tests in TestDataStoreBasicOperations twice, setting its dsSubclassInitTuple variable to either InMemoryDataStore
    # or SqlDataStore. note that the tests in TestDataStoreBasicOperations also run when directly called by IntelliJ,
    # but dsSubclassInitTuple is None in that case. This scheme is from:
    # http://stackoverflow.com/questions/3742791/how-do-i-repeat-python-unit-tests-on-different-data
    #
    # The 'TEST_SERVERS' environment variable controls which servers to run against. It is a comma-separated list of
    # supported servers, currently 'postgres' and 'mysql (no quotes). Include those that you want to run tests on.
    # Set it to nothing (or don't set it at all) to only run the in-memory and sqlite3 tests. For example, to run
    # against just postgress:
    #
    #   export TEST_SERVERS=postgres
    #
    # Or both postgres and mysql:
    #
    #   export TEST_SERVERS=mysql,postgres
    #
    # NB: running against postgres or mysql requires an active server - see README.

    SUPPORTED_SERVERS= ['mysql', 'postgres']

    def runTest(self):

        # while ugly, solves circular import problems:
        from test.datastore.testDataStoreBasicOperations import TestDataStoreBasicOperations
        from test.datastore.testDataStoreRelationalPath import TestDataStoreRelationalPath

        dsSubclassInitTuples = [        # tuple of DataStore class, connect kwargs dict
            (InMemoryDataStore, None),
            (Sqlite3DataStore, {'database':':memory:'}),
            ]
        testServers = os.environ.get('TEST_SERVERS')    # comma-separated list
        if testServers:
            testServers = testServers.split(',')
            for server in testServers:
                if server not in self.SUPPORTED_SERVERS:
                    raise Exception("unspported server {!r}. Must be one of {}".format(server, self.SUPPORTED_SERVERS))

            if 'postgres' in testServers:
                from causality.datastore.PostgreSqlDataStore import PostgreSqlDataStore
                logger.info('adding server tests for PostgreSqlDataStore')
                dsSubclassInitTuples.append((PostgreSqlDataStore, {'dbname':'test', 'user':'test', 'password':'test'}))
            if 'mysql' in testServers:
                from causality.datastore.MySqlDataStore import MySqlDataStore
                logger.info('adding server tests for MySqlDataStore')
                dsSubclassInitTuples.append((MySqlDataStore, {'database':'test', 'user':'test', 'password':'test'}))
        else:
            logger.info('skipping server tests for {}'.format(self.SUPPORTED_SERVERS))

        for dsSubclassInitTuple in dsSubclassInitTuples:
            for baseTestClass in [
                TestDataStoreBasicOperations,
                TestDataStoreRelationalPath,
                ]:

                testClassName = '{baseClass}_{dsSubclass}'.format(
                    baseClass=baseTestClass.__name__,
                    dsSubclass=dsSubclassInitTuple[0].__name__)
                testClass = type(testClassName, (baseTestClass, ), {'dsSubclassInitTuple': dsSubclassInitTuple})    # set the dsSubclassInitTuple property. will be None if called directly from unittest
                suite = unittest.TestLoader().loadTestsFromTestCase(testClass)
                logger.info('running constructed test: {}'.format(testClassName))
                TeamcityTestRunner().run(suite)


def skipOrInstantiateDsSubclass(testClass):
    """
    Utility to help above baseTestClass instances manage their injected classes. Drops all tables in the newly-created
    dataStoreInstance before returning it, for the case when databases persist across connections.
    """
    if not testClass.dsSubclassInitTuple:
        testClass.skipTest("skipped {}: no dsSubclassInitTuple".format(testClass))
    dataStoreClass = testClass.dsSubclassInitTuple[0]
    initArg = testClass.dsSubclassInitTuple[1]
    dataStoreInstance = dataStoreClass(**initArg) if initArg else dataStoreClass()

    logger.debug('dropping tables in new instance: {}'.format(dataStoreInstance))
    dataStoreInstance.dropAllTables()
    return dataStoreInstance
