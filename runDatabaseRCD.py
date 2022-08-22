from causality.model.Schema import Schema
from causality.learning.RCD import RCD
from causality.citest.CITest import LinearCITest
from causality.datastore.PostgreSqlDataStore import PostgreSqlDataStore
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

schema = Schema()
schema.addEntity('A')
schema.addAttribute('A', 'X')
schema.addAttribute('A', 'W')
schema.addEntity('B')
schema.addAttribute('B', 'Y')
schema.addRelationship('AB', ('A', Schema.ONE), ('B', Schema.MANY))
schema.addEntity('C')
schema.addAttribute('C', 'Z')
schema.addAttribute('C', 'V')
schema.addRelationship('BC', ('B', Schema.ONE), ('C', Schema.MANY))
logger.info(schema)

# NB: Following step requires prior loading of rcd-test-data.sql into the test database.
dataStore = PostgreSqlDataStore(dbname='test', user='test', password='test', host='localhost', port='5432')
linearCITest = LinearCITest(schema, dataStore)

hopThreshold = 4
rcd = RCD(schema, linearCITest, hopThreshold)
rcd.identifyUndirectedDependencies()
rcd.orientDependencies()