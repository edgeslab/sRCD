import pdb
import random
import logging

import numpy as np

from causality.citest.CITest import Oracle
from causality.learning import ModelEvaluation
from causality.learning.RCD import RCD
from causality.learning.sRCD import sRCD
from causality.learning.RelFCI import RelFCI
from causality.model.Schema import Schema
from causality.model.Model import Model
from causality.modelspace import ModelGenerator
from causality.modelspace import SchemaGenerator
from causality.graph.GraphUtil import isPossibleParent, isPossibleAncestor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

rseed = 123

random.seed(rseed)
np.random.seed(rseed)

# Parameters
numEntities = 3
numRelationships = 2
numDependencies = 4
numFeedbackLoops = 1
hopThreshold = 4
maxNumParents = rcdDepth = 3

# Parameters
# schema = SchemaGenerator.generateSchema(numEntities, numRelationships, allowCycles=True, oneRelationshipPerPair=True)
# logger.info(schema)
# model = ModelGenerator.generateModel(schema, hopThreshold, numDependencies, numFeedbackLoops, maxNumParents=maxNumParents)

# Self Loop example
schema = Schema()
schema.addEntity('A')
schema.addAttribute('A', 'X1')
schema.addEntity('B')
schema.addAttribute('B', 'Y1')
schema.addRelationship('AB', ('A', Schema.MANY), ('B', Schema.MANY))
model = Model(schema, ['[A, AB, B, AB, A].X1 -> [A].X1', '[A, AB, B].Y1 -> [A].X1'])

logger.info('Model: %s', model.dependencies)
oracle = Oracle(model, 2*hopThreshold, sep='sigma')

# Run RCD algorithm and collect statistics on learned model
rcd = RCD(schema, oracle, hopThreshold, depth=rcdDepth)
rcd.identifyUndirectedDependencies()
rcd.orientDependencies()


trueAggs = RCD.constructTrueAggsFromDependencies(schema, model.dependencies, hopThreshold)
# p, r, f1 = ModelEvaluation.parentalQuery(trueAggs, rcd.perspectiveToAgg, isPossibleParent)
# print("isPossibleParent: P:%.2lf, R:%.2lf, F1:%.2lf" % (p, r, f1))
p, r, f1 = ModelEvaluation.parentalQuery(trueAggs, rcd.perspectiveToAgg, isPossibleAncestor)
print("isPossibleAncestor: P:%.2lf, R:%.2lf, F1:%.2lf" % (p, r, f1))

print()
print(trueAggs['A'].edges())
print()
print(rcd.perspectiveToAgg['A'].edges())