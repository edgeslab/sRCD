import collections
from ..model.RelationalDependency import RelationalVariable
from ..model.Aggregator import AverageAggregator
from ..model.Aggregator import IdentityAggregator
from ..model import ParserUtil
from ..dseparation.DSeparation import DSeparation
from ..dseparation.SigmaSeparation import SigmaSeparation
import rpy2.robjects as robjects
r = robjects.r
import logging

logger = logging.getLogger(__name__)

class CITest(object):

    def isConditionallyIndependent(self, relVar1Str, relVar2Str, condRelVarStrs):
        raise NotImplementedError


class LinearCITest(CITest):

    def __init__(self, schema, dataStore, alpha=0.05, soeThreshold=0.01):
        self.schema = schema
        self.dataStore = dataStore
        self.alpha = alpha
        self.soeThreshold = soeThreshold


    def isConditionallyIndependent(self, relVar1Str, relVar2Str, condRelVarStrs):
        logger.debug("testing %s _||_ %s | { %s }", relVar1Str, relVar2Str, condRelVarStrs)
        if not isinstance(relVar1Str, str) and not isinstance(relVar1Str, RelationalVariable) or not relVar1Str:
            raise Exception("relVar1Str must be a parseable RelationalVariable string")
        if not isinstance(relVar2Str, str) and not isinstance(relVar2Str, RelationalVariable) or not relVar2Str:
            raise Exception("relVar2Str must be a parseable RelationalVariable string")
        if not isinstance(condRelVarStrs, collections.Iterable) or isinstance(condRelVarStrs, str):
            raise Exception("condRelVarStrs must be a sequence of parseable RelationalVariable strings")

        relVar1 = ParserUtil.parseRelVar(relVar1Str)
        relVar2 = ParserUtil.parseRelVar(relVar2Str)
        if len(relVar1.path) > 1 and len(relVar2.path) > 1:
            raise Exception("At least one of relVar1Str or relVar2Str must have a singleton path")

        if len(relVar2.path) > 1:   # Swap the vars if relvar1 has singleton path
            relVar1, relVar2 = ParserUtil.parseRelVar(relVar2Str), ParserUtil.parseRelVar(relVar1Str)

        baseItemName = relVar1.getBaseItemName()
        relVarAggrs = [AverageAggregator(relVar1Str), IdentityAggregator(relVar2Str)]
        relVarAggrs.extend([AverageAggregator(condRelVarStr) for condRelVarStr in condRelVarStrs])

        relVar1Data = []
        relVar2Data = []
        condVarsData = []
        for i in range(len(condRelVarStrs)):
            condVarsData.append([])

        for idVal, row in self.dataStore.getValuesForRelVarAggrs(self.schema, baseItemName, relVarAggrs):
            if None in row:
                continue
            relVar1Data.append(float(row[0]))
            relVar2Data.append(float(row[1]))
            for i, value in enumerate(row[2:]):
                condVarsData[i].append(float(value))

        robjects.globalenv['treatment'] = robjects.FloatVector(relVar1Data)
        robjects.globalenv['outcome'] = robjects.FloatVector(relVar2Data)
        for i, condVarData in enumerate(condVarsData):
            robjects.globalenv['cond{}'.format(i)] = robjects.FloatVector(condVarData)

        if not condVarsData: # marginal
            linearModel = r.lm('outcome ~ treatment')
            effectSize = r('cor(treatment, outcome)^2')[0]
            summary = r.summary(linearModel)
        else:
            condVarIndexes = range(len(condVarsData))
            linearModel = r.lm('outcome ~ treatment + cond{}'.format(' + cond'.join(map(str, condVarIndexes))))
            effectSize = r('cor(residuals(lm(outcome ~ cond{condVarStrs})), '
                           'residuals(lm(treatment ~ cond{condVarStrs})))^2'.format(
                            condVarStrs=(' + cond'.join(map(str, condVarIndexes)))))[0]
            summary = r.summary(linearModel)

        pval =  summary.rx2('coefficients').rx(2,4)[0]
        logger.debug('soe: {}, pval: {}'.format(effectSize, pval))
        return pval > self.alpha or effectSize < self.soeThreshold


class Oracle(CITest):

    def __init__(self, model, hopThreshold=0, sep='d'):
        self.model = model
        self.hopThreshold = hopThreshold
        self.sep_type = sep

        self.sep_methods = {
            'd'     :   DSeparation(model),
            'sigma' :   SigmaSeparation(model)
        }

    def isConditionallyIndependent(self, relVar1Str, relVar2Str, condRelVarStrs, sep_type=None):
        sep_type = self.sep_type if sep_type is None else sep_type
        sep = self.sep_methods[sep_type]
        res = sep.separated(self.hopThreshold, [relVar1Str], [relVar2Str], condRelVarStrs)
        # print('%s [%s _|_ %s | %s] = %s' % (sep_type, relVar1Str, relVar2Str, condRelVarStrs, res))
        return res