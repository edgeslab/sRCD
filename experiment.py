import pdb
import copy
import json
import random
import logging
import hashlib
import argparse

from datetime import datetime

import numpy as np
import pandas as pd
import diskcache as dc

import multiprocessing as mp
from joblib import Parallel, delayed

from causality.citest.CITest import Oracle
from causality.graph.GraphUtil import isPossibleParent, isPossibleAncestor
from causality.learning import ModelEvaluation
from causality.learning.RCD import RCD
from causality.learning.sRCD import sRCD
from causality.learning.RelFCI import RelFCI
from causality.modelspace import ModelGenerator
from causality.modelspace import SchemaGenerator


cache = dc.Cache('cache/model/aaai23')

algo_map = {
    'RCD'   :   RCD.runRCD,
    'sRCD'  :   sRCD.runsRCD,
    'RelFCI':   RelFCI.runRelFCI
}

EVAL_ITEMS      = ['a', 'f']
EVAL_METRICS    = [[i+'_precision', i+'_recall'] for i in EVAL_ITEMS]
EVAL_METRICS    = [item for sublist in EVAL_METRICS for item in sublist]
EVAL_METRICS    = EVAL_METRICS + ['rule_CD', 'rule_RBO', 'rule_KNC', 'rule_CA', 'rule_MR3']


def build_model(params):
    random.seed(params['rseed'])
    np.random.seed(params['rseed'])
    print('trying seed {} ...'.format(params['rseed']))

    numEntities = params['num_entities']
    numDependencies = params['num_dependencies']
    numFeedbackLoops = params['num_feedback_loops']
    hopThreshold = params['hop_threshold']
    maxNumParents = rcdDepth = params['max_depth']

    schema = SchemaGenerator.generateSchema(numEntities, numEntities-1, allowCycles=True, oneRelationshipPerPair=True)
    model = ModelGenerator.generateModel(schema, hopThreshold, numDependencies, numFeedbackLoops, maxNumParents=maxNumParents)
    return schema, model


def get_model(params, config_hash):
    schema, model = None, None
    if config_hash not in cache:
        schema, model = build_model(params)
    else:
        if cache[config_hash] is None:
            raise Exception
        schema, model = cache[config_hash]
    return schema, model
    

# def run_exp(seed, logger, num_trials):
def run_exp(i, config, logger):
    config['params'][config['target']] = config['params'][config['target']][i]
    results = {}
    rule_frequency = {}

    for name in config["algos"]:
        for m in EVAL_METRICS:
            results['%s_%s' % (name, m)] = []
        rule_frequency[name] = {}

    count = 0
    trial = 0
    wcnt = 0
    wins = 0
    empty = 0
    while count < config['num_trials']:
    
        rseed = config['seed'] + trial
        config['params']['rseed'] = rseed
        config_hash = hashlib.sha256(json.dumps(config['params']).encode('utf8')).hexdigest()

        try:
            schema, model = get_model(config['params'], config_hash)
            cache.add(config_hash, (schema, model))
        except:
            trial += 1
            cache.add(config_hash, None)
            continue

        logger.info('Seed: %d', rseed)
        logger.info('Trial: %d, Count: %d', trial, count)
        logger.info(schema)
        logger.info('Model: %s', model.dependencies)

        print('rseed: {}'.format(rseed))

        a1, a2 = 0, 0
        trueAggs = RCD.constructTrueAggsFromDependencies(schema, model.dependencies, config['params']['hop_threshold'])
        for name in config["algos"]:
            random.seed(rseed)
            np.random.seed(rseed)

            sep_type, nameOnly = name.split('-')
            oracle = Oracle(model, 2*config['params']['hop_threshold'], sep=sep_type)

            algoObj = algo_map[nameOnly](schema, oracle, config['params']['hop_threshold'], depth=config['params']['max_depth'])
            # results['%s_s_precision' % name].append(ModelEvaluation.skeletonPrecision(model, algoObj.undirectedDependencies))
            # results['%s_s_recall' % name].append(ModelEvaluation.skeletonRecall(model, algoObj.undirectedDependencies))

            results['%s_f_precision' % name].append(ModelEvaluation.feedbackPrecision(trueAggs, algoObj.perspectiveToAgg))
            results['%s_f_recall' % name].append(ModelEvaluation.feedbackRecall(trueAggs, algoObj.perspectiveToAgg))

            # p, r, f1 = ModelEvaluation.parentalQuery(trueAggs, algoObj.perspectiveToAgg, isPossibleParent)
            # results['%s_p_precision' % name].append(p)
            # results['%s_p_recall' % name].append(r)

            p, r, f1 = ModelEvaluation.parentalQuery(trueAggs, algoObj.perspectiveToAgg, isPossibleAncestor)
            results['%s_a_precision' % name].append(p)
            results['%s_a_recall' % name].append(r)

            # a1 = r
            # if a1 > 0 and a1 < 1.0:
            #     trueAgg = trueAggs['A']
            #     outAgg = algoObj.perspectiveToAgg['A']
            #     pdb.set_trace()

            for rule, freq in algoObj.edgeOrientationRuleFrequency.items():
                results['%s_rule_%s' % (name, rule)].append(freq)
        
        trial += 1
        count += 1

    return pd.DataFrame(results)


def dump_results(config, res_all, target_vals, exp_name, out_dir):
    res_data = {config['target'] : []}
    for algo in config['algos']:
        for m in EVAL_METRICS:
            res_data['%s_%s' % (algo, m)] = []


    for i in range(len(target_vals)):
        res_data[config['target']].append(target_vals[i])
        for algo in config['algos']:
            for m in EVAL_METRICS:
                res_data['%s_%s' % (algo, m)].append(res_all[i]['%s_%s' % (algo, m)].mean())

    df = pd.DataFrame(res_data)

    for m in EVAL_METRICS:
        if 'rule' in m:
            continue
        cols_m = [config['target']] + [col for col in res_data.keys() if m in col]
        print(df[cols_m])
        df[cols_m].to_csv('%s/%s_%s.csv' % (out_dir, exp_name, m), index=None)
    
    cols_r = list(filter(lambda x: 'rule' in x, res_data.keys()))
    for algo in config['algos']:
        cols_r_a = list(filter(lambda x: algo in x, cols_r))
        df[cols_r_a] = df[cols_r_a].div(df[cols_r_a].sum(axis=1), axis=0)
    df.fillna(0, inplace=True)
    print(df[[config['target']] + cols_r])
    df[[config['target']] + cols_r].to_csv('%s/%s_rules.csv' % (out_dir, exp_name), index=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, default='configs/sample.json', help="config", required=True)
    parser.add_argument("-s", type=int, default=123, help="seed", required=False)
    parser.add_argument("-o", type=str, default='out', help="output dir", required=False)
    parser.add_argument("-d", type=int, default=0, help="debug mode (0/1)", required=False)
    parser.add_argument("--nop", action='store_true', help="don't run parallel?", required=False)
    
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG if args.d else logging.INFO)

    config = json.loads(open(args.config, 'r').read())
    target_vals = config['params'][config['target']]

    if args.nop:
        results = [run_exp(i, copy.deepcopy(config), logger) for i in range(len(target_vals))]
    else:
        num_jobs = min(mp.cpu_count(), len(target_vals))
        results = Parallel(n_jobs=num_jobs)(delayed(run_exp)(i, copy.deepcopy(config), logger) for i in range(len(target_vals)))

    exp_name = args.config.split('.')[0].split('/')[-1]
    dump_results(config, results, target_vals, exp_name, args.o)



if __name__ == '__main__':
    main()