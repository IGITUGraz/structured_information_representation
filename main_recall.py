#!/usr/bin/env python3

import argparse
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import os.path
from os.path import join
import pickle as pkl

import config
from utils import *
from variable_binding import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test RECALL operation')
    meg = parser.add_mutually_exclusive_group()
    meg.add_argument('-K', dest='contents', type=int, default=None, choices=range(1, 6), help='number of content assemblies to load and RECALL (default: all in the content space)')
    meg.add_argument('-N', dest='runs', type=int, default=10, help='number of STORE/RECALL operations per content assemblies (default: 10)')
    meg.add_argument('-S', dest='statistics', action='store_true', help='gather statistics instead of performing the usual runs')
    parser.add_argument('-c', dest='configname', type=str, default='final_config', help='config name')
    parser.add_argument('-C', dest='cspacename', type=str, required=True, help='content space name')

    args = parser.parse_args()

    num_contents = args.contents
    num_runs = args.runs
    configname = args.configname
    cspacename = args.cspacename

    op = 'statistics' if args.statistics else 'run'

    # get content space and config data

    datadir = 'data'
    cspacefile = join(datadir, cspacename, 'trained_swta.pkl')

    if not os.path.isfile(cspacefile):
        raise IOError('cspace file not found: ' + cspacefile)

    print('using cspace file {0:s}'.format(cspacefile))
    print('using config {0:s}'.format(configname))

    setup_numpy_and_matplotlib()

    # load config

    config_c, config_v, variant, recall_cfg = config.load(configname)

    if op == 'run':

        assert num_runs >= 1

        # setup logging

        outdir = setup_outdir(join('out', 'recall_single') if num_runs == 1 else join('out', 'recall', configname, cspacename))
        logfile = join(outdir, 'log.txt')

        logger = Logger(logfile, mode='overwrite')
        sys.stdout = sys.stderr = logger

        # run

        costs_c = []
        costs_v = []
        results_k = []
        readout_errors = []

        for n in range(num_runs):
            if num_runs > 1:
                print('run {0:d}/{1:d}'.format(n+1, num_runs))

            cost_c, cost_v, results = store_recall(
                outdir,
                cspacefile,
                variant,
                config_c=config_c,
                config_v=config_v,
                recall_cfg=recall_cfg,
                k_pattern=list(range(num_contents)) if num_contents is not None else None,
                print_results=True,
                print_assemblies=True,
                plot=(num_runs == 1),
                show=(num_runs == 1))

            costs_c += [cost_c]
            costs_v += [cost_v]
            readout_errors += [results['readout_error']]

            for r_k, r in enumerate(results['results_k']):
                if len(results_k) < r_k + 1:
                    results_k += [{}]

                for key, val in r.items():
                    if key not in results_k[r_k].keys():
                        results_k[r_k][key] = []

                    results_k[r_k][key] += [val]

        results_all = {key: [x for r in results_k for x in r[key]] for key in results_k[0].keys()}

        def compact_results(key, val):
            if key != 'success':
                return dict(mean=np.mean(val), std=np.std(val))
            else:
                return dict(succeeded=sum(val), failed=len(val)-sum(val), all=len(val))

        results_k_ms = [{key: compact_results(key, val) for key, val in r.items()} for r in results_k]
        results_all_ms = {key: compact_results(key, val) for key, val in results_all.items()}

        cost_c = dict(mean=np.mean(costs_c), std=np.std(costs_c))
        cost_v = dict(mean=np.mean(costs_v), std=np.std(costs_v))

        readout_error = dict(mean=np.mean(readout_errors), std=np.std(readout_errors))
        results_all_ms['readout_error'] = readout_error

        # print

        results = {
            'N': num_runs,
            'cspacename': cspacename,
            'configname': configname,
            'variant': variant,
            'results_k': results_k,
            'results_k_compact': results_k_ms,
            'results': results_all,
            'results_compact': results_all_ms,
            'readout_errors': readout_errors,
            'readout_error': readout_error,
            'cost_C': cost_c,
            'cost_V': cost_v,
            'success': results_all['success'],
        }

        print('cost_c:', cost_c)
        print('cost_v:', cost_v)
        print('success: {}/{}'.format(results_all_ms['success']['succeeded'], results_all_ms['success']['all']))

        dump_dict(config_c, dumpfile=join(outdir, 'config_c.json'), print_stdout=True, key='config_c')
        dump_dict(config_v, dumpfile=join(outdir, 'config_v.json'), print_stdout=True, key='config_v')
        dump_dict(results, dumpfile=join(outdir, 'results.json'), print_stdout=False, key='results')
        dump_dict(results_all_ms, dumpfile=join(outdir, 'results_compact.json'), print_stdout=True, key='results (compact)')

    if op == 'statistics':
        outdir = setup_outdir(join('out', 'recall_statistics', configname, cspacename))

        store_recall(
            outdir,
            cspacefile,
            variant,
            config_c=config_c,
            config_v=config_v,
            recall_cfg=recall_cfg,
            gather_statistics=True,
            plot=True,
            show=True)

    plt.show()
