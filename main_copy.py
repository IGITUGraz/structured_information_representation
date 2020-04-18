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
    parser = argparse.ArgumentParser(description='test COPY operation')
    parser.add_argument('-N', dest='runs', type=int, default=10, help='number of COPY operations to run (default: 10)')
    parser.add_argument('-c', dest='configname', type=str, default='final_config', help='config name')
    parser.add_argument('-C', dest='cspacename', type=str, required=True, help='content space name')

    args = parser.parse_args()

    num_runs = args.runs
    configname = args.configname
    cspacename = args.cspacename

    op = 'run'

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

        outdir = setup_outdir(join('out', 'copy_single') if num_runs == 1 else join('out', 'copy', configname, cspacename))
        logfile = join(outdir, 'log.txt')

        logger = Logger(logfile, mode='overwrite')
        sys.stdout = sys.stderr = logger

        # run

        costs_c = []
        results = []

        for n in range(num_runs):
            if num_runs > 1:
                print('run {0:d}/{1:d}'.format(n+1, num_runs))

            cost_c, run_results = store_copy_recall(
                outdir,
                cspacefile,
                variant,
                config_c=config_c,
                config_v=config_v,
                recall_cfg=recall_cfg,
                k_pattern=(n % 5),  # cyclicly test each pattern
                print_results=True,
                plot=(num_runs == 1),
                show=(num_runs == 1))

            costs_c += [cost_c]
            results += [run_results]

        def compact_results(key, val):
            if key != 'success':
                return dict(mean=np.mean(val), std=np.std(val))
            else:
                return dict(succeeded=sum(val), failed=len(val)-sum(val), all=len(val))

        results_all = {key: [r[key] for r in results] for key in results[0].keys()}
        results_all_ms = {key: compact_results(key, val) for key, val in results_all.items()}

        cost_c = dict(mean=np.mean(costs_c), std=np.std(costs_c))

        results = {
            'cspacename': cspacename,
            'configname': configname,
            'variant': variant,
            'recall_cfg': recall_cfg,
            'results': results_all,
            'results_compact': results_all_ms,
            'success': results_all_ms['success'],
            'readout_error': results_all_ms['readout_error'],
        }

        # print

        print('cost_c:', cost_c)
        print('success: {}/{}'.format(results_all_ms['success']['succeeded'], results_all_ms['success']['all']))

        dump_dict(config_c, dumpfile=join(outdir, 'config_c.json'), print_stdout=True, key='config_c', ordered=True)
        dump_dict(config_v, dumpfile=join(outdir, 'config_v.json'), print_stdout=True, key='config_v', ordered=True)
        dump_dict(results, dumpfile=join(outdir, 'results.json'), print_stdout=False, key='results', ordered=True)
        if num_runs > 1:
            dump_dict(results_all_ms, dumpfile=join(outdir, 'results_compact.json'), print_stdout=True, key='results (compact)')

    plt.show()
