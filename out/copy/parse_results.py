#!/usr/bin/env python3

import argparse
import glob
import json
import os
import os.path
from os.path import join
import numpy as np
import sys


def print_stats(a, title=None, fmt='{:.2g}', median=False, arange=True, percentiles=False, binary=False):
    title = '' if title is None else title
    a = np.asarray(a)

    if a.size == 0:
        print('{:s}: empty'.format(title))
        return

    stats = []

    if not binary:
        s = '{:s}: ' + fmt + ' +/- ' + fmt
        stats += [a.mean(), a.std()]

        in_paren = False

        if median:
            s += ' (median: ' + fmt
            in_paren = True
            stats += [np.median(a)]

        if arange:
            if not in_paren:
                s += ' ('
                in_paren = True
            else:
                s += ', '
            s += 'range: ' + fmt + ' -- ' + fmt
            stats += [a.min(), a.max()]

        if percentiles:
            if not in_paren:
                s += ' ('
                in_paren = True
            else:
                s += ', '
            s += '5--95 percentile: ' + fmt + ' -- ' + fmt
            stats += [*np.percentile(a, [5, 95])]

        if in_paren:
            s += ')'
    else:
        s = '{:s}: {:d}/{:d}'
        if fmt != '':
            s += ' (mean: ' + fmt + ')'
        stats = [a.sum(), len(a.reshape(-1)), a.mean()]

    s = s.format(title, *stats)
    print(s)

    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(dest='output_directory', type=str, nargs='+')
    parser.add_argument('-t', '--num-test', dest='num_test', type=int, default=0, help='number of test directories (default: 0)')
    parser.add_argument('-i', '--individual', dest='individual', action='store_true', default=False, help='print individual results instead of aggregated stats (this implies num_test 0)')
    args = parser.parse_args()

    dn = args.output_directory


    configs = []
    cspaces = []
    shared = []
    missing = []
    excess = []
    success = []
    readout_errors = []

    for cspace in dn:
        if cspace[-1] == '/':
            cspace = cspace[:-1]

        assert os.path.isdir(cspace), "cant find run directory: "+cspace

        cfn = join(cspace, 'results.json')

        if not os.path.exists(cfn):
            print('cant find results file, skipping: '+cfn)
            continue

        with open(cfn, 'rb') as f:
            d_ = json.load(f)[1]

        configs += [d_['configname']]
        cspaces += [d_['cspacename']]
        shared += [d_['results']['match_C_TR_shared']]
        missing += [d_['results']['match_C_TR_missing']]
        excess += [d_['results']['match_C_TR_excess']]
        success += [d_['results']['success']]
        readout_errors += [d_['results']['readout_error']]

    if len(readout_errors) < 1:
        print('no data, aborting')
        sys.exit(1)

    individual = args.individual
    num_test = args.num_test

    if individual:
        for i, (config, cspace) in enumerate(zip(configs, cspaces)):
            num_op = sum([len(s) for s in shared[i]])

            print('stats for config {0:s}, content space {1:s} ({2:d} ops):'.format(config, cspace, num_ops))
            print_stats(shared[i], '  shared', '{:.1f}')
            print_stats(missing[i], '  missing', '{:.1f}')
            print_stats(excess[i], '  excess', '{:.1f}')
            print_stats(success[i], '  similarity criterion', binary=True)
            print_stats(readout_errors[i], '  readout_error', '{:.3f}')

    else:
        def output(title='overall', skip=None, lim=None):
            u_configs = sorted(set(configs[skip:lim]))
            u_cspaces = sorted(set(cspaces[skip:lim]))

            num_config = len(u_configs)
            num_cspace = len(u_cspaces)
            num_tests = len(configs[skip:lim])

            num_ops = sum([len(s) for s in shared[skip:lim]])

            #print('stats for {0:d} configs, {1:d} content spaces ({2:d} ops total):'.format(num_config, num_cspace, num_ops))

            print('stats {0:s}:'.format(title))
            print('  {0:d} configs {1:s}:'.format(num_config, title), ', '.join(u_configs))
            print('  {0:d} content spaces {1:s}:'.format(num_cspace, title), ', '.join(u_cspaces))
            print('  {0:d} tests {1:s}'.format(num_tests, title))
            print('  {0:d} ops {1:s}'.format(num_ops, title))
            print_stats(shared[skip:lim], '  shared', '{:.1f}')
            print_stats(missing[skip:lim], '  missing', '{:.1f}')
            print_stats(excess[skip:lim], '  excess', '{:.1f}')
            print_stats(success[skip:lim], '  similarity criterion', binary=True)
            print_stats(readout_errors[skip:lim], '  readout_error', '{:.3f}')

        output()

        if num_test > 0:
            output('training', lim=-num_test)
            output('test', skip=-num_test)
