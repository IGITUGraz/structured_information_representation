#!/usr/bin/env python3

import argparse
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import os.path
from os.path import join
import pickle as pkl
import shelve

import config
from utils import *
from variable_binding import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test COMPARE operation')
    parser.add_argument('-k', dest='num_assemblies', type=int, default=5, help='number of assemblies to compare (default: 5)')
    parser.add_argument('-c', dest='configname', type=str, default='final_config', help='config name')
    parser.add_argument('-C', dest='cspacename', type=str, required=True, help='content space name')
    parser.add_argument('-n', dest='n_readout', type=int, default=50, help='number of readout neurons')

    args = parser.parse_args()

    num_assemblies = args.num_assemblies
    configname = args.configname
    cspacename = args.cspacename
    n_readout = args.n_readout

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

        assert num_assemblies >= 1

        k_pattern_1_ = [*range(num_assemblies)]
        k_pattern_2_ = [*range(num_assemblies)]

        # setup logging

        outdir = setup_outdir(join('out', 'compare_single') if num_assemblies == 1 else join('out', 'compare', configname, cspacename))
        logfile = join(outdir, 'log.txt')

        logger = Logger(logfile, mode='overwrite')
        sys.stdout = sys.stderr = logger

        results_ = []

        for k_pattern_1 in k_pattern_1_:
            for k_pattern_2 in k_pattern_2_:

                results = compare(
                        outdir,
                        cspacefile,
                        variant,
                        config_c=config_c,
                        config_v=config_v,
                        k_pattern_1=k_pattern_1,
                        k_pattern_2=k_pattern_2,
                        recall_cfg=recall_cfg,
                        n_readout=n_readout,
                        plot=False,
                        show=False)

                results_ += [results]

        dbfile = 'data_{0:s}_{1:s}_n{2:d}.shelf'.format(configname, cspacename, n_readout)

        with shelve.open(join(outdir, dbfile), 'c') as shelf:
            shelf['configname'] = configname
            shelf['cspacename'] = cspacename
            shelf['results_'] = results_

        # plot

        ro_t_same = [r['t'] for r in results_ if r['same']]
        ro_v_same = [r['v'] for r in results_ if r['same']]
        ro_t_diff = [r['t'] for r in results_ if not r['same']]
        ro_v_diff = [r['v'] for r in results_ if not r['same']]

        plt.figure()

        first_same = True
        first_diff = True

        for t, v in zip(ro_t_diff, ro_v_diff):
            l = 'different pattern' if first_diff else ''
            first_diff = False

            plt.plot(t, v, c='C1', label=l)

        for t, v in zip(ro_t_same, ro_v_same):
            l = 'same pattern' if first_same else ''
            first_same = False

            plt.plot(t, v, c='C0', label=l)

        plt.xlabel('$t$ / ms')
        plt.ylabel('$V_m$ / mV' if n_readout == 1 else 'filtered activity')
        plt.legend(loc='best')
        plt.tight_layout()

        savefile = 'readout_{0:s}_{1:s}_n{2:d}.pdf'.format(configname, cspacename, n_readout)
        plt.savefig(join(outdir, savefile))

    plt.show()


