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
    parser = argparse.ArgumentParser(description='test Frankland and Greene experiments')
    parser.add_argument('-c', dest='configname', type=str, default='final_config', help='config name')
    parser.add_argument('-C', dest='cspacename', type=str, required=True, help='content space name')
    parser.add_argument('-e', dest='experiment', type=int, default=1, choices=[1, 2], help='number of experiment')

    args = parser.parse_args()
    configname = args.configname
    cspacename = args.cspacename
    experiment = args.experiment
    use_data = 'mean_sum'

    op = 'run'
    datadir = 'data'
    cspacefile = join(datadir, cspacename, 'trained_swta.pkl')

    if not os.path.isfile(cspacefile):
        raise IOError('cspace file not found: ' + cspacefile)

    print('using cspace file {0:s}'.format(cspacefile))
    print('using config {0:s}'.format(configname))

    setup_numpy_and_matplotlib()

    # load config

    config_c, config_v, variant, recall_cfg = config.load(configname)

    print('using data:', use_data)

    outname = 'frankland_greene'

    if op == 'run':

        outdir = setup_outdir(join('out', outname, 'experiment_'+str(experiment), configname, cspacename))
        logfile = join(outdir, 'log.txt')

        logger = Logger(logfile, mode='overwrite')
        sys.stdout = sys.stderr = logger

        results = frankland_greene(
                outdir,
                cspacefile,
                variant,
                config_c=config_c,
                config_v=config_v,
                experiment=experiment,
                recall_cfg=recall_cfg,
                use_data=use_data,
                plot=True,
                show=False)

        results['configname'] = configname
        results['cspacename'] = cspacename

        dump_dict(results, dumpfile=join(outdir, 'results.json'), print_stdout=False, key='results', ordered=True)
