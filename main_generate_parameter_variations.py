#!/usr/bin/env python3

import argparse
import numpy as np
import os.path
from os.path import join
import pickle as pkl

import config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate variations to a config')
    parser.add_argument('-s', dest='std', type=float, default=.1, help='standard deviation around base config values')
    parser.add_argument('-N', dest='variations', type=int, default=100, help='number of parameter variations to generate (default: 100)')
    parser.add_argument('-c', dest='configname', type=str, default='final_config', help='base config name')

    args = parser.parse_args()

    std = args.std
    num_vars = args.variations
    configname = args.configname

    op = 'run'

    # get content space and config data

    config_c, config_v, variant, recall_cfg = config.load(configname)

    if op == 'run':

        for n in range(num_vars):
            config_c0 = config_c.copy()
            config_v0 = config_v.copy()

            # change values

            for k, v in config_c0.items():
                if k[:6] == 'alpha_':
                    continue

                config_c0[k] = np.random.normal(v, std*v)

            for k, v in config_v0.items():
                if k[:6] == 'alpha_':
                    continue

                config_v0[k] = np.random.normal(v, std*v)

            # check variables so they make sense

            # time constants must be positive
            config_c0['tau_minus'] = max(config_c0['tau_minus'], 0.1)
            config_c0['tau_plus_SE'] = max(config_c0['tau_plus_SE'], 0.1)
            config_v0['tau_minus'] = max(config_v0['tau_minus'], 0.1)
            config_v0['tau_plus_SE'] = max(config_v0['tau_plus_SE'], 0.1)
            config_v0['tau_plus_EE'] = max(config_v0['tau_plus_EE'], 0.1)

            # excitatory weights must be positive
            config_c0['w_SE_low'] = max(config_c0['w_SE_low'], 0)
            config_c0['w_SE_high'] = max(config_c0['w_SE_high'], .01)
            config_v0['w_SE_low'] = max(config_v0['w_SE_low'], 0)
            config_v0['w_SE_high'] = max(config_v0['w_SE_high'], .01)
            config_v0['w_EE_low'] = max(config_v0['w_EE_low'], 0)
            config_v0['w_EE_high'] = max(config_v0['w_EE_high'], .01)

            # low weight must be smaller than high weight for weight init
            config_c0['w_SE_high'] = max(config_c0['w_SE_low']*1.01, config_c0['w_SE_high'])  # low < high
            config_c0['w_SE_max'] = max(config_c0['w_SE_high'], config_c0['w_SE_max'])
            config_v0['w_SE_high'] = max(config_v0['w_SE_low']*1.01, config_v0['w_SE_high'])  # low < high
            config_v0['w_SE_max'] = max(config_v0['w_SE_high'], config_v0['w_SE_max'])
            config_v0['w_EE_high'] = max(config_v0['w_EE_low']*1.01, config_v0['w_EE_high'])  # low < high
            config_v0['w_EE_max'] = max(config_v0['w_EE_high'], config_v0['w_EE_max'])

            # write config

            configname0 = configname+'_std{0:g}_{1:03d}'.format(std, n)

            config.write(configname0, config_c=config_c0, config_v=config_v0, variant=variant, recall_cfg=recall_cfg)


