#!/usr/bin/env python3

import datetime
import json
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import os.path
from os.path import join
import platform
import sys
from types import SimpleNamespace


from config import dump_dict


# import nest without verbose text
if 'nest' not in sys.modules:
    sys.argv.append('--quiet')


exec_name = sys.argv[0]
import nest
sys.argv[0] = exec_name

nest.set_verbosity(18)  # M_DEPRECATED


class Logger:
    def __init__(self, filename, mode='write', write_start_info=True):
        """
        mode: one of 'write', 'overwrite', 'append'
        """
        self.terminal = sys.stdout

        if mode == 'write':
            if os.path.isfile(filename):
                raise IOError('mode is "write", but file exists: {}'.format(filename))

            m = 'w'
        elif mode == 'overwrite':
            m = 'w'
        elif mode == 'append':
            m = 'a'

        self.log = open(filename, m)

        if write_start_info:
            hostname = platform.node()
            self.log.write('running on host {0:s}\n'.format(hostname))

            now = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
            self.log.write('process started on {0:s}\n'.format(now))

            self.log.write(''*70)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def setup_outdir(name):
    outdir = join(os.path.dirname(__file__), name)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    return outdir


def setup_numpy_and_matplotlib():
    np.set_printoptions(precision=3, suppress=True)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=18)


def setup_nest(seeds=None, modname=None, dt=.1, print_time=False, thread_offs=0, thread_div=1, verbose=True):
    """
    Setup nest: define number of cpus to use, set random seeds, load modules.

    args:
        print_time      enable nest time printing (default: False)
        thread_offs     number of cores not to use, default: 0
        thread_div      number of partitions for rest of cores, i.e. number of
                        threads is #CPUs / thread_div, if 0, do not use
                        multithreading, default: 1
        modname         if set, load this module, can also be a list
        verbose         print info on settings and actions (default: True)
    """

    if not verbose:
        nest.set_verbosity('M_DEPRECATED')

    nest.ResetKernel()

    # nest kernel settings
    if thread_div > 0:
        num_cpus = mp.cpu_count() - thread_offs
        num_thread = int(num_cpus / thread_div)
    else:
        num_thread = 1
    cfg = {'print_time': print_time, 'local_num_threads': num_thread, 'resolution': dt}
    nest.SetKernelStatus(cfg)

    # set random seeds
    if seeds is None:
        num_seeds = len(nest.GetKernelStatus()['rng_seeds']) + 1
        seeds = np.random.choice(1000, size=num_seeds, replace=False).tolist()
    elif type(seeds) is int:
        num_seeds = len(nest.GetKernelStatus()['rng_seeds']) + 1
        seeds = np.arange(seeds, num_seeds+seeds).tolist()
    else:
        assert len(seeds) == len(nest.GetKernelStatus()['rng_seeds']) + 1

    nest.SetKernelStatus({'grng_seed': seeds[0]})
    nest.SetKernelStatus({'rng_seeds': seeds[1:]})

    # load modules
    if modname is not None:
        if type(modname) is str:
            modname = [modname]

        for name in modname:
            load_module(name)

    if verbose:
        print('nest using {0:d} threads'.format(num_thread))

        if modname is not None:
            print('loaded nest modules:' + " ".join(modname))

        print('nest seeded with', seeds)


def load_module(modname):
    err = "DynamicModuleManagementError in Install: Module '{0:s}' is " + \
            "loaded already."

    try:
        nest.Install(modname)
    except nest.NESTError as ex:
        err_modname = err.format(modname)

        if ex.args[0] == err_modname:
            pass  # module already loaded
        else:
            raise ex


def multi_logical_and(masks):
    assert isinstance(masks, (tuple, list))
    assert len(masks) >= 2
    for mask in masks: assert isinstance(mask, np.ndarray), 'all masks must be numpy arrays'
    shape = masks[0].shape
    for mask in masks[1:]: assert shape == mask.shape, 'all masks must have the same shape'

    base = np.ones_like(masks[0])
    for mask in masks:
        base = (base & mask)

    return base


def multi_logical_or(masks):
    assert isinstance(masks, (tuple, list))
    assert len(masks) >= 2
    for mask in masks: assert isinstance(mask, np.ndarray), 'all masks must be numpy arrays'
    shape = masks[0].shape
    for mask in masks[1:]: assert shape == mask.shape, 'all masks must have the same shape'

    base = np.zeros_like(masks[0])
    for mask in masks:
        base = (base | mask)

    return base


def make_room(factor, side='top', ax=None):
    assert factor > 0

    if ax is None:
        ax = plt.gca()

    if side in ['top', 'bottom']:
        ylim = ax.get_ylim()
        if side == 'top':
            ax.set_ylim(ylim[0], ylim[1] + factor*(ylim[1]-ylim[0]))
        else:
            ax.set_ylim(ylim[0] - factor*(ylim[1]-ylim[0]), ylim[1])

    elif side in ['right', 'left']:
        xlim = ax.get_xlim()
        if side == 'right':
            ax.set_xlim(xlim[0], xlim[1] + factor*(xlim[1]-xlim[0]))
        else:
            ax.set_xlim(xlim[0] - factor*(xlim[1]-xlim[0]), xlim[1])

    else:
        raise ValueError('bad value for side: must be one of \'top\', \'bottom\', \'left\', \'right\'')

def inset_text(text, vpos='top', hpos='left', ax=None, ha=None, va=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    if type(hpos) is str:
        if hpos == 'left':
            x = .02
            if ha is None:
                ha = 'left'
        elif hpos in ['center', 'middle']:
            x = .5
            if ha is None:
                ha = 'center'
        elif hpos == 'right':
            x = 1.
            if ha is None:
                ha = 'right'
        else:
            raise ValueError('bad value for hpos: '+hpos)
    else:
        x = hpos
        if ha is None:
            ha = 'center'

    if type(vpos) is str:
        if vpos == 'top':
            y = .99
            if va is None:
                va = 'top'
        elif vpos in ['center', 'middle']:
            y = .5
            if va is None:
                va = 'center'
        elif vpos == 'bottom':
            y = .01
            if va is None:
                va = 'bottom'
        else:
            raise ValueError('bad value for vpos: '+vpos)
    else:
        y = vpos
        if va is None:
            va = 'center'

    ax.text(x, y, text, ha=ha, va=va, transform=ax.transAxes, **kwargs)
