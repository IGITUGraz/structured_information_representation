#!/usr/bin/env python3

import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
from os.path import join
import pickle as pkl

from utils import *
import nest

from analysis import *
from inpop import *
from swta import *


def train_content_space(*, load=False, datafile=None, num_assemblies=5, outdir=None, save_data=True, plot=True, show=False):

    assert outdir is not None

    # ----------------------------------------------------------------------
    #  create circuit and save

    if not load:

        # input parameters: defaults

        # training parameters
        t_per_pattern = 200  # ms
        t_silence = 200  # ms
        N_show_pattern = num_assemblies * 40
        #N_show_pattern = 200
        stratified = False  # need to activate for larger numbers of assemblies, e.g. > 10, but not used for the main results in the paper

        rate_on = 100
        #rate_on = 75.
        #rate_on = 60.

        rate_off = .1
        #rate_off = 1.
        #rate_off = 5.

        N_E = 1000
        #N_E = 500
        #N_E = 2000
        #N_E = 3000
        #N_E = 4000

        print('t_per_pattern', t_per_pattern)
        print('t_silence', t_silence)
        print('N_show_pattern', N_show_pattern)
        print('stratified', stratified)
        print('inpop rate_on', rate_on)
        print('inpop rate_off', rate_off)

        activity_mu = 1

        # setup inputs
        inpop = InPop(rate_on=rate_on, rate_off=rate_off, num_pattern=num_assemblies, num_neurons=num_assemblies*25+75)

        # create circuit
        c_space = SWTACircuit('C', activity_mu=activity_mu, N_E=N_E)
        c_space.connect_input(inpop.pop_X)

        # save config
        dump_dict(c_space.config, dumpfile=join(outdir, 'config.json'))


        # train

        weights_initial = c_space.get_weights()

        t_sim = (t_per_pattern + t_silence) * N_show_pattern  # ms

        train_pattern = []
        pattern_queue = []

        for k in range(N_show_pattern):
            print('pattern {0:d}/{1:d}'.format(k+1, N_show_pattern))

            if not stratified:
                pat = inpop.set_random()
            else:
                if len(pattern_queue) == 0:
                    ro = np.arange(num_assemblies)
                    np.random.shuffle(ro)
                    pattern_queue = ro.tolist()

                pat = pattern_queue.pop()
                inpop.set(pat)

            train_pattern += [pat]

            nest.Simulate(t_per_pattern)

            inpop.clear()
            nest.Simulate(t_silence)

        t_train = nest.GetKernelStatus()['time']


        # analysis

        train_spikes_E, _ = c_space.get_spikes_legacy()
        train_spikes_per_neuron = spike_analysis(train_spikes_E['senders'], c_space.pop_E)

        train_spike_rec_E = c_space.spike_rec_E


        # save weights

        c_wiring = c_space.get_wiring()

        p = [num_assemblies, c_wiring, t_per_pattern, t_silence, N_show_pattern]

        if datafile is not None:
            with open(datafile, 'wb') as f:
                pkl.dump(p, f)


    # ----------------------------------------------------------------------
    #  restore

    else:  # load
        with open(datafile, 'rb') as f:
            p = pkl.load(f)

        c_wiring, t_per_pattern, t_silence, N_show_pattern = p

        # setup inputs
        inpop = InPop()

        # create circuit
        c_space = SWTACircuit('C', wiring_EE=c_wiring['EE'], N_E=N_E)
        c_space.connect_input(inpop.pop_X, wiring_XE=c_wiring['XE'])


    # ----------------------------------------------------------------------
    # test

    # connect new spike detectors for convenient plotting
    c_space.clear_spikes()

    test_spike_rec_X = nest.Create('spike_detector')
    nest.Connect(inpop.pop_X, test_spike_rec_X, {"rule": "all_to_all"})

    test_start_times = []
    test_end_times = []

    # simulate one period of silence
    inpop.clear()
    nest.Simulate(t_silence)

    for k in range(inpop.num_pattern):
        test_start_times += [nest.GetKernelStatus()['time']]
        inpop.set(k)
        nest.Simulate(t_per_pattern)

        test_end_times += [nest.GetKernelStatus()['time']]

        inpop.clear()
        nest.Simulate(t_silence)


    # ----------------------------------------------------------------------
    # analysis

    print("---\nanalysis\n---")


    # assembly analysis

    spikes_per_pattern, firing_rates_per_pattern, assemblies, assembly_occurances = assembly_analysis(
            c_space, test_start_times, test_end_times, print_results=True)

    # weight analysis

    if save_data or plot:
        weights = c_space.get_weights()

    # save data

    if save_data:
        def spike_rec_data(spike_rec):
            ev = nest.GetStatus(spike_rec)[0]['events']
            return ev['times'], ev['senders']

        data = dict(
                train_pattern=train_pattern,
                train_spike_rec_E=spike_rec_data(c_space.spike_rec_E),
                train_spike_rec_I=spike_rec_data(c_space.spike_rec_I),
                test_spike_rec_X=spike_rec_data(test_spike_rec_X),
                test_start_times=test_start_times,
                test_end_times=test_end_times,
                weights=weights)

        with open(join(outdir, 'training_data.pkl'), 'wb') as f:
            pkl.dump(data, f)

    # weights in content space

    c_wiring_nn = c_space.get_wiring(normalize_indices=False)

    r_weights = analyze_assembly_weights(c_wiring_nn['EE'], assemblies)
    w_c_within_a, w_c_between_a, w_c_between_a_and_noa, w_c_within_noa = r_weights

    def elementary_stats(w):
        return { 'num_synapses': len(w), 'weight_mean': w.mean(), 'weight_std': w.std()}

    stats = {
            'within_assemblies': elementary_stats(w_c_within_a),
            'between_assemblies': elementary_stats(w_c_between_a),
            'between_assemblies_and_free': elementary_stats(w_c_between_a_and_noa),
            'between_free': elementary_stats(w_c_within_noa)}

    # plots

    if plot:
        plot_neuron_count = 200

        if load == False:
            targets = np.arange(plot_neuron_count) + min(c_space.pop_E)

            plot_training_activity(train_spike_rec_E, N_show_pattern, t_per_pattern+t_silence, targets=targets, div=inpop.num_pattern, save=join(outdir, 'train_spikes_after_{}_pat.pdf'), close=(not show))

            plot_hist(train_spikes_per_neuron, '', xlabel='spikes', save=join(outdir, 'train_spike_hist.pdf'), close=(not show))

        targets_X = np.arange(plot_neuron_count) + min(inpop.pop_X)
        np.random.shuffle(targets_X)
        targets_E = np.arange(plot_neuron_count) + min(c_space.pop_E)
        targets_I = np.arange(plot_neuron_count) + min(c_space.pop_I)
        x_test=[min(test_start_times), max(test_end_times)]
        xlim = [x_test[0]-100., x_test[1]+100.]
        markers = []
        for p in zip(test_start_times, test_end_times):
            markers += p
        xticks_t = [(a + b) / 2. for a, b in zip(test_start_times, test_end_times)]
        xticks_l = [str(x) for x in range(1, 6)]
        xticks = [x for x in zip(xticks_t, xticks_l)]

        plot_spikes_fancy(test_spike_rec_X, targets=targets_X, ylabel = '$\mathcal{X}$', title='', xlabel='time', markers=markers, marker_color_div=2, xticks=xticks, xlim=xlim, wide=True, save=join(outdir, 'test_activity_X.pdf'), close=(not show))
        plot_spikes_fancy(c_space.spike_rec_E, targets=targets_E, ylabel = '$\mathcal{E}$', title='', xlabel='$\;$', markers=markers, marker_color_div=2, xticks=[], xlim=xlim, wide=True, save=join(outdir, 'test_activity_E.pdf'), close=(not show))
        plot_spikes_fancy(c_space.spike_rec_I, targets=targets_I, ylabel = '$\mathcal{I}$', title='', xlabel='$\;$', markers=markers, marker_color_div=2, xticks=[], xlim=xlim, wide=True, save=join(outdir, 'test_activity_I.pdf'), close=(not show))

        plot_dual_weight_hist(weights['XE'], 'XE', weights['EE'], 'EE', join(outdir, 'weights_XE_EE.pdf'), close=(not show))

        plot_synd_weight_correlation(inpop.pop_X, c_space.pop_E, save=join(outdir, 'synd_weight_corr_XE.pdf'), close=(not show))

        plot_assembly_correlations(r_weights, 'in_c', save=join(outdir, 'weights_in_c.pdf'), close=(not show))

        if show:
            plt.show()

    results = {
        'assembly_sizes': [len(a) for a in assemblies],
        'assembly_occurances': assembly_occurances,
    }

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pre-train content space instances for later use with all other scripts')
    meg = parser.add_mutually_exclusive_group(required=True)
    meg.add_argument('-C', dest='name', type=str, help='output content space name')
    meg.add_argument('-S', dest='statistics', action='store_true', help='gather statistics over 10 trainings instead of performing the usual training')
    parser.add_argument('-k', dest='num_assemblies', type=int, default=5, help='number of assemblies to train (default: 5)')

    args = parser.parse_args()
    name = args.name
    num_assemblies = args.num_assemblies

    # operation to perform

    op = 'statistics' if args.statistics else 'train'

    # ----------------------------------------------------------------------

    setup_numpy_and_matplotlib()

    if op == 'train':
        datadir = setup_outdir(join('data', name))
        outdir = setup_outdir(join('out', 'train_content_space', name))
        logfile = join(outdir, 'log.txt')

        logger = Logger(logfile, mode='overwrite')
        sys.stdout = sys.stderr = logger

        load = False
        datafile = join(datadir, 'trained_swta.pkl')

        if not load and os.path.isfile(datafile):
            raise IOError('output file {0:s} already exists'.format(datafile))

        setup_nest(modname='vb_module')
        results = train_content_space(load=load, datafile=datafile, outdir=outdir, num_assemblies=num_assemblies)

        dump_dict(results, dumpfile=join(datadir, 'results.json'), print_stdout=False, key='results')
        dump_dict(results, dumpfile=join(outdir, 'results.json'), print_stdout=False, key='results')

    else:
        outdir = setup_outdir(join('out', 'train_content_space_statistics'))
        logfile = join(outdir, 'log.txt')

        logger = Logger(logfile, mode='overwrite')
        sys.stdout = sys.stderr = logger

        assembly_sizes = []
        assembly_occurances = []

        N = 10

        for n in range(N):
            setup_nest()  # reset kernel
            results = train_content_space(datafile=None, outdir=outdir, num_assemblies=num_assemblies, plot=False)

            assembly_sizes += results['assembly_sizes']
            assembly_occurances += [results['assembly_occurances']]

        assembly_sizes = np.asarray(assembly_sizes)
        assembly_occurances = np.asarray(assembly_occurances)
        assembly_occurances_mean = assembly_occurances.mean(axis=0)
        assembly_occurances_std = assembly_occurances.std(axis=0)

        results = {
            'assembly_size_mean': float(assembly_sizes.mean()),
            'assembly_size_std': float(assembly_sizes.std()),
            'occurance_1_mean': float(assembly_occurances_mean[0]),
            'occurance_1_std': float(assembly_occurances_std[0]),
            'occurance_2_mean': float(assembly_occurances_mean[1]),
            'occurance_2_std': float(assembly_occurances_std[1]),
            'occurance_3_mean': float(assembly_occurances_mean[2]),
            'occurance_3_std': float(assembly_occurances_std[2]),
            'occurance_4_mean': float(assembly_occurances_mean[3]),
            'occurance_4_std': float(assembly_occurances_std[3]),
            'occurance_5_mean': float(assembly_occurances_mean[4]),
            'occurance_5_std': float(assembly_occurances_std[4])
        }

        dump_dict(results, dumpfile=join(outdir, 'results.json'), print_stdout=True)
