#!/usr/bin/env python3

import matplotlib.pyplot as plt
import nest
import numpy as np
from os.path import join
import pickle as pkl

from analysis import *
from neural_space import *
from inpop import *
from utils import *


# parameters for trace extraction

_t_burn_in = 50.  # discarded at beginning so activity can settle
_tau_filter = 20.  # time constant of filter
_len_filter = 100.  # filter length

_C_readout_args = dict(t_burn_in=_t_burn_in, tau_filter=_tau_filter,
        len_filter=_len_filter, use_data='all', noise_var=0, drop_silent=False)


# plot settings

plt.rcParams.update({'figure.max_open_warning': 0})


def setup_variable_binding(outdir, swtafile, variant, config_c, config_v, num_variable_spaces=1):
    if not isinstance(variant, dict):
        assert variant == 'big_V'

    setup_nest(modname='vb_module')

    # load content space parameters

    with open(swtafile, 'rb') as f:
        p = pkl.load(f)

        if len(p) == 4:  # legacy
            wiring, t_per_pattern, t_silence, N_show_pattern = p
            num_assemblies = 5
        else:
            num_assemblies, wiring, t_per_pattern, t_silence, N_show_pattern = p

    rate_on = 100
    #rate_on = 75.
    #rate_on = 60.

    rate_off = .1
    #rate_off = 1.
    #rate_off = 5.

    # setup inputs
    inpop = InPop(rate_on=rate_on, rate_off=rate_off, num_pattern=num_assemblies, num_neurons=num_assemblies*25+75)

    N_pattern = inpop.num_pattern

    print('using swta file', swtafile)
    print('  t_per_pattern', t_per_pattern)
    print('  t_silence', t_silence)
    print('  N_show_pattern', N_show_pattern)

    print('using inpop')
    print('  num_neurons', inpop.num_neurons)
    print('  num_pattern', inpop.num_pattern)
    print('  rate_off', inpop.rate_off)
    print('  rate_on', inpop.rate_on)
    print('  rate_clear', inpop.rate_clear)

    c_activity_mu = 1
    n_activity_mu = 1


    # setup circuit

    print('restoring content space...')

    if isinstance(variant, dict):
        config_c.update({'N_E': int(variant['size_C']), 'N_I': int(variant['size_C']*.25)})

        C_min_ind = min([w[0] for w in wiring['EE']] + [w[1] for w in wiring['EE']])
        C_max_ind = max([w[0] for w in wiring['EE']] + [w[1] for w in wiring['EE']])
        assert C_max_ind - C_min_ind == variant['size_C'] - 1

    c_space = NeuralSpace('C', config=config_c, wiring_EE=wiring['EE'], activity_mu=c_activity_mu)
    c_space.connect_input(inpop.pop_X, wiring_XE=wiring['XE'])

    c_space.set_synaptic_plasticity(XE=False, EE=False)  # freeze input weights


    print('creating variable space...')

    # modifications

    if variant == 'big_V':
        print('  using big neural spaces')

        symmetric = False
        config_v.update({'N_E': 2000, 'N_I': 500})

    elif isinstance(variant, dict):

        symmetric = False
        config_v.update({'N_E': int(variant['size_N']), 'N_I': int(variant['size_N']*.25)})
        assert c_space.config['N_E'] == int(variant['size_C'])
        assert c_space.config['N_I'] == int(variant['size_C']*.25)
        assert len(c_space.pop_E) == int(variant['size_C'])
        assert len(c_space.pop_I) == int(variant['size_C']*.25)

    else:
        raise ValueError()

    n_spaces = []

    for k in range(num_variable_spaces):
        name = chr(ord('V') - k)
        n_space = NeuralSpace(name, config=config_v, activity_mu=n_activity_mu)
        n_space.connect_input_space(c_space)
        c_space.connect_input_space(n_space, symmetric=symmetric)

        n_space.inhibit()

        n_spaces += [n_space]

    # ----------------------------------------------------------------------
    # test

    # connect new spike detectors for convenient plotting
    c_space.clear_spikes()

    t_test_start = []
    t_test_end = []
    labels = []

    # simulate one period of silence
    inpop.clear()
    nest.Simulate(t_silence)

    for k in range(N_pattern):
        print("TEST {}/{}".format(k+1, N_pattern))
        labels += [k]

        t_test_start += [nest.GetKernelStatus()['time']]

        inpop.set(k)
        nest.Simulate(t_per_pattern)

        t_test_end += [nest.GetKernelStatus()['time']]

        inpop.clear()
        nest.Simulate(t_silence)

    # train readouts for decoding from C

    print("train readout on C")

    t_lim = (t_test_start[0], t_test_end[-1])
    spikes_C = c_space.get_spikes(t_lim=t_lim)[0]

    _, x, y = spikes_to_traces(spikes_C, c_space.pop_E, [*zip(t_test_start, t_test_end, labels)], **_C_readout_args)

    C_readout_test = train_readout(x, y)

    # ----------------------------------------------------------------------
    # CREATE assemblies in variable space(s)

    # return a list of lists, i.e. a list containing 2 lists containing 5 items
    # each for num_spaces=2 and num_pattern=5
    t_create_start = []
    t_create_end = []

    k_pattern = list(range(num_assemblies))

    for n, n_space in enumerate(n_spaces):
        t_create_start_cur = []
        t_create_end_cur = []

        for k, pat in enumerate(k_pattern):
            print("CREATE {}/{}".format(len(k_pattern)*n+k+1, len(n_spaces)*len(k_pattern)))

            inpop.clear()
            nest.Simulate(1000.)

            n_space.disinhibit()

            t_create_start_cur += [nest.GetKernelStatus()['time']]
            inpop.set(pat)
            nest.Simulate(1000.)
            t_create_end_cur += [nest.GetKernelStatus()['time']]

            n_space.inhibit(reset_sfa=True)
            c_space.reset_sfa()

        t_create_start += [t_create_start_cur]
        t_create_end += [t_create_end_cur]

    return inpop, c_space, n_spaces, t_test_start, t_test_end, t_create_start, t_create_end, C_readout_test


def store_recall(outdir, swtafile, variant, config_c, config_v, recall_cfg, k_pattern=None, print_results=True, print_assemblies=True, gather_statistics=False, plot=True, show=True):

    # unpack input

    inhibit_V = recall_cfg['inhibit_V']
    t_store = recall_cfg['t_store']
    t_recall_1 = recall_cfg['t_recall_1']
    t_recall_2 = recall_cfg['t_recall_2']

    # setup

    r = setup_variable_binding(
        outdir,
        swtafile,
        variant,
        config_c=config_c,
        config_v=config_v,
        num_variable_spaces=1
    )

    inpop, c_space, n_spaces, t_test_start, t_test_end, t_create_start, t_create_end, C_readout_test = r
    N_pattern = inpop.num_pattern

    # only one variable space is used
    assert len(n_spaces) == 1
    assert len(t_create_start) == 1
    assert len(t_create_end) == 1
    n_space = n_spaces[0]
    t_create_start = t_create_start[0]
    t_create_end = t_create_end[0]

    # assert E_sfa clipping is used
    assert nest.GetStatus(n_space.pop_E)[0]['E_sfa_clip'] == True
    assert nest.GetStatus(n_space.pop_E)[0]['E_sfa_max'] < 0.

    # gather statistics

    if gather_statistics:
        arg = [
            ('a_test_C', c_space, t_test_start, t_test_end, False),
            #('a_create_C', c_space, t_create_start, t_create_end, False),
            ('a_create_V', n_space, t_create_start, t_create_end, False)]

        r = multi_assembly_analysis(arg)

        a_test_C = r['a_test_C']
        #a_create_C = r['a_create_C']
        a_create_V = r['a_create_V']

        # weights in content space

        c_wiring = c_space.get_wiring(normalize_indices=False)

        r = analyze_assembly_weights(c_wiring['EE'], a_test_C)
        w_c_within_a, w_c_between_a, w_c_between_a_and_noa, w_c_within_noa = r

        if plot:
            plot_assembly_correlations(r, 'in_c', save=join(outdir, 'weights_in_c.pdf'), close=(not show))

        # weights in variable space

        n_wiring = n_space.get_wiring(normalize_indices=False)

        r = analyze_assembly_weights(n_wiring['EE'], a_create_V)
        w_n_within_a, w_n_between_a, w_n_between_a_and_noa, w_n_within_noa = r

        if plot:
            plot_assembly_correlations(r, 'in_n', save=join(outdir, 'weights_in_n.pdf'), close=(not show))

        # weights between neural spaces

        r_cn = analyze_assembly_weights(n_wiring['SE_C_to_V'], a_test_C, a_create_V)
        w_cn_within_a, w_cn_between_a, w_cn_between_a_and_noa, w_cn_within_noa = r_cn

        r_nc = analyze_assembly_weights(c_wiring['SE_V_to_C'], a_create_V, a_test_C)
        w_nc_within_a, w_nc_between_a, w_nc_between_a_and_noa, w_nc_within_noa = r_nc

        if plot:
            plot_assembly_correlations(r_cn, 'c_to_n', save=join(outdir, 'weights_c_to_n.pdf'), close=(not show))
            plot_assembly_correlations(r_nc, 'n_to_c', save=join(outdir, 'weights_n_to_c.pdf'), close=(not show))

        # stats

        def elementary_stats(w):
            return { 'num_synapses': len(w), 'weight_mean': w.mean(), 'weight_std': w.std()}

        stats = {
                'content_space': {
                    'within_assemblies': elementary_stats(w_c_within_a),
                    'between_assemblies': elementary_stats(w_c_between_a),
                    'between_assemblies_and_free': elementary_stats(w_c_between_a_and_noa),
                    'between_free': elementary_stats(w_c_within_noa)},
                'variable_space': {
                    'within_assemblies': elementary_stats(w_n_within_a),
                    'between_assemblies': elementary_stats(w_n_between_a),
                    'between_assemblies_and_free': elementary_stats(w_n_between_a_and_noa),
                    'between_free': elementary_stats(w_n_within_noa)},
                'feedforward_projection': {
                    'between_linked_assemblies': elementary_stats(w_cn_within_a),
                    'between_nonlinked_assemblies': elementary_stats(w_cn_between_a)},
                'feedback_projection': {
                    'between_linked_assemblies': elementary_stats(w_nc_within_a),
                    'between_nonlinked_assemblies': elementary_stats(w_nc_between_a)}}

        dump_dict(stats, dumpfile=join(outdir, 'statistics.json'))

        # spontaneous activity of content space

        t_spontaneous = 200.
        plot_num = 100
        inpop.clear()

        c_space.disinhibit()
        n_space.inhibit(reset_sfa=True)

        t_c_spontaneous_start = nest.GetKernelStatus()['time']

        nest.Simulate(t_spontaneous)

        t_c_spontaneous_end = nest.GetKernelStatus()['time']

        spikes_exc, spikes_inh = c_space.get_spikes(t_lim=(t_c_spontaneous_start, t_c_spontaneous_end))

        spontaneous_rate_c_exc = len(spikes_exc[0]) / len(c_space.pop_E) / t_spontaneous * 1000
        spontaneous_rate_c_inh = len(spikes_inh[0]) / len(c_space.pop_I) / t_spontaneous * 1000
        spontaneous_rate_c = (len(spikes_exc[0])+len(spikes_inh[0])) / len(c_space.pop) / t_spontaneous * 1000

        plot_spikes_fancy(
                c_space.spike_rec_E,
                targets=np.arange(plot_num)+min(c_space.pop_E),
                xlim=(t_c_spontaneous_start, t_c_spontaneous_end),
                save=join(outdir, 'spontaneous_activity_c_exc.pdf'))

        plot_spikes_fancy(
                c_space.spike_rec_I,
                targets=np.arange(plot_num)+min(c_space.pop_I),
                xlim=(t_c_spontaneous_start, t_c_spontaneous_end),
                save=join(outdir, 'spontaneous_activity_c_inh.pdf'))

        # spontaneous activity of variable space

        c_space.inhibit()
        n_space.disinhibit()

        t_n_spontaneous_start = nest.GetKernelStatus()['time']

        nest.Simulate(t_spontaneous)

        t_n_spontaneous_end = nest.GetKernelStatus()['time']

        spikes_exc, spikes_inh = n_space.get_spikes(t_lim=(t_n_spontaneous_start, t_n_spontaneous_end))

        spontaneous_rate_n_exc = len(spikes_exc[0]) / len(n_space.pop_E) / t_spontaneous * 1000
        spontaneous_rate_n_inh = len(spikes_inh[0]) / len(n_space.pop_I) / t_spontaneous * 1000
        spontaneous_rate_n = (len(spikes_exc[0])+len(spikes_inh[0])) / len(n_space.pop) / t_spontaneous * 1000

        plot_spikes_fancy(
                n_space.spike_rec_E,
                targets=np.arange(plot_num)+min(n_space.pop_E),
                xlim=(t_n_spontaneous_start, t_n_spontaneous_end),
                save=join(outdir, 'spontaneous_activity_n_exc.pdf'))

        plot_spikes_fancy(
                n_space.spike_rec_I,
                targets=np.arange(plot_num)+min(n_space.pop_I),
                xlim=(t_n_spontaneous_start, t_n_spontaneous_end),
                save=join(outdir, 'spontaneous_activity_n_inh.pdf'))

        # stats

        spontaneous_stats = {
                'time': t_spontaneous,
                'content_space': {
                    'mean_excitatory_rate': spontaneous_rate_c_exc,
                    'mean_inhibitory_rate': spontaneous_rate_c_inh,
                    'mean_rate': spontaneous_rate_c},
                'variable_space': {
                    'mean_excitatory_rate': spontaneous_rate_n_exc,
                    'mean_inhibitory_rate': spontaneous_rate_n_inh,
                    'mean_rate': spontaneous_rate_n}}

        dump_dict(spontaneous_stats, dumpfile=join(outdir, 'spontaneous_activtiy_statistics.json'))

        if not show:
            plt.close('all')

        return

    # check parameters

    if k_pattern is None:
        k_pattern = list(range(N_pattern))

    for p in k_pattern:
        assert p < N_pattern

    # perform STORE/RECALL

    t_store_start = []
    t_store_end = []
    t_recall_start = []
    t_recall_end = []
    labels = []

    for k, pat in enumerate(k_pattern):
        print("STORE/RECALL {}/{}".format(k+1, len(k_pattern)))
        labels += [pat]

        # STORE

        n_space.inhibit(reset_sfa=True)

        inpop.clear()
        nest.Simulate(1000.)

        t_store_start += [nest.GetKernelStatus()['time']]

        n_space.disinhibit()

        inpop.set(pat)
        nest.Simulate(t_store)

        t_store_end += [nest.GetKernelStatus()['time']]


        # RECALL

        c_space.inhibit()
        if inhibit_V:
            n_space.inhibit()

        inpop.clear()
        nest.Simulate(5000.)

        t_recall_start += [nest.GetKernelStatus()['time']]

        if inhibit_V:
            n_space.disinhibit()

        if t_recall_1 > 0:
            nest.Simulate(t_recall_1)

        c_space.disinhibit()
        nest.Simulate(t_recall_2)

        t_recall_end += [nest.GetKernelStatus()['time']]

    # ----------------------------------------------------------------------
    # analysis

    print('analysis')

    if plot:
        c_weights = c_space.get_weights()
        n_weights = n_space.get_weights()

    if inhibit_V:
        for t0, t1 in zip(t_store_end, t_recall_start):
            # make sure neural spaces are silent when they should be
            spikes_n = nest.GetStatus(n_space.spike_rec_E)[0]['events']['times']

            delta = 50.  # allow spikes in first 50 ms of each inhibited period

            spikes_v_delay = (spikes_n > t0+delta) & (spikes_n < t1)
            nspikes_v_d = sum(spikes_v_delay)

            if nspikes_v_d > 0:
                print('WARNING: spikes in V during delay period (count: {0:d})'.format(nspikes_v_d))


    # assembly analysis

    print_test_C = True if print_assemblies else False

    arg = [
        ('a_test_C', c_space, t_test_start, t_test_end, print_test_C),
        ('a_create_C', c_space, t_create_start, t_create_end, False),
        ('a_create_V', n_space, t_create_start, t_create_end, False),
        ('a_recall_C', c_space, t_recall_start, t_recall_end, False),
        ('a_recall_V', n_space, t_recall_start, t_recall_end, False)]

    r = multi_assembly_analysis(arg)

    cost_v = 0
    cost_c = 0
    results_k = []

    for j in range(len(k_pattern)):
        a_test_C = r['a_test_C'][k_pattern[j]]
        a_create_C = r['a_create_C'][j]
        a_create_V = r['a_create_V'][j]
        a_recall_C = r['a_recall_C'][j]
        a_recall_V = r['a_recall_V'][j]

        # assembly match V: create vs. recall
        am_V_CR = assembly_match(a_create_V, a_recall_V)

        # assembly match C: test vs. recall
        am_C_TR = assembly_match(a_test_C, a_recall_C)

        if print_results:
            print('STORE/RECALL {0:d}/{1:d}'.format(j+1, len(k_pattern)))

            if print_assemblies:
                print('assembly in C during test:')
                print(a_test_C, 'size:', len(a_test_C))
                print('assembly in C during create:')
                print(a_create_C, 'size:', len(a_create_C))
                print('assembly in V during create:')
                print(a_create_V, 'size:', len(a_create_V))
                print('assembly in V during recall:')
                print(a_recall_V, 'size:', len(a_recall_V))
                print('assembly in C during recall:')
                print(a_recall_C, 'size:', len(a_recall_C))
            print('  assembly match in V (create vs. recall): {} shared, {} only in create, {} only in recall'.format(*am_V_CR))
            print('  assembly match in C (test vs. recall): {} shared, {} only in test, {} only in recall'.format(*am_C_TR))

        cost_v += am_V_CR[1] + am_V_CR[2]
        cost_c += am_C_TR[1] + am_C_TR[2]

        # calculate success criterion
        # if > threshold % of the test assembly are active, and if
        # <= (100 - threshold) % are wrongly active, count as success
        threshold = .8

        if len(a_test_C) > 0 and \
                (am_C_TR[0] / len(a_test_C) > threshold) and \
                (am_C_TR[2] / len(a_test_C) <= (1-threshold)):
            success = True
        else:
            success = False

        results_k += [{
            'a_test_C_size': len(a_test_C),
            'a_create_C_size': len(a_create_C),
            'a_create_V_size': len(a_create_V),
            'a_recall_C_size': len(a_recall_C),
            'a_recall_V_size': len(a_recall_V),
            'match_V_shared': am_V_CR[0],
            'match_V_only_in_create': am_V_CR[1],
            'match_V_only_in_recall': am_V_CR[2],
            'match_C_shared': am_C_TR[0],
            'match_C_only_in_test': am_C_TR[1],
            'match_C_only_in_recall': am_C_TR[2],
            'success': success,
        }]

    # check success with readout

    t_lim = (t_recall_start[0], t_recall_end[-1])
    spikes_C = c_space.get_spikes(t_lim=t_lim)[0]

    args = _C_readout_args.copy()
    args['t_burn_in'] += t_recall_1

    _, xt, yt = spikes_to_traces(spikes_C, c_space.pop_E, [*zip(t_recall_start, t_recall_end, labels)], **args)

    readout_error = C_readout_test(xt, yt)

    print('readout error', readout_error)


    # ----------------------------------------------------------------------
    # plots

    if plot:
        plot_multi_weight_hist(c_weights, save=join(outdir, 'weights_C'))
        plot_multi_weight_hist(n_weights, save=join(outdir, 'weights_V'))

        time_test = [min(t_test_start), max(t_test_end)]
        plot_spikes(c_space.spike_rec_E, '$C$ during test', xlim=time_test, save=join(outdir, 'test_C.pdf'))

        # fancy plots

        for j in range(len(k_pattern)):
            num = 200
            targets_x = np.arange(num) + min(inpop.pop_X)
            np.random.shuffle(targets_x)
            targets_c = np.arange(num) + min(c_space.pop_E)
            targets_v = np.arange(num) + min(n_space.pop_E)

            xlim1 = [t_store_start[j]-50., t_store_end[j]+200.]
            xlim2 = [t_recall_start[j]-200., t_recall_end[j]]
            markers = [t_store_start[j], t_store_end[j], t_recall_start[j]]
            labels = ['LOAD', 'DELAY', 'RECALL']
            xticks_labeled = [x for x in zip(markers, labels)]
            xticks_unlabeled = []

            plot_spikes_fancy(inpop.spike_rec, targets=targets_x, xlabel='time', ylabel='$\mathcal{X}$', xlim=xlim1, markers=markers, xticks=xticks_labeled, hide_spines_right=True, save=join(outdir, 'overview_X_{}a.pdf'.format(j)))
            plot_spikes_fancy(inpop.spike_rec, targets=targets_x, xlabel='time', ylabel='', xlim=xlim2, markers=markers, xticks=xticks_labeled, hide_spines_left=True, save=join(outdir, 'overview_X_{}b.pdf'.format(j)))
            plot_spikes_fancy(c_space.spike_rec_E, targets=targets_c, xlabel='', ylabel='$\mathcal{C}$', xlim=xlim1, markers=markers, xticks=xticks_unlabeled, hide_spines_right=True, save=join(outdir, 'overview_C_{}a.pdf'.format(j)))
            plot_spikes_fancy(c_space.spike_rec_E, targets=targets_c, xlabel='', ylabel='', xlim=xlim2, markers=markers, xticks=xticks_unlabeled, hide_spines_left=True, save=join(outdir, 'overview_C_{}b.pdf'.format(j)))
            plot_spikes_fancy(n_space.spike_rec_E, targets=targets_v, xlabel='', ylabel='$\mathcal{N}_V$', xlim=xlim1, markers=markers, xticks=xticks_unlabeled, hide_spines_right=True, save=join(outdir, 'overview_V_{}a.pdf'.format(j)))
            plot_spikes_fancy(n_space.spike_rec_E, targets=targets_v, xlabel='', ylabel='', xlim=xlim2, markers=markers, xticks=xticks_unlabeled, hide_spines_left=True, save=join(outdir, 'overview_V_{}b.pdf'.format(j)))

        # also save all spike data
        spikes_X = inpop.get_spikes()
        spikes_C = c_space.get_spikes()[0]
        spikes_V = n_space.get_spikes()[0]

        with open(join(outdir, 'all-spikes.pkl'), 'wb') as f:
            I_pop_X = inpop.pop_X
            C_pop_E = c_space.pop_E
            V_pop_E = n_space.pop_E

            pkl.dump([
                t_create_start,
                t_create_end,
                t_store_start,
                t_store_end,
                t_recall_start,
                t_recall_end,
                spikes_X,
                spikes_C,
                spikes_V,
                I_pop_X,
                C_pop_E,
                V_pop_E], f)

        if not show:
            plt.close('all')

    results = {
        'k_pattern': k_pattern,
        'results_k': results_k,
        'readout_error': readout_error,
    }

    return cost_c, cost_v, results


def store_copy_recall(outdir, swtafile, variant, config_c, config_v, recall_cfg, k_pattern=None, print_results=True, plot=True, show=False):

    # unpack input

    inhibit_V = recall_cfg['inhibit_V']
    t_store = recall_cfg['t_store']
    t_recall_1 = recall_cfg['t_recall_1']
    t_recall_2 = recall_cfg['t_recall_2']

    dump_dict(config_v, ordered=True, key='config_v')
    dump_dict(config_c, ordered=True, key='config_c')

    # procedure: copy pointer in V to U
    #
    # 1. clear network activity
    # 2. perform create op for t_store, U inhibited
    # 3. delay for t_copy_delay_1, C and U inhibited (and possibly V)
    # 4. recall from V for t_recall_1, C and U inhibited
    # 5. recall from V for t_recall_2, U inhibited
    # 6. recall from V for t_copy_create
    # 7. delay for t_copy_delay_2, C and V inhibited (and possibly U)
    # 8. recall from U for t_recall_1, C and V inhibited
    # 9. recall from U for t_recall_2, V inhibited

    t_copy_delay_1 = 400.  # time to wait after first create in V
    t_copy_create = 100.  # time of disinhibited U space after recall from V
    t_copy_delay_2 = 400.  # delay before recall from U


    # setup

    r = setup_variable_binding(
        outdir,
        swtafile,
        variant,
        config_c=config_c,
        config_v=config_v,
        num_variable_spaces=2
    )

    inpop, c_space, n_spaces, t_test_start, t_test_end, t_create_start, t_create_end, C_readout_test = r
    N_pattern = inpop.num_pattern

    assert len(n_spaces) == 2
    assert len(t_create_start) == 2
    assert len(t_create_end) == 2
    n_space_v = n_spaces[0]
    n_space_u = n_spaces[1]
    t_create_start_V = t_create_start[0]
    t_create_start_U = t_create_start[1]
    t_create_end_V = t_create_end[0]
    t_create_end_U = t_create_end[1]

    # assert E_sfa clipping is used

    assert nest.GetStatus(n_space_v.pop_E)[0]['E_sfa_clip'] == True
    assert nest.GetStatus(n_space_v.pop_E)[0]['E_sfa_max'] < 0.
    assert nest.GetStatus(n_space_u.pop_E)[0]['E_sfa_clip'] == True
    assert nest.GetStatus(n_space_u.pop_E)[0]['E_sfa_max'] < 0.

    # check parameters

    if k_pattern is None:
        # test with random pattern
        k_pattern = np.random.randint(0, N_pattern)

    assert type(k_pattern) is int
    assert k_pattern < N_pattern


    # perform STORE on V

    print("STORE")

    # clear network activity: inhibit V und U spaces, give random input for C

    n_space_v.inhibit()
    n_space_u.inhibit()

    inpop.clear()
    nest.Simulate(1000.)

    # perform short create op from C to V

    n_space_v.disinhibit()

    t_store_start = nest.GetKernelStatus()['time']

    inpop.set(k_pattern)
    nest.Simulate(t_store)

    t_store_end = nest.GetKernelStatus()['time']

    # first delay while pointer is saved in V

    c_space.inhibit()
    if inhibit_V:
        n_space_v.inhibit()

    inpop.clear()
    nest.Simulate(t_copy_delay_1)

    print("COPY")

    # recall from V to C in two stages

    if inhibit_V:
        n_space_v.disinhibit()

    t_copy_start = nest.GetKernelStatus()['time']

    if t_recall_1 > 0:
        nest.Simulate(t_recall_1)

    c_space.disinhibit()
    nest.Simulate(t_recall_2)

    # now, copy pointer by additionally activating U

    n_space_u.disinhibit()
    nest.Simulate(t_copy_create)

    t_copy_end = nest.GetKernelStatus()['time']

    # delay again before recall from U begins

    c_space.inhibit()
    n_space_v.inhibit()
    #n_space_v.inhibit(reset_sfa=True)  # reset to mute space completely

    if inhibit_V:
        n_space_u.inhibit()

    nest.Simulate(t_copy_delay_2)

    print("RECALL")

    # recall from U in two stages

    t_recall_start = nest.GetKernelStatus()['time']

    if inhibit_V:
        n_space_u.disinhibit()

    if t_recall_1 > 0:
        nest.Simulate(t_recall_1)

    c_space.disinhibit()
    nest.Simulate(t_recall_2)

    t_recall_end = nest.GetKernelStatus()['time']

    # ----------------------------------------------------------------------
    # analysis

    print('analysis')

    if inhibit_V:
        # make sure neural spaces are silent when they should be
        spikes_nv = nest.GetStatus(n_space_v.spike_rec_E)[0]['events']['times']
        spikes_nu = nest.GetStatus(n_space_u.spike_rec_E)[0]['events']['times']

        delta = 50.  # allow spikes in first 50 ms of each inhibited period

        spikes_v_delay_1 = (spikes_nv > t_store_end+delta) & (spikes_nv < t_copy_start)
        spikes_u_delay_1 = (spikes_nu > t_store_end+delta) & (spikes_nu < t_copy_start)
        spikes_v_delay_2 = (spikes_nv > t_copy_end+delta) & (spikes_nv < t_recall_start)
        spikes_u_delay_2 = (spikes_nu > t_copy_end+delta) & (spikes_nu < t_recall_start)

        nspikes_v_d1 = sum(spikes_v_delay_1)
        nspikes_u_d1 = sum(spikes_u_delay_1)
        nspikes_v_d2 = sum(spikes_v_delay_2)
        nspikes_u_d2 = sum(spikes_u_delay_2)

        if nspikes_v_d1 > 0:
            print('WARNING: spikes in V during delay period 1 (count: {0:d})'.format(nspikes_v_d1))
        if nspikes_u_d1 > 0:
            print('WARNING: spikes in U during delay period 1 (count: {0:d})'.format(nspikes_u_d1))
        if nspikes_v_d2 > 0:
            print('WARNING: spikes in V during delay period 2 (count: {0:d})'.format(nspikes_v_d2))
        if nspikes_u_d2 > 0:
            print('WARNING: spikes in U during delay period 2 (count: {0:d})'.format(nspikes_u_d2))

        if nspikes_v_d1 == 0 and nspikes_u_d1 == 0 and \
            nspikes_v_d2 == 0 and nspikes_u_d2 == 0:
            print('neural inhibition ok.')

    if plot:
        c_weights = c_space.get_weights()
        nv_weights = n_space_v.get_weights()
        nu_weights = n_space_u.get_weights()


    # assembly analysis

    t_copy_start_U = t_copy_start + t_recall_1 + t_recall_2

    arg = [
        ('a_test_C', c_space, t_test_start, t_test_end, False),
        ('a_store_C', c_space, [t_store_start], [t_store_end], False),
        ('a_store_V', n_space_v, [t_store_start], [t_store_end], False),
        ('a_copy_C', n_space_v, [t_copy_start], [t_copy_end], False),
        ('a_copy_V', n_space_v, [t_copy_start], [t_copy_end], False),
        ('a_copy_U', n_space_u, [t_copy_start_U], [t_copy_end], False),
        ('a_recall_C', c_space, [t_recall_start], [t_recall_end], False),
        ('a_recall_U', n_space_u, [t_recall_start], [t_recall_end], False)]

    r = multi_assembly_analysis(arg)

    a_test_C = r['a_test_C'][k_pattern]
    a_store_C = r['a_store_C'][0]
    a_store_V = r['a_store_V'][0]
    a_copy_C = r['a_copy_C'][0]
    a_copy_V = r['a_copy_V'][0]
    a_copy_U = r['a_copy_U'][0]
    a_recall_C = r['a_recall_C'][0]
    a_recall_U = r['a_recall_U'][0]

    # assembly match C: test vs. store
    am_C_TS = assembly_match(a_test_C, a_store_C)

    # assembly match C: test vs. copy
    am_C_TC = assembly_match(a_test_C, a_copy_C)

    # assembly match C: test vs. recall
    am_C_TR = assembly_match(a_test_C, a_recall_C)

    # assembly match V: store vs. copy
    am_V_SC = assembly_match(a_store_V, a_copy_V)

    # assembly match U: copy vs. recall
    am_U_CR = assembly_match(a_copy_U, a_recall_U)

    if print_results:
        print('assembly match in C (test vs. recall): {} shared, {} only in test, {} only in recall'.format(*am_C_TR))

    cost_c = am_C_TR[1] + am_C_TR[2]

    # calculate success criterion
    # if > threshold % of the test assembly are active, and if
    # <= (100 - threshold) % are wrongly active, count as success
    threshold = .8

    if (am_C_TR[0] / len(a_test_C) > threshold) and \
        (am_C_TR[2] / len(a_test_C) <= (1-threshold)):
        success = True
    else:
        success = False

    # check success with readout

    t_lim = (t_recall_start, t_recall_end)
    spikes_C = c_space.get_spikes(t_lim=t_lim)[0]

    args = _C_readout_args.copy()
    args['t_burn_in'] += t_recall_1

    label = k_pattern
    _, xt, yt = spikes_to_traces(spikes_C, c_space.pop_E, [(t_recall_start, t_recall_end, label)], **args)

    readout_error = C_readout_test(xt, yt)

    print('readout error', readout_error)

    # save results

    results = {
        'a_test_C_size': len(a_test_C),
        'a_store_C_size': len(a_store_C),
        'a_store_V_size': len(a_store_V),
        'a_copy_C_size': len(a_copy_C),
        'a_copy_V_size': len(a_copy_V),
        'a_copy_U_size': len(a_copy_U),
        'a_recall_C_size': len(a_recall_C),
        'a_recall_U_size': len(a_recall_U),
        'match_C_TS_shared': am_C_TS[0],
        'match_C_TS_missing': am_C_TS[1],
        'match_C_TS_excess': am_C_TS[2],
        'match_C_TC_shared': am_C_TC[0],
        'match_C_TC_missing': am_C_TC[1],
        'match_C_TC_excess': am_C_TC[2],
        'match_C_TR_shared': am_C_TR[0],
        'match_C_TR_missing': am_C_TR[1],
        'match_C_TR_excess': am_C_TR[2],
        'match_V_SC_shared': am_V_SC[0],
        'match_V_SC_missing': am_V_SC[1],
        'match_V_SC_excess': am_V_SC[2],
        'match_U_CR_shared': am_U_CR[0],
        'match_U_CR_missing': am_U_CR[1],
        'match_U_CR_excess': am_U_CR[2],
        'success': success,
        'readout_error': readout_error,
    }


    # ----------------------------------------------------------------------
    # plots

    if plot:
        plot_multi_weight_hist(c_weights, save=join(outdir, 'weights_C'))
        plot_multi_weight_hist(nv_weights, save=join(outdir, 'weights_V'))
        plot_multi_weight_hist(nu_weights, save=join(outdir, 'weights_U'))

        t_store = [t_store_start, t_store_end]
        t_copy = [t_copy_start, t_copy_end]
        t_recall = [t_recall_start, t_recall_end]

        # fancy plots

        num = 200
        targets_x = np.arange(num) + min(inpop.pop_X)
        np.random.shuffle(targets_x)
        targets_c = np.arange(num) + min(c_space.pop_E)
        targets_v = np.arange(num) + min(n_space_v.pop_E)
        targets_u = np.arange(num) + min(n_space_u.pop_E)

        xlim = [t_store_start-50., t_recall_end]
        markers = [t_store_start, t_store_end, t_copy_start, t_copy_end, t_recall_start]
        labels = ['LOAD', 'DELAY', 'COPY', 'DELAY', 'RECALL']
        xticks_labeled = [x for x in zip(markers, labels)]
        xticks_unlabeled = []

        plot_spikes_fancy(inpop.spike_rec, targets=targets_x, xlabel='time', ylabel='$\mathcal{X}$', xlim=xlim, markers=markers, xticks=xticks_labeled, wide=True, save=join(outdir, 'overview_X.pdf'))
        plot_spikes_fancy(c_space.spike_rec_E, targets=targets_c, xlabel='', ylabel='$\mathcal{C}$', xlim=xlim, markers=markers, xticks=xticks_unlabeled, wide=True, save=join(outdir, 'overview_C.pdf'))
        plot_spikes_fancy(n_space_v.spike_rec_E, targets=targets_v, xlabel='', ylabel='$\mathcal{N}_V$', xlim=xlim, markers=markers, xticks=xticks_unlabeled, wide=True, save=join(outdir, 'overview_V.pdf'))
        plot_spikes_fancy(n_space_u.spike_rec_E, targets=targets_u, xlabel='', ylabel='$\mathcal{N}_U$', xlim=xlim, markers=markers, xticks=xticks_unlabeled, wide=True, save=join(outdir, 'overview_U.pdf'))

        # debug: check all neurons for activity during DELAYs
        plot_spikes_fancy(n_space_v.spike_rec_E, targets=None, xlabel='', ylabel='$\mathcal{N}_V$', xlim=xlim, markers=markers, xticks=xticks_unlabeled, wide=True)
        plot_spikes_fancy(n_space_u.spike_rec_E, targets=None, xlabel='', ylabel='$\mathcal{N}_U$', xlim=xlim, markers=markers, xticks=xticks_unlabeled, wide=True)

        # also save all spike data
        spikes_X = inpop.get_spikes()
        spikes_C = c_space.get_spikes()[0]
        spikes_V = n_space_v.get_spikes()[0]
        spikes_U = n_space_u.get_spikes()[0]

        with open(join(outdir, 'all-spikes.pkl'), 'wb') as f:
            I_pop_X = inpop.pop_X
            C_pop_E = c_space.pop_E
            V_pop_E = n_space_v.pop_E
            U_pop_E = n_space_u.pop_E

            pkl.dump([t_create_start, t_create_end, t_store_start, t_store_end,
                t_copy_start, t_copy_end, t_recall_start, t_recall_end,
                spikes_X, spikes_C, spikes_V, spikes_U, I_pop_X, C_pop_E,
                V_pop_E, U_pop_E], f)

        if not show:
            plt.close('all')

    return cost_c, results


def compare(outdir, swtafile, variant, config_c, config_v, recall_cfg, k_pattern_1=None, k_pattern_2=None, n_readout=1, print_results=True, plot=True, show=True):

    # unpack input

    inhibit_V = recall_cfg['inhibit_V']
    t_store = recall_cfg['t_store']
    t_recall_1 = recall_cfg['t_recall_1']
    t_recall_2 = recall_cfg['t_recall_2']

    dump_dict(config_v, ordered=True, key='config_v')
    dump_dict(config_c, ordered=True, key='config_c')

    # procedure:
    #
    # 1. clear network activity
    # 2. perform create op using k_pattern_1 for t_store in V, U inhibited
    # 3. delay for t_delay
    # 4. perform create op using k_pattern_1 for t_store in U, V inhibited
    #    note: this is only possible if inhibit_V is True, if not (persistent
    #    activity case), we need to set all the weights between C and V to
    #    zero for this period
    # 5. delay for t_delay
    # 6. create readout neuron
    # 7. recall from U for t_recall_1 with C inhibited, then disinhibit C and
    #    recall for t_recall_2
    # 8. recall from V for t_recall_1 with C inhibited, then disinhibit C and
    #    recall for t_recall_2
    #    note: in case of inhibit_V = False, we need to restore the weights

    t_delay = 50.  # time to wait between operations


    # setup

    r = setup_variable_binding(
        outdir,
        swtafile,
        variant,
        config_c=config_c,
        config_v=config_v,
        num_variable_spaces=2
    )

    inpop, c_space, n_spaces, t_test_start, t_test_end, t_create_start, t_create_end, C_readout_test = r
    N_pattern = inpop.num_pattern

    assert len(n_spaces) == 2
    assert len(t_create_start) == 2
    assert len(t_create_end) == 2
    n_space_v = n_spaces[0]
    n_space_u = n_spaces[1]
    t_create_start_V = t_create_start[0]
    t_create_start_U = t_create_start[1]
    t_create_end_V = t_create_end[0]
    t_create_end_U = t_create_end[1]

    # assert E_sfa clipping is used

    assert nest.GetStatus(n_space_v.pop_E)[0]['E_sfa_clip'] == True
    assert nest.GetStatus(n_space_v.pop_E)[0]['E_sfa_max'] < 0.
    assert nest.GetStatus(n_space_u.pop_E)[0]['E_sfa_clip'] == True
    assert nest.GetStatus(n_space_u.pop_E)[0]['E_sfa_max'] < 0.

    # check parameters

    if k_pattern_1 is None:
        # test with random pattern
        k_pattern_1 = np.random.randint(0, N_pattern)

    if k_pattern_2 is None:
        # test with random pattern
        k_pattern_2 = np.random.randint(0, N_pattern)

    assert type(k_pattern_1) is int
    assert type(k_pattern_2) is int
    assert k_pattern_1 < N_pattern
    assert k_pattern_2 < N_pattern


    # perform STORE on V

    print("STORE")

    # clear network activity: inhibit V und U spaces, give noise input to C

    n_space_v.inhibit()
    n_space_u.inhibit()

    inpop.clear()
    nest.Simulate(1000.)

    # perform create op using k_pattern_1 for t_store in V, U inhibited

    print("LOAD 1")

    n_space_v.disinhibit()

    t_store_1_start = nest.GetKernelStatus()['time']

    inpop.set(k_pattern_1)
    nest.Simulate(t_store)

    t_store_1_end = nest.GetKernelStatus()['time']

    # delay for t_delay

    c_space.inhibit()
    if inhibit_V:
        n_space_v.inhibit()

    inpop.clear()
    nest.Simulate(t_delay)

    # perform create op using k_pattern_1 for t_store in U, V inhibited

    print("LOAD 2")

    c_space.disinhibit()
    n_space_u.disinhibit()

    if not inhibit_V:
        # save all weights between C und V, and then set them to zero, also
        # set learning rate to zero to prevent adaptation

        conn_C_V = nest.GetConnections(c_space.pop_E, n_space_v.pop_E)
        conn_V_C = nest.GetConnections(n_space_v.pop_E, c_space.pop_E)

        w_C_V = [s['weight'] for s in nest.GetStatus(conn_C_V)]
        w_V_C = [s['weight'] for s in nest.GetStatus(conn_V_C)]
        l_C_V = [s['lambda'] for s in nest.GetStatus(conn_C_V)]
        l_V_C = [s['lambda'] for s in nest.GetStatus(conn_V_C)]

        nest.SetStatus(conn_C_V, {'weight': 0.})
        nest.SetStatus(conn_V_C, {'weight': 0.})
        nest.SetStatus(conn_C_V, {'lambda': 0.})
        nest.SetStatus(conn_V_C, {'lambda': 0.})

    t_store_2_start = nest.GetKernelStatus()['time']

    inpop.set(k_pattern_2)
    nest.Simulate(t_store)

    t_store_2_end = nest.GetKernelStatus()['time']

    # delay for t_delay

    c_space.inhibit()
    if inhibit_V:
        n_space_u.inhibit()

    inpop.clear()
    nest.Simulate(t_delay)

    # create readout neuron(s)

    if n_readout == 1:
        ro_params = {
                'V_m': -60.,
                'E_L': -60.,
                'C_m': 1000.,
                'tau_m': 20.,
                't_ref': 0.,
                'V_th': 1e3,
                'I_e': 0.,
        }
        w = 1.
    else:
        ro_params = {
                'V_m': -60.,
                'E_L': -60.,
                'V_reset': -60.,
                'C_m': 500.,
                'tau_m': 20.,
                't_ref': 5.,
                'V_th': -20.,
                'I_e': 0.,
        }
        w = 50.

    ro_neurons = nest.Create('iaf_psc_delta', n_readout, ro_params)

    # depressing params from:
    # Gupta, Anirudh, Yun Wang, and Henry Markram. "Organizing principles for
    # a diversity of GABAergic interneurons and synapses in the neocortex."
    # Science 287.5451 (2000): 273-278.

    U = .25
    D = .706
    F = .021

    syn_params = {
            'tau_psc': 3.0,
            'tau_fac': F * 1000,  # facilitation time constant in ms
            'tau_rec': D * 1000,  # recovery time constant in ms
            'U': U,  # utilization
            'delay': 1.,
            'u': 0.0,
            'x': 1.0}
    nest.CopyModel('tsodyks_synapse', 'depressing_syn', syn_params)

    syn_spec = {'model': 'depressing_syn', 'weight': w}

    if n_readout == 1:
        rule = {'rule': 'all_to_all'}
        nest.Connect(c_space.pop_E, ro_neurons, rule, syn_spec)

        ro_vm = nest.Create('voltmeter', params={'interval': .1})
        nest.Connect(ro_vm, ro_neurons)
    else:
        rule = {'rule': 'pairwise_bernoulli', 'p': .1}
        nest.Connect(c_space.pop_E, ro_neurons, rule, syn_spec)

        ro_sd = nest.Create('spike_detector')
        nest.Connect(ro_neurons, ro_sd)

    # recall from U for t_recall_1 with C inhibited, then disinhibit C and
    # recall for t_recall_2

    print("RECALL 1")

    t_recall_1_start = nest.GetKernelStatus()['time']

    if inhibit_V:
        n_space_u.disinhibit()

    if t_recall_1 > 0:
        nest.Simulate(t_recall_1)

    c_space.disinhibit()
    nest.Simulate(t_recall_2)

    t_recall_1_end = nest.GetKernelStatus()['time']

    # recall from V for t_recall_1 with C inhibited, then disinhibit C and
    # recall for t_recall_2

    print("RECALL 2")

    t_recall_2_start = nest.GetKernelStatus()['time']

    c_space.inhibit()
    n_space_u.inhibit()

    if not inhibit_V:
        # restore weights and learning rates between C und V

        for c, w, l in zip(conn_C_V, w_C_V, l_C_V):
            nest.SetStatus([c], {'weight': w, 'lambda': l})

        for c, w, l in zip(conn_V_C, w_V_C, l_V_C):
            nest.SetStatus([c], {'weight': w, 'lambda': l})
    else:
        n_space_v.disinhibit()

    if t_recall_1 > 0:
        nest.Simulate(t_recall_1)

    c_space.disinhibit()
    nest.Simulate(t_recall_2)

    t_recall_2_end = nest.GetKernelStatus()['time']

    # ----------------------------------------------------------------------
    # analysis

    print('analysis')

    if inhibit_V:
        # make sure neural spaces are silent when they should be
        spikes_nv = nest.GetStatus(n_space_v.spike_rec_E)[0]['events']['times']
        spikes_nu = nest.GetStatus(n_space_u.spike_rec_E)[0]['events']['times']

        delta = 50.  # allow spikes in first 50 ms of each inhibited period

        spikes_v_delay_1 = (spikes_nv > t_store_1_end+delta) & (spikes_nv < t_store_2_start)
        spikes_u_delay_1 = (spikes_nu > t_store_1_end+delta) & (spikes_nu < t_store_2_start)
        spikes_v_delay_2 = (spikes_nv > t_store_2_end+delta) & (spikes_nv < t_recall_1_start)
        spikes_u_delay_2 = (spikes_nu > t_store_2_end+delta) & (spikes_nu < t_recall_1_start)

        nspikes_v_d1 = sum(spikes_v_delay_1)
        nspikes_u_d1 = sum(spikes_u_delay_1)
        nspikes_v_d2 = sum(spikes_v_delay_2)
        nspikes_u_d2 = sum(spikes_u_delay_2)

        if nspikes_v_d1 > 0:
            print('WARNING: spikes in V during delay period 1 (count: {0:d})'.format(nspikes_v_d1))
        if nspikes_u_d1 > 0:
            print('WARNING: spikes in U during delay period 1 (count: {0:d})'.format(nspikes_u_d1))
        if nspikes_v_d2 > 0:
            print('WARNING: spikes in V during delay period 2 (count: {0:d})'.format(nspikes_v_d2))
        if nspikes_u_d2 > 0:
            print('WARNING: spikes in U during delay period 2 (count: {0:d})'.format(nspikes_u_d2))

        if nspikes_v_d1 == 0 and nspikes_u_d1 == 0 and \
            nspikes_v_d2 == 0 and nspikes_u_d2 == 0:
            print('neural inhibition ok.')

    # readout neuron(s)

    if n_readout == 1:
        ro_vm_events = nest.GetStatus(ro_vm)[0]['events']

        ro_t = ro_vm_events['times']
        ro_v = ro_vm_events['V_m']

    else:
        ro_sd_events = nest.GetStatus(ro_sd)[0]['events']
        ro_times = ro_sd_events['times']
        ro_senders = ro_sd_events['senders']

        ro_t, ro_spikes_filt = lp_filter_spike_train(
                ro_times,
                ro_senders,
                s_lim=[min(ro_neurons), max(ro_neurons)],
                t_lim=[t_recall_1_start, t_recall_2_end],
                tau_filter=_tau_filter,
                len_filter=_len_filter)

        ro_v = ro_spikes_filt.sum(axis=1)

    results = {
        't': ro_t,
        'v': ro_v,
        'n_readout': n_readout,
        't_burn_in': _t_burn_in,
        'tau_filter': _tau_filter,
        'len_filter': _len_filter,
        'k_pattern_1': k_pattern_1,
        'k_pattern_2': k_pattern_2,
        'same': k_pattern_1==k_pattern_2,
    }

    # ----------------------------------------------------------------------
    # plots

    if plot:
        # fancy plots

        num = 200
        targets_x = np.arange(num) + min(inpop.pop_X)
        np.random.shuffle(targets_x)
        targets_c = np.arange(num) + min(c_space.pop_E)
        targets_v = np.arange(num) + min(n_space_v.pop_E)
        targets_u = np.arange(num) + min(n_space_u.pop_E)

        xlim = [t_store_1_start-50., t_recall_2_end]
        markers = [t_store_1_start, t_store_1_end, t_store_2_start, t_store_2_end, t_recall_1_start, t_recall_2_start]
        labels = ['LOAD 1', 'DELAY', 'LOAD 2', 'DELAY', 'RECALL 1', 'RECALL 2']
        xticks_labeled = [x for x in zip(markers, labels)]
        xticks_unlabeled = []

        plot_spikes_fancy(inpop.spike_rec, targets=targets_x, xlabel='time', ylabel='$\mathcal{X}$', xlim=xlim, markers=markers, xticks=xticks_labeled, wide=True, save=join(outdir, 'overview_X.pdf'))
        plot_spikes_fancy(c_space.spike_rec_E, targets=targets_c, xlabel='', ylabel='$\mathcal{C}$', xlim=xlim, markers=markers, xticks=xticks_unlabeled, wide=True, save=join(outdir, 'overview_C.pdf'))
        plot_spikes_fancy(n_space_v.spike_rec_E, targets=targets_v, xlabel='', ylabel='$\mathcal{N}_V$', xlim=xlim, markers=markers, xticks=xticks_unlabeled, wide=True, save=join(outdir, 'overview_V.pdf'))
        plot_spikes_fancy(n_space_u.spike_rec_E, targets=targets_u, xlabel='', ylabel='$\mathcal{N}_U$', xlim=xlim, markers=markers, xticks=xticks_unlabeled, wide=True, save=join(outdir, 'overview_U.pdf'))

        # readout neuron

        plt.figure()

        plt.plot(ro_t, ro_v)
        plt.xlabel('$t$')
        plt.ylabel('$V_m$')
        plt.title('same: {0:s}'.format(str(k_pattern_1==k_pattern_2)))

        plt.savefig(join(outdir, 'readout_n{0:d}.pdf'.format(n_readout)))

        if show:
            plt.show()
        else:
            plt.close('all')

    return results


def frankland_greene(outdir, swtafile, variant, config_c, config_v, recall_cfg, experiment=1, print_results=True, use_data='all', plot=True, show=True):

    assert experiment in [1, 2]

    dump_dict(config_v, key='config_v', dumpfile=join(outdir, 'config_v.json'))
    dump_dict(config_c, key='config_c', dumpfile=join(outdir, 'config_c.json'))
    dump_dict(config_c, key='config_recall', dumpfile=join(outdir, 'config_recall.json'))

    # unpack input

    inhibit_V = recall_cfg['inhibit_V']
    t_store = recall_cfg['t_store']
    t_recall_1 = recall_cfg['t_recall_1']
    t_recall_2 = recall_cfg['t_recall_2']

    noise_var = {'all': 4, 'mean': 4, 'final': 4, 'mean_mean': .0, 'mean_sum': 5}[use_data]


    # setup

    r = setup_variable_binding(outdir, swtafile, variant,
        config_c=config_c, config_v=config_v, num_variable_spaces=2)

    inpop, c_space, n_spaces, t_test_start, t_test_end, t_create_start, t_create_end, C_readout_test = r

    assert len(n_spaces) == 2
    assert len(t_create_start) == 2
    assert len(t_create_end) == 2

    n_space_v = n_spaces[0]
    n_space_u = n_spaces[1]
    t_create_start_V = t_create_start[0]
    t_create_start_U = t_create_start[1]
    t_create_end_V = t_create_end[0]
    t_create_end_U = t_create_end[1]

    # assert E_sfa clipping is used
    assert nest.GetStatus(n_space_v.pop_E)[0]['E_sfa_clip'] == True
    assert nest.GetStatus(n_space_v.pop_E)[0]['E_sfa_max'] < 0.
    assert nest.GetStatus(n_space_u.pop_E)[0]['E_sfa_clip'] == True
    assert nest.GetStatus(n_space_u.pop_E)[0]['E_sfa_max'] < 0.

    # clear network activity: inhibit V und U spaces, give noise input to C

    def pause(t):
        n_space_v.inhibit()
        n_space_u.inhibit()

        inpop.clear()

        if t > 0:
            nest.Simulate(float(t))

    pause(1000)


    if experiment == 1:

        results = {
                'experiment': 1,
                'swtafile': swtafile,
        }

        # sentences:
        # S1: 0->V, 1->U
        # S2: 1->U, 0->V
        # S3: 1->V, 0->U
        # S4: 0->U, 1->V

        k_pattern_1 = 0
        k_pattern_2 = 1

        # define presentation procedure for one sentence

        t_present = 200.
        t_pause_in_sentence = 0.
        t_pause_between_sentences = 1000.

        def present(space_1, pattern_1, space_2, pattern_2, name):
            print(name)

            c_space.reset_sfa()
            n_space_v.reset_sfa()
            n_space_u.reset_sfa()

            # presentation 1

            space_1.disinhibit()
            inpop.set(pattern_1)

            t0 = nest.GetKernelStatus()['time']

            nest.Simulate(t_present)

            t1 = nest.GetKernelStatus()['time']

            pause(t_pause_in_sentence)

            # presentation 2

            space_2.disinhibit()
            inpop.set(pattern_2)

            t2 = nest.GetKernelStatus()['time']

            nest.Simulate(t_present)

            t3 = nest.GetKernelStatus()['time']

            pause(t_pause_between_sentences)

            return [t0, t1], [t2, t3]


        # training sentences

        t_train_start = nest.GetKernelStatus()['time']
        t_train_ = []

        t_train_ += present(n_space_v, k_pattern_1, n_space_u, k_pattern_2, 'S1/4 (train)')
        t_train_ += present(n_space_u, k_pattern_2, n_space_v, k_pattern_1, 'S2/4 (train)')
        t_train_ += present(n_space_v, k_pattern_2, n_space_u, k_pattern_1, 'S3/4 (train)')
        t_train_ += present(n_space_u, k_pattern_1, n_space_v, k_pattern_2, 'S4/4 (train)')

        t_train_end = nest.GetKernelStatus()['time']

        assert len(t_train_) == 8


        # test sentences

        t_test_start = nest.GetKernelStatus()['time']
        t_test_ = []

        t_test_ += present(n_space_v, k_pattern_1, n_space_u, k_pattern_2, 'S1/4 (test)')
        t_test_ += present(n_space_u, k_pattern_2, n_space_v, k_pattern_1, 'S2/4 (test)')
        t_test_ += present(n_space_v, k_pattern_2, n_space_u, k_pattern_1, 'S3/4 (test)')
        t_test_ += present(n_space_u, k_pattern_1, n_space_v, k_pattern_2, 'S4/4 (test)')

        t_test_end = nest.GetKernelStatus()['time']

        assert len(t_test_) == 8


        # save settings

        results['t_present'] = t_present
        results['t_pause_in_sentence'] = t_pause_in_sentence
        results['t_pause_between_sentences'] = t_pause_between_sentences
        results['t_burn_in'] = _t_burn_in
        results['tau_filter'] = _tau_filter
        results['len_filter'] = _len_filter


        # save spike trains

        print('analysis')

        t_lim = [t_train_start, t_test_end]
        spikes_C = c_space.get_spikes(t_lim=t_lim)[0]
        spikes_V = n_space_v.get_spikes(t_lim=t_lim)[0]
        spikes_U = n_space_u.get_spikes(t_lim=t_lim)[0]

        data = dict(t_train_start=t_train_start, t_train_end=t_train_end,
                t_train_=t_train_, t_test_start=t_test_start,
                t_test_end=t_test_end, t_test_=t_test_,
                spikes_X=inpop.get_spikes(t_lim=t_lim), spikes_C=spikes_C,
                spikes_V=spikes_V, spikes_U=spikes_U, inpop_pop_X=inpop.pop_X,
                C_pop_E=c_space.pop_E, V_pop_E=n_space_v.pop_E,
                U_pop_E=n_space_u.pop_E, results=results)

        with open(join(outdir, 'spike_data.pkl'), 'wb') as f:
            pkl.dump(data, f)


        # filter spike trains

        t_all_ = t_train_ + t_test_

        args = dict(t_burn_in=_t_burn_in, tau_filter=_tau_filter,
                len_filter=_len_filter, use_data=use_data, noise_var=noise_var)

        t_, trace_C = spikes_to_traces(spikes_C, c_space.pop_E, t_all_, **args)
        t_, trace_V = spikes_to_traces(spikes_V, n_space_v.pop_E, t_all_, **args)
        t_, trace_U = spikes_to_traces(spikes_U, n_space_u.pop_E, t_all_, **args)

        print('trace_C  mean: {0:.2g}  var: {1:.2g}'.format(trace_C.mean(), trace_C.var()))
        print('trace_V  mean: {0:.2g}  var: {1:.2g}'.format(trace_V.mean(), trace_V.var()))
        print('trace_U  mean: {0:.2g}  var: {1:.2g}'.format(trace_U.mean(), trace_U.var()))

        # merge V und U traces
        trace_UV = np.hstack((trace_V, trace_U))

        # split training and test

        L = int(trace_C.shape[0] / 2)  # first dimension is time

        assert trace_V.shape[0] == 2*L
        assert trace_U.shape[0] == 2*L
        assert trace_UV.shape[0] == 2*L

        x_C, xt_C = trace_C[L:,:], trace_C[:L,:]
        x_UV, xt_UV = trace_UV[L:,:], trace_UV[:L,:]

        # create labels

        y = np.zeros(L)
        y[int(L/2):] = -2
        y[:int(L/2)] = 2

        yt = y

        # train readout

        _, error_C = train_readout(x_C, y, xt_C, yt)
        _, error_UV = train_readout(x_UV, y, xt_UV, yt)

        results['readout'] = {'error_C': error_C, 'error_UV': error_UV}

        print('error C:', error_C)
        print('error UV:', error_UV)

        # plot

        if plot:
            num = 200
            targets_x = np.arange(num) + min(inpop.pop_X)
            np.random.shuffle(targets_x)
            targets_c = np.arange(num) + min(c_space.pop_E)
            targets_v = np.arange(num) + min(n_space_v.pop_E)
            targets_u = np.arange(num) + min(n_space_u.pop_E)

            xlim = [t_train_start-50., t_test_end]
            markers = [t_train_start, t_test_start]
            labels = ['TRAIN', 'TEST']
            xticks_labeled = [x for x in zip(markers, labels)]
            xticks_unlabeled = []

            plot_spikes_fancy(inpop.spike_rec, targets=targets_x, xlabel='time', ylabel='$\mathcal{X}$', xlim=xlim, markers=markers, xticks=xticks_labeled, wide=True, save=join(outdir, 'overview_X.pdf'))
            plot_spikes_fancy(c_space.spike_rec_E, targets=targets_c, xlabel='', ylabel='$\mathcal{C}$', xlim=xlim, markers=markers, xticks=xticks_unlabeled, wide=True, save=join(outdir, 'overview_C.pdf'))
            plot_spikes_fancy(n_space_v.spike_rec_E, targets=targets_v, xlabel='', ylabel='$\mathcal{N}_V$', xlim=xlim, markers=markers, xticks=xticks_unlabeled, wide=True, save=join(outdir, 'overview_V.pdf'))
            plot_spikes_fancy(n_space_u.spike_rec_E, targets=targets_u, xlabel='', ylabel='$\mathcal{N}_U$', xlim=xlim, markers=markers, xticks=xticks_unlabeled, wide=True, save=join(outdir, 'overview_U.pdf'))

            if show:
                plt.show()

    elif experiment == 2:

        results = {
                'experiment': 2,
                'swtafile': swtafile,
        }

        # consider 40 sentences: five items, of which two can be presented to
        # two different neural spaces; furthermore, the order of presentation
        # can be swapped; thus, we have 40 combinations

        t_present = 200.
        t_pause_in_sentence = 0.
        t_pause_between_sentences = 1000.

        def present(space_1, pattern_1, space_2, pattern_2, name):
            print(name)

            c_space.reset_sfa()
            n_space_v.reset_sfa()
            n_space_u.reset_sfa()

            # presentation 1

            space_1.disinhibit()
            inpop.set(pattern_1)

            t0 = nest.GetKernelStatus()['time']

            nest.Simulate(t_present)

            t1 = nest.GetKernelStatus()['time']

            pause(t_pause_in_sentence)

            # presentation 2

            space_2.disinhibit()
            inpop.set(pattern_2)

            t2 = nest.GetKernelStatus()['time']

            nest.Simulate(t_present)

            t3 = nest.GetKernelStatus()['time']

            pause(t_pause_between_sentences)

            return [t0, t1], [t2, t3]

        data_V_ = []  # presentation times and labels for V
        data_U_ = []  # presentation times and labels for U

        t_start =  nest.GetKernelStatus()['time']

        n = 0  # sentence number

        for k_pattern_1 in range(5):
            for k_pattern_2 in range(5):
                if k_pattern_1 == k_pattern_2:
                    continue

                # present k_pattern_1 to V first, then k_pattern_2 to U

                n += 1

                times = present(n_space_v, k_pattern_1, n_space_u, k_pattern_2, 'S{0:d}/40'.format(n))

                data_V_ += [[*times[0], k_pattern_1]]
                data_U_ += [[*times[1], k_pattern_2]]

                # now, present in alternative order: k_pattern_2 to U first,
                # then k_pattern_1 to V

                n += 1

                times = present(n_space_u, k_pattern_2, n_space_v, k_pattern_1, 'S{0:d}/40'.format(n))

                data_U_ += [[*times[0], k_pattern_2]]
                data_V_ += [[*times[1], k_pattern_1]]

        t_end =  nest.GetKernelStatus()['time']


        # save settings

        results['t_present'] = t_present
        results['t_pause_in_sentence'] = t_pause_in_sentence
        results['t_pause_between_sentences'] = t_pause_between_sentences
        results['t_burn_in'] = _t_burn_in
        results['tau_filter'] = _tau_filter
        results['len_filter'] = _len_filter


        # save spike trains

        print('analysis')

        t_lim = [t_start, t_end]
        spikes_C = c_space.get_spikes(t_lim=t_lim)[0]
        spikes_V = n_space_v.get_spikes(t_lim=t_lim)[0]
        spikes_U = n_space_u.get_spikes(t_lim=t_lim)[0]

        data = dict(t_start=t_start, t_end=t_end, data_V_=data_V_,
                data_U_=data_U_, spikes_X=inpop.get_spikes(t_lim=t_lim),
                spikes_C=spikes_C, spikes_V=spikes_V, spikes_U=spikes_U,
                I_pop_X=inpop.pop_X, C_pop_E=c_space.pop_E,
                V_pop_E=n_space_v.pop_E, U_pop_E=n_space_u.pop_E, results=results)

        with open(join(outdir, 'spike_data.pkl'), 'wb') as f:
            pkl.dump(data, f)


        # filter spike trains

        # shuffle data
        np.random.shuffle(data_V_)
        np.random.shuffle(data_U_)

        args = dict(t_burn_in=_t_burn_in, tau_filter=_tau_filter,
                len_filter=_len_filter, use_data=use_data, noise_var=noise_var)

        t_V, trace_V, labels_V = spikes_to_traces(spikes_V, n_space_v.pop_E, data_V_, **args)
        t_U, trace_U, labels_U = spikes_to_traces(spikes_U, n_space_u.pop_E, data_U_, **args)

        print('trace_V  mean: {0:.2g}  var: {1:.2g}'.format(trace_V.mean(), trace_V.var()))
        print('trace_U  mean: {0:.2g}  var: {1:.2g}'.format(trace_U.mean(), trace_U.var()))

        # create data matrices for fitting

        L_V = int(trace_V.shape[0] / 2)  # first dimension is time
        L_U = int(trace_U.shape[0] / 2)  # first dimension is time

        assert trace_V.shape[0] == 2*L_V
        assert trace_U.shape[0] == 2*L_U

        x_V, xt_V = trace_V[L_V:,:], trace_V[:L_V,:]
        y_V, yt_V = labels_V[L_V:], labels_V[:L_V]

        x_U, xt_U = trace_U[L_U:,:], trace_U[:L_U,:]
        y_U, yt_U = labels_U[L_U:], labels_U[:L_U]

        # train readout

        _, error_V = train_readout(x_V, y_V, xt_V, yt_V)
        _, error_U = train_readout(x_U, y_U, xt_U, yt_U)

        results['readout'] = {'error_V': error_V, 'error_U': error_U}

        print('error V:', error_V)
        print('error U:', error_U)


        # plot

        if plot:
            num = 200
            targets_x = np.arange(num) + min(inpop.pop_X)
            np.random.shuffle(targets_x)
            targets_c = np.arange(num) + min(c_space.pop_E)
            targets_v = np.arange(num) + min(n_space_v.pop_E)
            targets_u = np.arange(num) + min(n_space_u.pop_E)

            xlim = [t_start-50., t_end]
            markers = []
            labels = []
            xticks_labeled = [x for x in zip(markers, labels)]
            xticks_unlabeled = []

            plot_spikes_fancy(inpop.spike_rec, targets=targets_x, xlabel='time', ylabel='$\mathcal{X}$', xlim=xlim, markers=markers, xticks=xticks_labeled, wide=True, save=join(outdir, 'overview_X.pdf'))
            plot_spikes_fancy(c_space.spike_rec_E, targets=targets_c, xlabel='', ylabel='$\mathcal{C}$', xlim=xlim, markers=markers, xticks=xticks_unlabeled, wide=True, save=join(outdir, 'overview_C.pdf'))
            plot_spikes_fancy(n_space_v.spike_rec_E, targets=targets_v, xlabel='', ylabel='$\mathcal{N}_V$', xlim=xlim, markers=markers, xticks=xticks_unlabeled, wide=True, save=join(outdir, 'overview_V.pdf'))
            plot_spikes_fancy(n_space_u.spike_rec_E, targets=targets_u, xlabel='', ylabel='$\mathcal{N}_U$', xlim=xlim, markers=markers, xticks=xticks_unlabeled, wide=True, save=join(outdir, 'overview_U.pdf'))

            if show:
                plt.show()

    return results
