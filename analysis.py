#!/usr/bin/env python2

import matplotlib.pyplot as plt
import multiprocessing as mp
import nest
from nest import raster_plot
import numpy as np
from sklearn.linear_model import LogisticRegression


from utils import inset_text, make_room, multi_logical_or


def get_weights_from_connections(connections):
    weights = np.zeros(len(connections))

    for k, conn in enumerate(connections):
        weights[k] = nest.GetStatus([conn])[0]['weight']

    return weights


def get_weights_from_population(pop_in, pop_target, num_target_neurons=None):
    """
Create a histogram of all inbound connections from a number of neurons picked
randomly from a population.
    """

    if num_target_neurons is not None:
        mask = np.random.choice(len(pop_target), size=(num_target_neurons))
        pop_target = pop_target[mask]

    connections = nest.GetConnections(pop_in, pop_target)

    return get_weights_from_connections(connections)


def get_spikes_from_spike_recorder(spike_recorder, population, t_lim=None, count=None, random=False):
    """
    Extract spikes from spike recorder and perform some optional processing.

    args:
        spike_recorder      NEST spike recorder
        population          targets of spike recorder, can be None if count
                            parameter is not used
        t_lim               optional, tuple of start and end time for
                            extracting only spikes within a certain time
                            range (useful for plotting)
        count               optional, number of neurons to extract spikes
                            from
        random              optional, only used if count is not None, bool
                            indicating whether the number of neurons to be
                            extract should be random, default: False

    returns:
        times               extracted spike times
        senders             extracted spike sender ids
    """
    events = nest.GetStatus(spike_recorder)[0]['events']
    times = events['times']
    senders = events['senders']

    if count is not None:
        ids = np.arange(min(population), max(population)+1)

        if not random:
            id_selection = ids[:count]
        else:
            id_selection = np.random.choice(ids, count, replace=False)

        mask = np.isin(senders, id_selection)

        times = times[mask]
        senders = senders[mask]

    if t_lim is not None:
        times, senders = filter_spikes(times, senders, t_lim=t_lim)

    return times, senders


def filter_spikes(times, senders, *, targets=None, t_lim=None):
    """
    Filter spikes extracted from a spike recorder.

    args:
        times               spike times
        senders             spike sender ids

    keyword-only args:
        targets             if set, extract only neurons which have an id given
                            in this list
        t_lim               optional, tuple of start and end time for
                            extracting only spikes within a certain time range
                            (useful for plotting)

    returns:
        times               filtered spike times
        senders             filtered spike sender ids

    """
    if targets is not None:
        mask = np.isin(senders, targets)

        times = times[mask]
        senders = senders[mask]

    if t_lim is not None:
        assert len(t_lim) == 2

        mask = (times > t_lim[0]) & (times <= t_lim[1])

        times = times[mask]
        senders = senders[mask]

    return [times, senders]


def reorder_neurons(times, senders, targets):
    """
    Reorder neurons extracted from spike recorder according to given list of
    target ids.

    args:
        times               spike times
        senders             spike sender ids
        targets             list of ids, should contain the same elements as
                            the current neuron ids given in senders; the ids
                            will re-assigned in ascending order according to
                            the order in this list

    returns:
        senders             re-arranged senders
    """
    assert min(senders) > len(targets)

    d = {}
    for k, t in enumerate(targets):
        d[t] = k

    return np.vectorize(d.__getitem__)(senders)


def spike_analysis(spikes_E, pop_E):
    spikes_per_neuron = np.bincount(spikes_E)[min(pop_E):max(pop_E)+1]

    return spikes_per_neuron


def assembly_analysis(space, test_start_times, test_end_times, assembly_rate_threshold=50, print_results=False):
    assert len(test_start_times) == len(test_end_times)
    N_pattern = len(test_start_times)

    spikes_E, _ = space.get_spikes_legacy()
    pop_E = space.pop_E

    spikes_per_pattern = []
    firing_rates_per_pattern = []
    assemblies = []

    for k in range(N_pattern):
        start_time = test_start_times[k]
        end_time = test_end_times[k]

        dt = end_time - start_time
        start_time += dt / 2.  # allow burn-in

        mask = np.logical_and(start_time <= spikes_E['times'], spikes_E['times'] <= end_time)
        pattern_spikes = spikes_E['senders'][mask]

        hist = np.bincount(pattern_spikes, minlength=max(pop_E)+1)
        firing_rates = np.asarray(hist, dtype=np.float) / (end_time - start_time) * 1e3

        mask = firing_rates > assembly_rate_threshold

        # prepend pop_E with zeros since neuron ids may start at values > 0
        pop_E_padded = np.hstack((np.arange(min(pop_E)),np.asarray(pop_E)))
        assembly = pop_E_padded[mask] if mask.sum() > 0 else []

        spikes_per_pattern += [pattern_spikes]
        firing_rates_per_pattern += [firing_rates]
        assemblies += [np.asarray(assembly, dtype=np.int)]

    if print_results:
        print('assemblies:')

    assembly_counts = np.bincount(np.hstack(assemblies))

    for k in range(N_pattern):
        assembly = assemblies[k]

        unique = 0
        for a in assembly:
            if assembly_counts[a] == 1:
                unique += 1

        if print_results:
            print('assembly {}:'.format(k+1), end='')
            print(assembly, end='')
            print(' unique: {}/{}'.format(unique, len(assembly)))

    assembly_occurances = []
    for k in range(N_pattern):
        assembly_occurances += [int((assembly_counts == k+1).sum())]

    if print_results:
        print('occurances in assemblies:', assembly_occurances)

    return spikes_per_pattern, firing_rates_per_pattern, assemblies, assembly_occurances


def multi_assembly_analysis(arg):
    """
    Argument is a list of 5-tuples, where each contains
      - the key of the result in the return dictionary
      - the neural space
      - the list of start times
      - the list of end times
      - a bool: print results or not
    """

    q = mp.Queue()
    proc = []

    for k in range(len(arg)):
        f = lambda k: q.put({arg[k][0]: assembly_analysis(
            arg[k][1],
            arg[k][2],
            arg[k][3],
            print_results=arg[k][4])[2]})  # note: stores only assemblies

        p = mp.Process(target=f, args=(k,))
        proc += [p]

    [p.start() for p in proc]

    r = {}
    for k in range(len(arg)):
        r.update(q.get())

    [p.join() for p in proc]

    return r


def assembly_match(a, b):
    intersect = np.intersect1d(a, b)

    shared = len(intersect)
    only_in_a = len(a) - shared
    only_in_b = len(b) - shared

    return shared, only_in_a, only_in_b


def plot_multi_weight_hist(weights, save=None, close=False):
    N = len(weights)

    assert N > 0
    assert N <= 4

    if N == 1:
        figsize = (6., 4.)
        subrow = 1
        subcol = 1
    if N == 2 or N == 3:
        figsize = (12., 4.)
        subrow = 1
        subcol = N
    if N == 4:
        figsize = (12., 8.)
        subrow = 2
        subcol = 2

    plt.figure(figsize=figsize)

    for k, t in enumerate(weights.keys()):
        d = weights[t].flatten()
        t = t.replace('_', ' ')

        plt.subplot(subrow, subcol, k+1)

        if len(d) > 0:
            plt.hist(d, bins=50, color='C0', edgecolor='C0')
        else:
            plt.plot([0], [0])

        plt.xlabel('w')
        plt.ylabel('count')
        plt.title(t)
        plt.tight_layout()

    if save is not None:
        plt.savefig(save)

    if close:
        plt.close()


def plot_dual_weight_hist(w1, t1, w2, t2, save=None, close=False):
    weights = {t1: w1, t2: w2}
    plot_multi_weight_hist(weights, save=save, close=close)


def plot_spikes_fancy(spike_rec, targets=None, title='', size=5., xlabel='time', ylabel='neuron', xlim=None, markers=[], marker_color_div=1, xticks=None, yticks=False, wide=False, height_scale=1., hide_spines_left=False, hide_spines_right=False, save=None, close=False):
    """
    args:
        spike_rec           spike recorder
        targets             None or a list of neurons to filter and solely
                            display
        title               title of plot
        size                size of spike marker, default: 20
        xlim                None or tuple (start_time, end_time) of plot
        markers             list of times where colored vertical lines will be
                            placed
        marker_color_div    number of consecutive markers sharing the same
                            color, default: 1
        xticks              None or list of tuples (time, string), then,
                            instead of normal xticks, the strings are placed
                            as ticks at the given times
        yticks              bool, whether to show yticks for neuron ids
        wide                make double width plot, default: False
        height_scale        scale height, default: 1
        hide_spines_left    hide plot spine on left for a broken-axis plot
        hide_spines_right   hide plot spine on right for a broken-axis plot
        save                None or file name for saving
        close               whether to close the figure after drawing it
    """
    ev = nest.GetStatus(spike_rec)[0]['events']
    times = ev['times']
    ids = ev['senders']

    if targets is not None:
        mask = np.in1d(ids, targets)

        times = times[mask]
        ids = ids[mask]

        # change ids
        ids_old = ids.copy()
        for k, target in enumerate(targets):
            ids[ids_old == target] = k

    width = 12. if wide else 6.
    height = 4. * height_scale
    plt.figure(figsize=(width, height))

    plt.scatter(times, ids, s=size, marker=2, color='#777777')

    for k, m in enumerate(markers):
        color = 'C{0:d}'.format(k // marker_color_div)
        plt.axvline(x=m, c=color, lw=1., linestyle='dashed')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if xticks is not None:
        plt.gca().set_xticks([x[0] for x in xticks])
        plt.gca().set_xticklabels([x[1] for x in xticks])
    else:
        plt.locator_params(axis='x', nbins=3)

    if not yticks:
        plt.yticks([])
    else:
        plt.locator_params(axis='y', nbins=3)

    if xlim is not None:
        assert len(xlim) == 2
        plt.xlim(*xlim)

    if hide_spines_left:
        plt.gca().spines['left'].set_visible(False)

    if hide_spines_right:
        plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()

    if save is not None:
        plt.savefig(save)

    if close:
        plt.close()


def plot_spikes(spike_rec, t, xlim=None, save=None, close=False):
    nest.raster_plot.from_device(spike_rec, hist=False, title=t)

    ax = plt.gca()
    ax.locator_params(nbins=4)
    ax.set_xlim(ax.dataLim.min[0], ax.dataLim.max[0])
    ax.set_ylim(ax.dataLim.min[1], ax.dataLim.max[1])

    fig = plt.gcf()
    fig.set_size_inches(6., 4., forward=True)

    plt.xlabel('$t$ / ms')
    plt.ylabel('neuron id')

    if xlim is not None:
        assert len(xlim) == 2
        plt.xlim(xlim[0], xlim[1])

    plt.tight_layout()

    if save is not None:
        plt.savefig(save)

    if close:
        plt.close()


def plot_training_activity(spike_rec, N_show_pattern, t_training_period, div=50, plot_len=5, targets=None, save=None, close=False):

    start_ind = [k*plot_len for k in range(10)] + [int(N_show_pattern*f) for f in [0.25, 0.5, 0.75]] +[N_show_pattern-plot_len]

    for si in start_ind:
        if si >= N_show_pattern:
            continue

        start_time = si * t_training_period
        end_time = start_time + plot_len*t_training_period

        plot_spikes_fancy(
            spike_rec,
            targets=targets,
            title='E-pool after {} training patterns'.format(si),
            xlabel='$t$ / ms',
            xlim=[start_time, end_time],
            save=save.format(si),
            close=close)


def plot_hist(d, t, bins=100, xlabel=None, ylabel='count', save=None, close=False):
    plt.figure(figsize=(6., 4.))

    if len(d) > 0:
        plt.hist(d, bins=bins)
    else:
        plt.plot([0], [0])

    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)

    plt.title(t)
    plt.tight_layout()

    if save is not None:
        plt.savefig(save)

    if close:
        plt.close()


def plot_pattern_responses(spikes, rates, swta, save=None, close=False):

    assert len(spikes) == len(rates)
    N_pattern = len(spikes)

    plt.figure()
    plt.suptitle('spikes per pattern')
    ax = None
    for k in range(N_pattern):
        plt.subplot(N_pattern, 1, k+1, sharex=ax)
        if ax is None:
            ax = plt.gca()
        plt.hist(spikes[k], bins=swta.pop_E)
        plt.tight_layout()

    if save is not None:
        plt.savefig(save[0])

    if close:
        plt.close()

    plt.figure()
    plt.suptitle('neuron firing rates per pattern')
    ax = None
    for k in range(N_pattern):
        plt.subplot(N_pattern, 1, k+1, sharex=ax)
        if ax is None:
            ax = plt.gca()
        plt.stem(np.arange(1, swta.config['N_E']+1), rates[k])
        plt.tight_layout()

    if save is not None:
        plt.savefig(save[1])

    if close:
        plt.close()


def plot_synd_weight_correlation(pop_in, pop_out, save=None, close=False):
    conn = nest.GetConnections(pop_in, pop_out)
    status = nest.GetStatus(conn)

    delays = np.zeros(len(status))
    weights = np.zeros(len(status))

    for k, s in enumerate(status):
        delays[k] = s['delay']
        weights[k] = s['weight']

    corr = correlation(delays, weights)

    plt.figure(figsize=(6., 4.))
    plt.scatter(delays, weights, marker='.', edgecolor='none')
    plt.xlabel(r'$\Delta_\mathrm{syn}$')
    plt.ylabel(r'$w$')
    plt.title('correlation: {0:.2f}'.format(corr))
    ax = plt.gca()
    ax.set_xlim(ax.dataLim.min[0], ax.dataLim.max[0])
    ax.set_ylim(ax.dataLim.min[1], ax.dataLim.max[1])
    plt.locator_params(nbins=3)
    plt.tight_layout()

    if save is not None:
        plt.savefig(save)

    if close:
        plt.close()


def plot_assembly_input_hist(a_target, a_source, t="", save=None, close=False):
    count = []

    for n in a_target:
        conn = nest.GetConnections(list(a_source), [n])
        count += [len(conn)]


    plt.figure(figsize=(6., 4.))

    if len(count) > 0:
        plt.hist(count, bins=50, color='C0', edgecolor='C0')
    else:
        plt.plot([0], [0])

    plt.xlabel('number of inputs')
    plt.ylabel('count')
    plt.title(t)
    plt.tight_layout()

    if save is not None:
        plt.savefig(save)

    if close:
        plt.close()


def analyze_assembly_weights(w_, a_pre_, a_post_=None):
    # to analyze a neural space, don't pass a_post_
    # to analyze a projection, pass a_pre_ from one space and a_post_ and w_ from another

    w_ = np.asarray(w_)
    num_assembly = len(a_pre_)

    if a_post_ is None:
        a_post_ = a_pre_

    assert len(a_pre_) == len(a_post_)

    # within assemblies
    masks_pre_within_a_ = [np.isin(w_[:,0], a) for a in a_pre_]
    masks_post_within_a_ = [np.isin(w_[:,1], a) for a in a_post_]
    masks_within_a_ = [(m_pre & m_post) for m_pre, m_post in zip(masks_pre_within_a_, masks_post_within_a_)]

    mask_within_a = multi_logical_or(masks_within_a_)
    w_within_a = w_[mask_within_a,-1]

    # between assemblies
    masks_between_a_ = []
    for k in range(num_assembly):
        mask_post_within_other = multi_logical_or(masks_post_within_a_[:k]+masks_post_within_a_[k+1:])
        masks_between_a_ += [(masks_pre_within_a_[k] & mask_post_within_other)]

    mask_between_a = multi_logical_or(masks_between_a_)
    w_between_a = w_[mask_between_a,-1]

    # connections between assembly neurons and neurons outside assemblies
    a_pre_flat = np.concatenate(a_pre_)
    a_post_flat = np.concatenate(a_post_)
    mask_pre_noa = ~np.isin(w_[:,0], a_pre_flat)
    mask_post_noa = ~np.isin(w_[:,1], a_post_flat)
    mask_all_noa = (mask_pre_noa | mask_post_noa)

    mask_within_noa = (mask_pre_noa & mask_post_noa)
    mask_between_a_and_noa = (mask_all_noa & ~mask_within_noa)
    assert ((mask_within_noa | mask_between_a_and_noa) == mask_all_noa).all()

    w_between_a_and_noa = w_[mask_between_a_and_noa,-1]

    # connections between neurons outside assemblies

    w_within_noa = w_[mask_within_noa,-1]

    return w_within_a, w_between_a, w_between_a_and_noa, w_within_noa

def plot_assembly_correlations(r_, name, save=None, close=False):

    plt.figure(name, figsize=(9, 6))

    t_ = ['within assemblies', 'between assemblies', 'between assemblies and unassigned', 'between unassigned']

    for k, (t, r) in enumerate(zip(t_, r_)):
        s = '${0:.3f} \pm {1:.3f}$  (range: {2:.3f} -- {3:.3f},  median {4:.3f})'.format(
                r.mean(), r.std(), r.min(), r.max(), np.median(r))

        plt.subplot(2, 2, k+1)
        plt.title(t, fontsize=12)
        plt.hist(r, bins=20)
        plt.xlabel('weight')
        plt.ylabel('count')
        make_room(.2)
        inset_text(s, 'top', 'center', fontsize=8)

    plt.tight_layout()

    if save is not None:
        plt.savefig(save)

    if close:
        plt.close()


def correlation(x, y, force_positive=False):
    """
Compute the correlation coefficient between two signals.

args:
    x:                  input vector 1
    y:                  input vector 2
    force_positive:     toggle sign to make return value positive, if neccessary

returns:
    r:                  correlation coefficient of x and y
    """

    # force vectors
    assert len(x.shape) <= 2
    assert len(x.shape) == 1 or x.shape[0] == 1 or x.shape[1] == 1
    assert len(y.shape) <= 2
    assert len(y.shape) == 1 or y.shape[0] == 1 or y.shape[1] == 1

    # strip unneccessary dimensions
    if len(x.shape) > 1:
        if x.shape[0] != 1:
            x = x.T
        x = x[0,:]

    if len(y.shape) > 1:
        if y.shape[0] != 1:
            y = y.T
        y = y[0,:]

    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape == y.shape

    x = x - x.mean()
    y = y - y.mean()

    norm = np.sqrt((x**2).mean() * (y**2).mean())

    r = (x * y).mean() / norm

    if force_positive:
        r = abs(r)

    return r


def create_spike_time_grid(times, senders, s_lim, t_lim, dt, drop_silent=False):
    """
    Create a time grid from spike times.

    args:
        times               list of spike times
        senders             list of spike sender ids
        t_lim               tuple containing start and end time of grid
        dt                  discrete time step of output grid
        drop_silent         don't create a trace for senders without spikes,
                            default: False
    """

    times = np.asarray(times)
    senders = np.asarray(senders)

    assert s_lim[1] > s_lim[0], 's_lim not ordered properly'
    assert t_lim[1] > t_lim[0], 't_lim not ordered properly'

    # drop spikes outside of range
    mask = (times >= t_lim[0]) & (times < t_lim[1])
    times = times[mask]
    senders = senders[mask]

    # drop senders outside of range
    mask = (senders >= s_lim[0]) & (senders < s_lim[1])
    senders = senders[mask]
    times = times[mask]

    assert len(times) == 0 or min(times) >= t_lim[0]
    assert len(times) == 0 or max(times) < t_lim[1]
    assert len(senders) == 0 or min(senders) >= s_lim[0]
    assert len(senders) == 0 or max(senders) < s_lim[1]

    # creating time vector as simple index count and changing to actual time
    # values after assignments are done to avoid numerical problems
    delta_t = t_lim[1] - t_lim[0]
    L = np.round(delta_t / dt).astype(np.int64)
    ti = np.arange(L)

    # create spike grid
    N = s_lim[1] - s_lim[0] + 1
    spike_grid = np.zeros((L, N))

    # create assignment vectors
    times_ind = np.round((times - t_lim[0]) / dt).astype(np.int64)
    senders_ind = (senders - s_lim[0]).astype(np.int64)

    assert len(times_ind) == 0 or max(times_ind) < L
    assert len(senders_ind) == 0 or max(senders_ind) < N

    # assign
    for time_ind, sender_ind in zip(times_ind, senders_ind):
        spike_grid[time_ind,sender_ind] += 1

    # assert that no spike got lost
    senders_in, spike_counts_in = np.unique(senders, return_counts=True)
    spike_counts_out = spike_grid.sum(axis=0)
    for s, c in zip(senders_in, spike_counts_in):
        assert spike_counts_out[s-s_lim[0]] == c

    # remove rows for neurons which never spike if requested
    if drop_silent:
        mask = spike_counts_out > 0
        spike_grid = spike_grid[:,mask]

    # convert time indices to actual time
    t = t_lim[0] + ti * dt

    return t, spike_grid


def lp_filter_spike_train(times, senders, s_lim, t_lim, tau_filter=20., len_filter=100., drop_silent=False):
    """
    Create lowpass-filtered traces from spike events.

    args:
        times           spike times
        senders         spiking neuron ids
        s_lim           tuple containing min and max neuron id for output
        t_lim           tuple containing start and end time
        tau_filter      time constant for exponential filter, default: 20.
        len_filter      length of exponential filter, default: 100.
        drop_silent     drop senders which do not spike during time
    """

    dt = nest.GetKernelStatus()['resolution']
    t, spike_grid = create_spike_time_grid(
            times,
            senders,
            s_lim,
            t_lim,
            dt,
            drop_silent=drop_silent)

    L = spike_grid.shape[0]
    T = L * dt

    # filter with exponential
    f_t = np.arange(0., min(100., T), dt)  # only make as long as necessary
    f = np.exp(-f_t / tau_filter)

    spike_counts = spike_grid.sum(axis=0)
    spikes_filtered = np.zeros(spike_grid.shape)

    for k in range(spike_grid.shape[1]):
        if spike_counts[k] == 0:
            continue

        spikes_filtered[:,k] = np.convolve(spike_grid[:,k], f)[:L]

    return t, spikes_filtered


def spikes_to_traces(spikes, pop, time_label_data, t_burn_in, tau_filter, len_filter, use_data='all', noise_var=0, drop_silent=True):

    assert use_data in ['all', 'mean', 'final', 'mean_mean', 'mean_sum']

    with_labels = (len(time_label_data[0]) == 3)

    t_ = []
    tr_ = []
    if with_labels:
        l_ = []

    print('using burn in time of {0:g}'.format(t_burn_in))

    for tld in time_label_data:
        assert len(tld) == 3 if with_labels else 2

        t0, t1 = tld[0], tld[1]
        t0 += t_burn_in  # let activity can settle

        t, tr = lp_filter_spike_train(*spikes, s_lim=[min(pop), max(pop)],
                t_lim=[t0, t1], tau_filter=tau_filter, len_filter=len_filter)

        t_ += [t]
        tr_ += [tr]

        if with_labels:
            l = np.ones(tr.shape[0]) * tld[2]
            l_ += [l]

    # subsample

    if use_data == 'final':
        t_ = [t[-1:] for t in t_]
        tr_ = [tr[-1:,:] for tr in tr_]
        if with_labels:
            l_ = [l[-1:] for l in l_]

    elif use_data == 'mean':
        t_ = [t[-1:] for t in t_]
        tr_ = [tr.mean(axis=0).reshape(1, -1) for tr in tr_]
        if with_labels:
            l_ = [l[-1:] for l in l_]

    elif use_data == 'mean_mean':
        N_voxel = 5

        t_ = [t[-1:] for t in t_]
        tr_ = [tr.reshape(tr.shape[0], N_voxel, -1).mean(axis=2).mean(axis=0, keepdims=True) for tr in tr_]
        if with_labels:
            l_ = [l[-1:] for l in l_]

    elif use_data == 'mean_sum':
        N_voxel = 5

        t_ = [t[-1:] for t in t_]
        tr_ = [tr.reshape(tr.shape[0], N_voxel, -1).sum(axis=2).mean(axis=0, keepdims=True) for tr in tr_]
        if with_labels:
            l_ = [l[-1:] for l in l_]

    t_ = np.concatenate(t_)
    tr_ = np.concatenate(tr_)
    if with_labels:
        l_ = np.concatenate(l_)

    # drop silent neurons
    if use_data == 'all' and drop_silent:
        mask = (tr_.sum(axis=0) > 0)
        tr_ = tr_[:,mask]

    if noise_var > 0:
        print('adding noise with var {:.1g}'.format(noise_var))
        tr_ += np.sqrt(noise_var) * np.random.randn(*tr_.shape)

    # decrease dimensionality

    if use_data == 'all':
        time_skip = 10
        neuron_skip = 1

        print('using time_skip', time_skip)
        print('using neuron_skip', neuron_skip)

        t_ = t_[::time_skip]
        tr_ = tr_[::time_skip,::neuron_skip]
        if with_labels:
            l_ = l_[::time_skip]

    elif use_data in ['mean', 'final']:
        neuron_skip = 2

        print('using neuron_skip', neuron_skip)

        tr_ = tr_[:,::neuron_skip]

    elif use_data in ['mean_mean', 'mean_sum']:
        # both temporal and spatial axes are already low-dim
        pass

    if with_labels:
        return t_, tr_, l_

    # check output

    assert t_.ndim == 1
    assert tr_.ndim == 2
    assert t_.shape[0] == tr_.shape[0]

    if with_labels:
        assert l_.ndim == 1
        assert t_.shape[0] == l_.shape[0]

    return t_, tr_


def train_readout(x, y, xt=None, yt=None):
    readout = LogisticRegression(solver='sag', max_iter=10000, multi_class='multinomial').fit(x, y)

    test = lambda xt, yt: sum(readout.predict(xt) != yt) / len(yt)

    if xt is None or yt is None:
        return test

    return test, test(xt, yt)
