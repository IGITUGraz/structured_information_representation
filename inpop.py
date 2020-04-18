#!/usr/bin/env python3

import nest
import numpy as np

from analysis import *

class InPop():
    def __init__(self, num_pattern=5, neurons_per_pattern=25, rate_on=100., rate_off=.1, rate_clear=None, num_neurons=200):
        """
        Create a new instance of InPop.

        arguments:
            num_pattern           default: 5
            neurons_per_pattern   default: 25
            rate_on               default: 100.
            rate_off              default: 0.1
            rate_clear            default: None, then, the average rate is used
            num_neurons           default: 200
        """
        assert num_neurons >= num_pattern * neurons_per_pattern

        self._N = num_neurons
        self._K = num_pattern
        self._r_on = rate_on
        self._r_off = rate_off

        # create input generators
        generators = nest.Create('poisson_generator', num_neurons)

        # generators cannot be connected to stdp synapses, therefore put one
        # interneuron between each connector and each target neuron
        neurons = nest.Create('parrot_neuron', self._N)
        nest.Connect(generators, neurons, {'rule': 'one_to_one'})

        # create patterns
        self._pat = []

        for i in range(num_pattern):
            rates = rate_off * np.ones(num_neurons)

            ind_start = i * neurons_per_pattern
            ind_end = (i + 1) * neurons_per_pattern

            rates[ind_start:ind_end] = rate_on

            self._pat += [rates]

        if rate_clear == None:
            # determine rate for clear state: all neurons fire with their
            # average rate
            self._r_clear = np.mean(self._pat[0])
        else:
            # use manual setting
            self._r_clear = rate_clear

        self._gen = generators
        self.pop_X = neurons

        self.clear_spikes()  # create spike_detectors
        self.clear()  # clear to have population in a defined state

    def set(self, index):
        """
        Apply a pattern.

        args:
            index           pattern id
        """
        assert index < self._K

        for g, r in zip(self._gen, self._pat[index]):
            nest.SetStatus([g], {'rate': r})

    def set_random(self):
        """
        Apply a randomly chosen pattern.
        """
        pat = np.random.randint(self._K)
        self.set(pat)
        return pat

    def clear(self):
        """
        Clear pattern and apply the clear rate to all neurons.
        """
        nest.SetStatus(self._gen, {'rate': self._r_clear})

    def clear_spikes(self):
        """
        (Re-)Create spike recorder.
        """

        self.spike_rec = nest.Create('spike_detector')

        nest.Connect(self.pop_X, self.spike_rec, {'rule': 'all_to_all'})

    def get_spikes(self, t_lim=None, count=None, random=False):
        """
        Extract spikes from population with optional preprocessing.

        args:
            t_lim               optional, tuple of start and end time for
                                extracting only spikes within a certain time
                                range (useful for plotting)
            count               optional, number of neurons to extract spikes
                                from
            random              optional, only used if count is not None,
                                bool indicating whether the number of neurons
                                to be extract should be random, default:
                                False

        returns:
            times               extracted spike times
            senders             extracted spike sender ids
        """

        times, senders = get_spikes_from_spike_recorder(
                self.spike_rec,
                self.pop_X,
                t_lim,
                count,
                random)

        return times, senders

    @property
    def num_neurons(self):
        return self._N

    @property
    def num_pattern(self):
        return self._K

    @property
    def rate_on(self):
        return self._r_on

    @property
    def rate_off(self):
        return self._r_off

    @property
    def rate_clear(self):
        return self._r_clear

    @property
    def rates(self):
        r = [x['rate'] for x in nest.GetStatus(self._gen)]
        return np.asarray(r)
