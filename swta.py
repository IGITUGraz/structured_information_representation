#!/usr/bin/env python3

import nest

from analysis import *
from neuron_parameters import *

class SWTACircuit:
    """
    Default state is disinhibited.
    """

    def __init__(self, name, wiring_EE=None, synapse_type='sem', disinhibited=True, N_E=1000, activity_mu=1.):

        self.name = name  # used to create unique model names for Nest models

        # network params

        config = {
            # neuron parameters
            'q_sfa': 0.,


            # network parameters

            'N_E': int(N_E),
            'N_I': int(N_E*.25),

            'p_XE': 1.,  # connection probability X (input) to E
            'p_EE': .1,  # connection probability E to E
            'p_EI': .575,  # connection probability E to I
            'p_IE': .6,  # connection probability I to E
            'p_II': .55,  # connection probability I to I

            'activity_mu': activity_mu,  # for gaussian prior
            'activity_var': 5. , # for gaussian prior
            'activity_cpsp': 1.,


            # connection params

            'bernoulli': True,  # use bernoulli or total number connections

            'w_XE_low': .0,  # uniform init
            'w_XE_high': .8,  # uniform init
            'w_XE_max': .8,  # upper bound

            'w_EE_low': .0,  # uniform init
            'w_EE_high': .01,  # uniform init
            'w_EE_max': .6,  # upper bound

            'synd_XE_low': 1.,  # random synaptic delay for XE
            'synd_XE_high': 10.,  # random synaptic delay for XE

            'synd_EE': 1.,  # const synaptic delay for EE, EI, IE, II
            'synd_EI': .5,
            'synd_IE': .5,
            'synd_II': 1.,


            # learning params

            'eta_XE': 0.01,
            'eta_EE': 0.0025
        }

        # STDP params

        config['synapse_type'] = synapse_type

        if synapse_type == 'sem':
            config['synapse'] = 'stdp_synapse_sem'
            config['tau_minus'] = 25.
            config['tau_plus_XE'] = 25.
            config['tau_plus_EE'] = 25.
            config['alpha_XE'] = 0.
            #config['alpha_EE'] = 0.
            config['alpha_EE'] = -1.
            config['A_minus_XE'] = .4
            config['A_minus_EE'] = .5
        else:
            raise NotImplementedError('bad synapse type requested.')

        # merge config from this base class with configs possibly set by
        # derived classed
        try:
            self.config
        except AttributeError:
            self.config = {}

        config.update(self.config)
        self.config = config


        # generate parameters

        self.config['gamma'] = 1.

        J_EI, J_IE, J_II, bias = self.__gen_connection_parameters(config)

        self.config['J_EI'] = J_EI
        self.config['J_IE'] = J_IE
        self.config['J_II'] = J_II
        self.config['bias'] = bias

        neuron_model, neuron_params_E, neuron_params_I = gen_EI_neuron_parameters(
            exc_gamma=config['gamma'],
            exc_bias=config['bias'],
            tau_minus=config['tau_minus'],
            q_sfa=config['q_sfa'])

        self.config['neuron_model'] = neuron_model
        self.config['neuron_params_E'] = neuron_params_E
        self.config['neuron_params_I'] = neuron_params_I

        # create nodes

        nest.CopyModel(neuron_model, self.name+'_E', neuron_params_E)
        self.pop_E = nest.Create(self.name+'_E', config['N_E'])

        nest.CopyModel(neuron_model, self.name+'I', neuron_params_I)
        self.pop_I = nest.Create(self.name+'I', config['N_I'])

        self.pop = self.pop_E + self.pop_I

        self._disinhibition_current_gen = nest.Create('step_current_generator')
        t, dt, t_max = self.__get_nest_times()
        inhibition_amplitude = -2.
        nest.SetStatus(self._disinhibition_current_gen, {
            'amplitude_values': np.asarray([inhibition_amplitude]),
            'amplitude_times': np.asarray([t + dt]),
            'start': t_max,
            'stop': t_max
        })


        # connect nodes
        self._create_models()

        if wiring_EE is None:
            nest.Connect(self.pop_E, self.pop_E, self.rule_dicts['EE'], self.model_dicts['EE'])
        else:
            # use wiring as passed
            model_dict = self.model_dicts['EE']
            self._restore_wiring(self.pop_E, self.pop_E, wiring_EE, model_dict)

        nest.Connect(self.pop_E, self.pop_I, self.rule_dicts['EI'], self.model_dicts['EI'])
        nest.Connect(self.pop_I, self.pop_E, self.rule_dicts['IE'], self.model_dicts['IE'])
        nest.Connect(self.pop_I, self.pop_I, self.rule_dicts['II'], self.model_dicts['II'])

        nest.Connect(self._disinhibition_current_gen, self.pop_E)
        nest.Connect(self._disinhibition_current_gen, self.pop_I)


        # misc setup

        self.clear_spikes()  # create spike_detectors

        self._EE_connections = nest.GetConnections(self.pop_E, self.pop_E)
        self._inhibited = False

        if not disinhibited:
            self.inhibit()

        self._synaptic_plasticity = {'EE': True}

        self.pop_X = None


    def _create_models(self):
        c = self.config


        # models for static connections

        nest.CopyModel('static_synapse', self.name+'_syn_EI', {'weight': c['J_EI']})
        nest.CopyModel('static_synapse', self.name+'_syn_IE', {'weight': c['J_IE']})
        nest.CopyModel('static_synapse', self.name+'_syn_II', {'weight': c['J_II']})


        # models for plastic connections

        if c['synapse'] == 'stdp_synapse_sem':

            # EE: recurrent connections in E-pool
            nest.CopyModel('stdp_synapse_sem', self.name+'_syn_EE', {
            'Wmax': c['w_EE_max'],
            'lambda': c['eta_EE'] / c['w_EE_max'],  # correct for update scaling in Nest
            'tau_plus': c['tau_plus_EE'],
            'A_minus': c['A_minus_EE'],
            'alpha': c['alpha_EE']
            })

            # XE: connections from an input population to E-pool
            nest.CopyModel('stdp_synapse_sem', self.name+'_syn_XE', {
                'Wmax': c['w_XE_max'],
                'lambda': c['eta_XE'] / c['w_XE_max'],
                'tau_plus': c['tau_plus_XE'],
                'A_minus': c['A_minus_XE'],
                'alpha': c['alpha_XE'],
            })

        else:
            raise ValueError('Bad synapse type set.')

        # model dictionaries for connections
        model_dicts = {}

        # EE
        model_dicts['EE'] = {
            'model': self.name+'_syn_EE',
            #'weight': {'distribution': 'normal', 'mu': w_mu, 'sigma': w_sigma},
            'weight': {'distribution': 'uniform', 'low': c['w_EE_low'], 'high': c['w_EE_high']},
            'delay': c['synd_EE']
        }

        # EI
        model_dicts['EI'] = {
            'model': self.name+'_syn_EI',
            'delay': c['synd_EI']
        }

        # IE
        model_dicts['IE'] = {
            'model': self.name+'_syn_IE',
            'delay': c['synd_IE']
        }

        # II
        model_dicts['II'] = {
            'model': self.name+'_syn_II',
            'delay': c['synd_II']
        }

        # XE
        model_dicts['XE'] = {
            'model': self.name+'_syn_XE',
            'weight': {
                'distribution': 'uniform',
                'low': c['w_XE_low'],
                'high': c['w_XE_high']},
            'delay': {
                'distribution': 'uniform',
                'low': c['synd_XE_low'],
                'high': c['synd_XE_high']}
        }

        # rule dictionaries for connections
        rule_dicts = {}

        if c['bernoulli']:
            rule_dicts['EE'] = {'rule': 'pairwise_bernoulli', 'p': c['p_EE']}
            rule_dicts['EI'] = {'rule': 'pairwise_bernoulli', 'p': c['p_EI']}
            rule_dicts['IE'] = {'rule': 'pairwise_bernoulli', 'p': c['p_IE']}
            rule_dicts['II'] = {'rule': 'pairwise_bernoulli', 'p': c['p_II']}
        else:
            # EE
            rule_dicts['EE'] = {
                "rule": "fixed_total_number",
                "N": int(c['N_E']*c['N_E']*c['p_EE'])}

            # EI
            rule_dicts['EI'] = {
                "rule": "fixed_total_number",
                "N": int(c['N_E']*c['N_I']*c['p_EI'])}

            # IE
            rule_dicts['IE'] = {
                "rule": "fixed_total_number",
                "N": int(c['N_I']*c['N_E']*c['p_IE'])}

            # II
            rule_dicts['II'] = {
                "rule": "fixed_total_number",
                "N": int(c['N_I']*c['N_I']*c['p_II'])}

        # XE
        # created in connect_input() because N_X is not known here

        self.model_dicts = model_dicts
        self.rule_dicts = rule_dicts


    def __gen_connection_parameters(self, config):
        return gen_connection_parameters(
            config['activity_mu'],
            config['activity_var'],
            config['activity_cpsp'],
            config['gamma'],
            config['p_EI'],
            config['p_IE'],
            config['p_II'])


    def connect_input(self, pop_X, wiring_XE=None):
        """
        Connect an input population to this space's E-pool. This operation may
        only be performed once.
        """
        c = self.config

        if self.pop_X is not None:
            raise ValueError('input already connected.')

        assert c['p_XE'] == 1.

        N_X = len(pop_X)

        model_dict = self.model_dicts['XE'].copy()

        # set lambda according to inhibition state
        if self._inhibited:
          model_dict['lambda'] = 0.

        if wiring_XE is None:
            if c['p_XE'] == 1:
                self.rule_dicts['XE'] = {'rule': 'all_to_all'}
            elif c['bernoulli']:
                self.rule_dicts['XE'] = {'rule': 'pairwise_bernoulli', 'p': c['p_XE']}
            else:
                self.rule_dicts['XE'] = {
                    'rule': 'fixed_total_number',
                    'N': int(N_X*c['N_E']*c['p_XE'])}

            nest.Connect(pop_X, self.pop_E, self.rule_dicts['XE'], model_dict)
        else:
            # use wiring as passed
            model_dict = self.model_dicts['XE']
            self._restore_wiring(pop_X, self.pop_E, wiring_XE, model_dict)

        self.pop_X = pop_X
        self._XE_connections = nest.GetConnections(self.pop_X, self.pop_E)
        self._synaptic_plasticity['XE'] = True

    def get_spikes_legacy(self):
        spikes_E = nest.GetStatus(self.spike_rec_E)[0]['events']
        spikes_I = nest.GetStatus(self.spike_rec_I)[0]['events']

        return spikes_E, spikes_I

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
            spikes_E            list containing the senders and times of the
                                spikes in the excitatory pool
            spikes_I            list containing the senders and times of the
                                spikes in the inhibitory pool
        """

        times_E, senders_E = get_spikes_from_spike_recorder(
                self.spike_rec_E,
                self.pop_E,
                t_lim,
                count,
                random)

        times_I, senders_I = get_spikes_from_spike_recorder(
                self.spike_rec_I,
                self.pop_I,
                t_lim,
                count,
                random)

        return [[times_E, senders_E], [times_I, senders_I]]


    def clear_spikes(self):

        self.spike_rec_E = nest.Create('spike_detector')
        self.spike_rec_I = nest.Create('spike_detector')

        nest.Connect(self.pop_E, self.spike_rec_E, {'rule': 'all_to_all'})
        nest.Connect(self.pop_I, self.spike_rec_I, {'rule': 'all_to_all'})


    def get_weights(self):
        weights = {'EE': get_weights_from_population(self.pop_E, self.pop_E)}

        if self.pop_X is not None:
            weights['XE'] = get_weights_from_population(self.pop_X, self.pop_E)

        return weights


    def get_wiring(self, normalize_indices=True):
        wiring = {'EE': self._get_wiring(self.pop_E, self.pop_E, normalize_indices=normalize_indices)}

        if self.pop_X is not None:
            wiring['XE'] = self._get_wiring(self.pop_X, self.pop_E, normalize_indices=normalize_indices)

        return wiring


    def _get_wiring(self, pop_in, pop_out, connections=None, normalize_indices=True):
        wiring = []

        if connections is None:
            connections = nest.GetConnections(pop_in, pop_out)

        status = nest.GetStatus(connections)

        for c, s in zip(connections, status):
            index_in = pop_in.index(c[0]) if normalize_indices else c[0]
            index_out = pop_out.index(c[1]) if normalize_indices else c[1]

            w = s['weight']
            wiring += [(index_in, index_out, w)]

        return wiring


    def _restore_wiring(self, pop_in, pop_out, wiring, model_dict):
        # create target unit vectors
        target_in = [pop_in[w[0]] for w in wiring]
        target_out = [pop_out[w[1]] for w in wiring]
        weights = [w[2] for w in wiring]

        model_dict['weight'] = weights

        # connect one-to-one
        nest.Connect(target_in, target_out, {'rule': 'one_to_one'}, model_dict)


    def inhibit(self, reset_sfa=False):
        t, dt, t_max = self.__get_nest_times()

        nest.SetStatus(self._disinhibition_current_gen, {'start': t + dt})

        if reset_sfa:
            self.reset_sfa()

        # set learning rates to zero to prevent depression
        self._disable_plasticity()

        self._inhibited = True


    def disinhibit(self):
        t, dt, t_max = self.__get_nest_times()

        nest.SetStatus(self._disinhibition_current_gen, {'start': t_max})

        # restore learning rates
        self._enable_plasticity()

        self._inhibited = False


    @property
    def inhibited(self):
        return self._inhibited


    def set_synaptic_plasticity(self, EE=None, XE=None):
        if EE is not None:
            self._synaptic_plasticity['EE'] = EE

        if XE is not None:
            assert self.pop_X is not None
            self._synaptic_plasticity['XE'] = XE

        if not self._inhibited:
            self._enable_plasticity()  # refresh plasticity state


    def _disable_plasticity(self):
        # This function allows to disable the synaptic plasticity only
        #   temporarily - for use with circuit inhibition - without setting the
        #   class variable _synaptic_plasticity. This way, the plasticity will
        #   be restored if the circuit is disinhibited. If the user wants to
        #   disable synaptic plasticity completely, the disable_plasticity()
        #   function without underscore makes the change permanent.

        nest.SetStatus(self._EE_connections, {'lambda': 0.})

        if self.pop_X is not None:
            nest.SetStatus(self._XE_connections, {'lambda': 0.})


    def _enable_plasticity(self):
        """
        Set the synaptic plasticity of all connections according to the values
        in the self._synaptic_plasticity dict.
        """

        # use correction for update scaling in Nest

        # EE
        if self._synaptic_plasticity['EE']:
            lambda_EE = self.config['eta_EE'] / self.config['w_EE_max']
        else:
            lambda_EE = 0.

        nest.SetStatus(self._EE_connections, {'lambda': lambda_EE})


        # XE
        if self.pop_X is not None:
            if self._synaptic_plasticity['XE']:
                lambda_XE = self.config['eta_XE'] / self.config['w_XE_max']
            else:
                lambda_XE = 0.

            nest.SetStatus(self._XE_connections, {'lambda': lambda_XE})


    @property
    def synaptic_plasticity(self):
        return self._synaptic_plasticity


    def __get_nest_times(self):
        ks = nest.GetKernelStatus()
        t = ks['time']
        dt = ks['resolution']
        t_max = ks['T_max']

        return t, dt, t_max


    def reset_sfa(self):
        nest.SetStatus(self.pop_E, {'E_sfa': 0.})
