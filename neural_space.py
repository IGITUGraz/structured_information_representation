#!/usr/bin/env python3

import nest
import warnings

from analysis import *
from swta import *

class NeuralSpace(SWTACircuit):
    """
    Neural Space class. Default state is disinhibited.
    """

    def __init__(self, name, config=None, wiring_EE=None, synapse_type='sem', disinhibited=True, activity_mu=1.):

        if config is not None and wiring_EE is not None:
            #warnings.warn("using both a custom config and reload a wiring scheme can make the circuit settings incoherent with the ones used in training!")
            print("NOTE: using both a custom config and reload a wiring scheme can make the circuit settings incoherent with the ones used in training!")

        # additional config required for NeuralSpace not included in SWTACircuit
        new_config = {
            'p_SE': .1,  # connection probability S (other space) to E
            'activity_mu': activity_mu, # mean of activity prior
        }

        # STDP params
        if synapse_type == 'sem':
            new_config['synapse'] = 'stdp_synapse_sem'
            new_config['w_SE_low'] = .0
            new_config['w_SE_max'] = .5
            new_config['w_SE_high'] = .5
            new_config['tau_plus_SE'] = 20.
            #new_config['alpha_SE'] = -1.
            new_config['alpha_SE'] = 0.
            new_config['eta_SE'] = 0.01
            new_config['A_minus_SE'] = .35
            new_config['synd_SE_low'] = 1.
            new_config['synd_SE_high'] = 10.
        else:
            raise NotImplementedError('bad synapse type requested.')

        if config is not None:
            new_config.update(config)  # overwrite default values with given ones

        # set config, this will also be used by the base class init
        self.config = new_config

        # init base class with this updated config
        super().__init__(name, wiring_EE=wiring_EE, synapse_type=synapse_type, disinhibited=disinhibited)


        # create models
        self._create_extra_models()

        # misc setup
        self._space_connections = []


    def _create_extra_models(self):
        c = self.config

        # models for plastic connections
        if c['synapse'] == 'stdp_synapse_sem':
            # SE: connections from other spaces to E-pool
            nest.CopyModel('stdp_synapse_sem', self.name+'_syn_SE', {
                'Wmax': c['w_SE_max'],
                'lambda': c['eta_SE'] / c['w_SE_max'],
                'tau_plus': c['tau_plus_SE'],
                'A_minus': c['A_minus_SE'],
                'alpha': c['alpha_SE'],
            })
        else:
            raise ValueError('Bad synapse type set.')

        # model dictionaries for connections
        model_dicts = {}

        # SE
        model_dicts['SE'] = {
            'model': self.name+'_syn_SE',
            'lambda': c['eta_SE'] / c['w_SE_max'],
            'weight': {
                'distribution': 'uniform',
                'low': c['w_SE_low'],
                'high': c['w_SE_high']},
            'delay': {
                'distribution': 'uniform',
                'low': c['synd_SE_low'],
                'high': c['synd_SE_high']}
        }

        # rule dictionaries for connections
        # cannot create rule_dict for SE connections since it depends on N_S

        self.model_dicts.update(model_dicts)


    def connect_input_space(self, space, symmetric=False):
        """
        E-pool of given space is connected to this space's E-pool. Multiple
        spaces can be connected this way.
        """

        # don't care about actually setting lambda here, this is done in the
        # __init__() of the NeuralSpaceConnection
        lambda_disinhibited = self.model_dicts['SE']['lambda']

        if not symmetric:
            c = self.config

            N_S = len(space.pop_E)

            # create rule_dict
            if c['p_SE'] == 1:
                rule_dict = {'rule': 'all_to_all'}
            elif c['bernoulli']:
                rule_dict = {'rule': 'pairwise_bernoulli', 'p': c['p_SE']}
            else:
                rule_dict = {
                    'rule': 'fixed_total_number',
                    'N': int(N_S*c['N_E']*c['p_SE'])}

            # connect spaces
            nest.Connect(space.pop_E, self.pop_E, rule_dict, self.model_dicts['SE'])
        else:
            # symmetric connect

            # find space connection
            space_connection = None

            for sc in self._space_connections:
                if sc._to_space == space:
                    space_connection = sc

            if space_connection == None:
                raise ValueError('attempting to symmetrically connect a space that hasn\'t previously been connected unsymmetrically in the other direction.')

            # get connections
            c = np.asarray(space_connection._connections)[:,:2]

            rule_dict = {'rule': 'one_to_one'}

            # connect reciprocally
            nest.Connect(c[:,1], c[:,0], rule_dict, self.model_dicts['SE'])

        # create entry for SE plasticity setting
        if not 'SE' in self._synaptic_plasticity.keys():
            self._synaptic_plasticity['SE'] = True

        # create connection management class, which will also trigger setting
        # the correct inhibition state
        conn = NeuralSpaceConnection(space, self, lambda_disinhibited)

        # register space connection for inhibition management
        self._space_connections += [conn]
        space._space_connections += [conn]

    def get_weights(self):
        weights = super().get_weights()

        for sc in self._space_connections:
            # only give inbound weights
            if not sc.to_space == self:
                continue

            from_name = sc.from_space.name

            label = 'SE_{0:s}'.format(from_name)
            conn = sc.connections

            weights[label] = get_weights_from_connections(conn)

        return weights


    def get_wiring(self, normalize_indices=True):
        wiring = super().get_wiring(normalize_indices=normalize_indices)

        for k, sc in enumerate(self._space_connections):
            #label = 'SE{0:d}'.format(k)
            label = 'SE_{0:s}_to_{1:s}'.format(sc.from_space.name, sc.to_space.name)

            pop_in = sc.from_space.pop_E
            pop_out = sc.to_space.pop_E
            conn = sc.connections

            wiring[label] = self._get_wiring(pop_in, pop_out, conn, normalize_indices=normalize_indices)

        return wiring


    def inhibit(self, reset_sfa=False):
        super().inhibit(reset_sfa=reset_sfa)
        self.__refresh_connections()


    def disinhibit(self):
        super().disinhibit()
        self.__refresh_connections()


    def set_synaptic_plasticity(self, EE=None, XE=None, SE=None):
        super().set_synaptic_plasticity(EE=EE, XE=XE)

        if SE is not None:
            assert len(self._space_connections) > 0
            self._synaptic_plasticity['SE'] = SE

        self.__refresh_connections()


    def __refresh_connections(self):
        for conn in self._space_connections:
            conn.refresh()


class NeuralSpaceConnection:
    def __init__(self, from_space, to_space, lambda_disinhibited, lambda_inhibited=0.):
        self._from_space = from_space
        self._to_space = to_space

        self.lambda_disinhibited = lambda_disinhibited
        self.lambda_inhibited = lambda_inhibited

        self._connections = nest.GetConnections(from_space.pop_E, to_space.pop_E)

        self._inhibited = None
        self.refresh()

    @property
    def from_space(self):
        return self._from_space

    @property
    def to_space(self):
        return self._to_space

    @property
    def connections(self):
        return self._connections

    def refresh(self):
        """
        Update inhibition status of connections according to the inhibition
        state of the two connected spaces.
        """
        from_inhibited = self._from_space._inhibited
        to_inhibited = self._to_space._inhibited
        no_plasticity = not self._to_space._synaptic_plasticity['SE']

        new_inhibition_state = from_inhibited or to_inhibited or no_plasticity

        if self._inhibited is None or self._inhibited != new_inhibition_state:
            if new_inhibition_state == True:
                l = self.lambda_inhibited
            else:
                l = self.lambda_disinhibited

            nest.SetStatus(self._connections, {'lambda': l})

        self._inhibited = new_inhibition_state
