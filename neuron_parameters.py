#!/usr/bin/env python2

import numpy as np


def gen_EI_neuron_parameters(exc_gamma=1., exc_bias=0., inh_bias=0., tau_minus=20., q_sfa=0.):
    """
Generate parameters for the neurons of E- and I-pool.

Parameters:
    exc_gamma:          weight of V_m factor in exponential for E-neurons
                            (default: 1)
    exc_bias:           constant offset voltage for excitatory neurons in mV,
                            achieved by setting bias current (default: 0)
    inh_bias:           constant offset voltage for inhibitory neurons in mV,
                            achieved by setting bias current (default: 0)
    tau_minus:          time constant for STDP depression (default: 20 ms)

All other parameters are set to defautls.

Returns:
    neuron_model
    E_params
    I_params
    """

    # dt in original model: 1 ms
    dt = 1e-3

    # since the rate in the original model is valid for each time-step, i.e.
    #   the unit is 1 / dt, we scale the Nest model up accordingly to get a
    #   rate in Hz
    linear_scale = 1. / dt

    # set sfa parameters
    q_sfa_exp = -q_sfa
    q_sfa_lin = 0.
    tau_sfa = 5000.

    E_params = gen_exp_pp_parameters(
        exc_gamma,
        bias=exc_bias,
        tau_minus=tau_minus,
        q_sfa=q_sfa_exp,
        tau_sfa=tau_sfa,
        linear_scale=linear_scale)

    I_params = gen_lin_pp_parameters(
        bias=inh_bias,
        tau_minus=tau_minus,
        q_sfa=q_sfa_lin,
        linear_scale=linear_scale)

    neuron_model = E_params[0]

    return neuron_model, E_params[1], I_params[1]


def gen_lin_pp_parameters(bias, tau_minus, q_sfa=0., tau_sfa=200., abs_refrac=True, abs_refrac_random=True, linear_scale=1., z_scale=3e-4):
    """
Generate parameters for exponential pp_psc_delta neurons.

Rate:

    r(t) = V'_m(t)

with tau_m = 10 ms.

If abs_refrac_random is True, then the absolute refractory period is sampled
from a Gamma distribution with mean 3.5 and shape 4.

Parameters:
    bias:               constant offset voltage, achieved by setting bias current
    tau_minus:          for STDP depression
    q_sfa:              adaptive bias increment in mV (default: 0)
    tau_sfa:            adaptive bias decay time constant in ms (default: 200)
    abs_refrac:         use absolute refractory period and reset membrane potential
    abs_refrac_random:  absolute refractory period is random, otherwise it is
                            set to 3.5 ms
    linear_scale:       linear scaling factor for rate (default: 1)
    """

    c_1 = linear_scale
    c_2 = 0.
    c_3 = 0.

    return __gen_pp_parameters(
        c_1,
        c_2,
        c_3,
        bias=bias,
        z_scale=z_scale,
        tau_minus=tau_minus,
        q_sfa=q_sfa,
        tau_sfa=tau_sfa,
        abs_refrac=abs_refrac,
        abs_refrac_random=abs_refrac_random)


def gen_exp_pp_parameters(gamma, bias, tau_minus, q_sfa=0., tau_sfa=200., abs_refrac=True, abs_refrac_random=True, linear_scale=1., z_scale=3e-4):
    """
Generate parameters for exponential pp_psc_delta_mod neurons.

Rate:

    r(t) = linear_scale * (exp(gamma * V'_m(t)) - 1)

with tau_m = 10 ms.

If abs_refrac_random is True, then the absolute refractory period is sampled
from a Gamma distribution with mean 3.5 and shape 4.

Parameters:
    gamma:              weight of V'_m factor in exponential
    bias:               constant offset voltage, achieved by setting bias current
    tau_minus:          for STDP depression
    q_sfa:              adaptive bias increment in mV (default: 0)
    tau_sfa:            adaptive bias decay time constant in ms (default: 200)
    abs_refrac:         use absolute refractory period and reset membrane potential
    abs_refrac_random:  absolute refractory period is random, otherwise it is
                            set to 3.5 ms
    linear_scale:       linear scaling factor for rate (default: 1)
    """

    c_1 = 0.
    c_2 = linear_scale
    c_3 = gamma

    return __gen_pp_parameters(
        c_1,
        c_2,
        c_3,
        bias=bias,
        tau_minus=tau_minus,
        z_scale=z_scale,
        q_sfa=q_sfa,
        tau_sfa=tau_sfa,
        abs_refrac=abs_refrac,
        abs_refrac_random=abs_refrac_random)


def __gen_pp_parameters(c_1, c_2, c_3, bias, tau_minus, z_scale, q_sfa=0., tau_sfa=200., abs_refrac=True, abs_refrac_random=True):
    """
Generate parameters for linear or exponential pp_psc_delta_mod neurons.

Rate:

    r(t) = c_1 * V'_m(t) +  c_2 * (exp(c_3 * 1000/Rm * V'_m(t)) - 1)

Note the correction of c_3 by 1000/Rm.

If abs_refrac_random is True, then the absolute refractory period is sampled
from a Gamma distribution with mean 3.5 and shape 4.

Parameters:
    c_1:                weight of linear factor
    c_2:                weight of V'_m factor in exponential
    c_3:                weight of expontial factor
    bias:               constant offset voltage, achieved by setting bias current
    tau_minus:          for STDP depression
    q_sfa:              adaptive bias increment in mV (default: 0)
    tau_sfa:            adaptive bias decay time constant in ms (default: 200)
    abs_refrac:         use absolute refractory period and reset membrane potential
    abs_refrac_random:  absolute refractory period is random, otherwise it is
                            set to 3.5 ms
    z_scale:            scaling factor for the response to incoming spikes
    """

    # sanitize parameter types
    c_1 = float(c_1)
    c_2 = float(c_2)
    c_3 = float(c_3)
    bias = float(bias)
    tau_minus = float(tau_minus)
    q_sfa = float(q_sfa)
    tau_sfa = float(tau_sfa)

    neuron_model = 'pp_psc_delta_mod'

    # constant neuron parameters
    Rm = 10.  # MOhm
    Cm = 1000.  # pF
    tau_m = Rm * Cm / 1000.  # ms
    assert tau_m == 10.

    # correct gamma
    #   - divide through Rm because currents are multiplied with it by Nest
    #   - multiply with 1000 since 1 pA input should lead to 1 mV membrane
    #     potential
    c_3 *= 1000. / Rm

    # derived neuron parameters
    I_e = bias

    # refractory period parameters
    if not abs_refrac:
        dead_time = 0.
        dead_time_random = False
        dead_time_shape = 0
        reset = False
    else:
        dead_time = 3.5
        dead_time_random = False
        dead_time_shape = 0
        reset = True

        if abs_refrac_random:
            dead_time_random = True
            dead_time_shape = 4

    # using q_sfa < 0 to get increasing excitability, limit min(E_sfa) to
    # avoid instabilities
    E_sfa_max = -0.005

    neuron_params = {
        'V_m': 0.,                            # membrane potential, mV
        'V_reset': 0.,                        # reset membrane potential
        'C_m': Cm,                            # membrane capacity, pF
        'tau_m': tau_m,                       # membrane time constant, ms
        'q_sfa': q_sfa,                       # adaptive bias additive quantity, mV
        'tau_sfa': tau_sfa,                   # adaptive bias decay time constant, ms
        'E_sfa_clip': True,                   # clip adaptive bias
        'E_sfa_max': E_sfa_max,               # max value for adaptive bias, mV
        'dead_time': dead_time,               # dead time duration (or mean if random), ms
        'dead_time_random': dead_time_random, # reset time is gamma-distributed
        'dead_time_shape': dead_time_shape,   # shape of gamma distribution
        't_ref_remaining': 0.,                # remaining dead time at simulation start, ms
        'with_reset': reset,                  # reset membrane potential after each spike
        'I_e': I_e,                           # external input current, pA
        'c_1': c_1,                           # slope of linear part of transfer function, Hz/mv
        'c_2': c_2,                           # prefactor of exponential part of transfer function, Hz
        'c_3': c_3,                           # exponential coefficient of transfer function, 1/mV
        'tau_minus': tau_minus,               # STDP depression time constant, ms
        'z_scale': z_scale                    # scaling of incoming spikes
    }

    return neuron_model, neuron_params


def gen_connection_parameters(mu, var, cpsp, gamma, p_EI, p_IE, p_II, delay_correction=10.):
    """
Generate parameters for EI-motif according to Jonke et al., 2017, STDP

Parameters:
    mu:                 activity prior mean
    var:                activity prior variance
    cpsp:               PSP shape correction term
    p_EI:               E -> I connection probability
    p_IE:               I -> E connection probability
    p_II:               I -> I connection probability
    delay_correction:   correction factor for delay (default: 10)
    """

    alpha = (2.*mu - 1.) / (2. * var * gamma)
    w_EI = delay_correction * cpsp / p_EI
    w_IE = -cpsp / (gamma * var * p_IE)
    w_II = -cpsp * delay_correction / p_IE

    return w_EI, w_IE, w_II, alpha

