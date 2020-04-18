#!/usr/bin/env python3

import matplotlib.pyplot as plt
import nest
import numpy as np

nest.Install('vb_module')

V_reset = -30.
neuron = nest.Create('pp_psc_delta_mod', params={'V_reset': V_reset})

sg = nest.Create('spike_generator', params={'spike_times': np.asarray([20., 25., 30., 35., 40.])})
nest.Connect(sg, neuron, syn_spec={'weight': 20.})

vm = nest.Create('voltmeter')
nest.Connect(vm, neuron)


# run

nest.Simulate(50.)


# analysis

t = nest.GetStatus(vm)[0]['events']['times']
v = nest.GetStatus(vm)[0]['events']['V_m']

plt.figure()
plt.plot(t, v, label='V_m')
plt.plot(t, V_reset*np.ones(t.shape), label='V_reset')
plt.legend(loc='best')
plt.show()
