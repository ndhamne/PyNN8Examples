import spynnaker8 as sim
import numpy as np
import pylab as plt
from signal_prep import *
from pyNN.random import NumpyRNG, RandomDistribution
import os
import subprocess

#================================================================================================
# Simulation parameters
#================================================================================================
ex_params = {#'cm': 0.25,  # nF
               # 'i_offset': 0.0,
               #'tau_m': 10.,#10.0,#2.,#3.,#
               # 'tau_refrac': 1.0,#2.0,#
               # 'tau_syn_E': 3.0,#2.5,#
               #'tau_syn_I': 10.0,#2.5,#
               'v_reset': -70.0,#-100.0, # use a large v_reset to produce 'boosting'
               'v_rest': -65.0,
               'v_thresh': -56.#-55.6
               }

w2s = 3.0
connection_weight = w2s/4.#initial_weight*2.#
number_of_inputs = 4
inh_weight = 0.15#w2s/4.#initial_weight#(n_connections-number_of_inputs)*(initial_weight)#*2.

input_spikes =[]
inh_spikes = []
isi = 100.
n_repeats = 10

for neuron in range(number_of_inputs):
    input_spikes.append([i*isi for i in range(n_repeats) if i<5 or i>10])
    inh_spikes.append([(i*isi)+2 for i in range(n_repeats) if i<5 or i>10])

#================================================================================================
# SpiNNaker setup
#================================================================================================
sim.setup(timestep=1.0, min_delay=1.0, max_delay=51.0)

#================================================================================================
# Populations
#================================================================================================
input_pop = sim.Population(number_of_inputs,sim.SpikeSourceArray(spike_times=input_spikes))
inh_pop = sim.Population(1,sim.SpikeSourceArray(spike_times=inh_spikes))
# cd_pop = sim.Population(1,sim.IF_curr_exp,target_cell_params)#,label="fixed_weight_scale")
# cd_pop = sim.Population(1,sim.IF_cond_exp,inh_cond_params,label="fixed_weight_scale_cond")
# cd_pop = sim.Population(1,sim.IF_curr_exp,ex_params,label="fixed_weight_scale")
cd_pop = sim.Population(1,sim.IF_curr_exp,ex_params,label="fixed_weight_scale")

cd_pop.record(["spikes","v"])

#================================================================================================
# Projections
#================================================================================================
input_projection = sim.Projection(input_pop,cd_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=connection_weight))
inh_projection = sim.Projection(inh_pop,cd_pop,sim.AllToAllConnector(),synapse_type=sim.StaticSynapse(weight=inh_weight),receptor_type='inhibitory')
# inh_projection = sim.Projection(inh_pop,cd_pop,sim.AllToAllConnector(),synapse_type=stdp_model_cd,receptor_type='inhibitory')

duration = max(input_spikes[0])

sim.run(duration)

cd_data = cd_pop.get_data(["spikes","v"])

sim.end()

mem_v = cd_data.segments[0].filter(name='v')
cell_voltage_plot_8(mem_v, plt, duration, [],scale_factor=0.001,title='cd pop')


spike_raster_plot_8(cd_data.segments[0].spiketrains,plt,duration/1000.,2,0.001,title="cd pop activity")
spike_raster_plot_8(input_spikes,plt,duration/1000.,number_of_inputs+1,0.001,title="input activity")

plt.show()