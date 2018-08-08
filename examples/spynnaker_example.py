import spynnaker8 as sim
import numpy
import math
import unittest
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

v_reset = -65
v_thresh = -50
rngseed = 98766987
parallel_safe = True
rng = sim.NumpyRNG(seed=rngseed, parallel_safe=parallel_safe)

# Initialise simulator
sim.setup(timestep=1)

# Spike sources
poisson_spike_source =  sim.Population(500, sim.SpikeSourcePoisson(
        rate=50, duration=5000), label='poisson_source')
spike_source_array = sim.Population(100, sim.SpikeSourceArray,
        {'spike_times': [6000]}, label='spike_source')

# Neuronal populations
pop_exc = sim.Population(500, sim.IF_curr_exp(),  label='excitatory_pop')
pop_inh = sim.Population(100, sim.IF_curr_exp(),  label='inhibitory_pop')
uniformDistr = sim.RandomDistribution('uniform', [v_reset, v_thresh], rng=rng)
pop_exc.set(v=uniformDistr)

# Spike input projections
spike_source_projection = sim.Projection(
    spike_source_array, pop_exc, sim.FixedProbabilityConnector(p_connect=0.01),
    sim.StaticSynapse(weight=5.0, delay=1), receptor_type='excitatory')
# Poisson source projections
poisson_projection_exc = sim.Projection(
    poisson_spike_source, pop_exc,
    sim.FixedProbabilityConnector(p_connect=0.20),
    synapse_type=sim.StaticSynapse(weight=0.025, delay=5),
    receptor_type='excitatory')
poisson_projection_inh = sim.Projection(
    poisson_spike_source, pop_inh,
    sim.FixedProbabilityConnector(p_connect=0.20),
    sim.StaticSynapse(weight=0.025, delay=6),
    receptor_type='excitatory')

# Recurrent projections
# exc_exc_rec = sim.Projection(pop_exc, pop_exc,
#     sim.OneToOneConnector(),
#     synapse_type=sim.StaticSynapse(weight=0.1, delay=2),
#     receptor_type='excitatory')
# inh_inh_rec = sim.Projection(pop_inh, pop_inh,
#     sim.FixedProbabilityConnector(p_connect=0.1),
#     synapse_type=sim.StaticSynapse(weight=0.33, delay=10),
#     receptor_type='inhibitory')

# # Neuronal population projections
# exc_to_inh = sim.Projection(pop_exc, pop_exc,
#     sim.FixedProbabilityConnector(p_connect=0.1),
#     synapse_type=sim.StaticSynapse(weight=0.1, delay=2),
#     receptor_type='excitatory')
# inh_to_exc = sim.Projection(pop_inh, pop_exc,
#     sim.FixedProbabilityConnector(p_connect=0.1),
#     synapse_type=sim.StaticSynapse(weight=0.33, delay=3),
#     receptor_type='inhibitory')

# Specify output recording
pop_exc.record('spikes')
#pop_inh.record('v', 'spikes')
pop_inh.record('spikes')

# Run simulation
sim.run(simtime=5000)

# Get results data
exc_data = pop_exc.get_data('spikes')
inh_data = pop_inh.get_data('spikes')


exc_data.segments[0].spiketrains.append(inh_data.segments[0].spiketrains)


# Plot spikes
Figure(
    Panel(exc_data.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, 5000)),
    Panel(inh_data.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, 5000)),
)
plt.show()

# Exit simulation
sim.end()

