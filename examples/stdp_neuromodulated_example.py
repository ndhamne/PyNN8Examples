"""
Simple test for neuromodulated-STDP
We take 10 populations of 5 stimuli neurons and connect to each
10 post-synaptic neurons. The spiking of stimuli causes some
spikes in post-synaptic neurons initially.
We then inject reward signals from dopaminergic neurons
periodically to reinforce synapses that are active. This
is followed by increased weights of some synapses and thus
increased response to the stimuli.
We then proceed to inject punishment signals from dopaminergic
neurons which causes an inverse effect to reduce response of
post-synaptic neurons to the same stimuli.
"""

try:
    import pyNN.spiNNaker as sim
except Exception:
    import spynnaker8 as sim
import pylab

timestep = 1.0
stim_rate = 50
duration = 12000
plastic_weights = 1.5

# Times of rewards and punishments
rewards = [x for x in range(2000, 2010)] + \
          [x for x in range(3000, 3020)] + \
          [x for x in range(4000, 4100)]
punishments = [x for x in range(6000, 6010)] + \
              [x for x in range(7000, 7020)] + \
              [x for x in range(8000, 8100)]

cell_params = {'cm': 0.25,
               'i_offset': 0.0,
               'tau_m': 20.0,
               'tau_refrac': 2.0,
               'tau_syn_E': 1.0,
               'tau_syn_I': 1.0,
               'v_reset': -70.0,
               'v_rest': -65.0,
               'v_thresh': -50.0
               }

sim.setup(timestep=timestep)

# Create a population of dopaminergic neurons for reward and punishment
reward_pop = sim.Population(1, sim.SpikeSourceArray,
                            {'spike_times': rewards}, label='reward')
punishment_pop = sim.Population(1, sim.SpikeSourceArray,
                                {'spike_times': punishments},
                                label='punishment')

pre_pops = []
stimulation = []
post_pops = []
reward_projections = []
punishment_projections = []
plastic_projections = []
stim_projections = []

for i in range(10):
    stimulation.append(sim.Population(1, sim.SpikeSourcePoisson,
                       {'rate': stim_rate, 'duration': duration}, label="pre"))
    post_pops.append(sim.Population(1,
                                    sim.IF_curr_exp_izhikevich_neuromodulation,
                                    cell_params, label='post'))
    reward_projections.append(sim.Projection(reward_pop, post_pops[i],
                              sim.OneToOneConnector(),
                              synapse_type=sim.StaticSynapse(weight=0.05),
                              receptor_type='reward', label='reward synapses'))
    punishment_projections.append(sim.Projection(punishment_pop, post_pops[i],
                                  sim.OneToOneConnector(),
                                  synapse_type=sim.StaticSynapse(weight=0.05),
                                  receptor_type='punishment',
                                  label='punishment synapses'))

# Create synapse dynamics with neuromodulated STDP.
synapse_dynamics = sim.STDPMechanism(
    timing_dependence=sim.IzhikevichNeuromodulation(
        tau_plus=2, tau_minus=1,
        A_plus=1, A_minus=1,
        tau_c=100.0, tau_d=5.0),
    weight_dependence=sim.MultiplicativeWeightDependence(w_min=0, w_max=20),
    weight=plastic_weights,
    neuromodulation=True)

# Create plastic connections between stimulation populations and observed
# neurons
for i in range(10):
    plastic_projections.append(
        sim.Projection(stimulation[i], post_pops[i],
                       sim.AllToAllConnector(),
                       synapse_type=synapse_dynamics,
                       receptor_type='excitatory',
                       label='Pre-post projection'))
    post_pops[i].record('spikes')

sim.run(duration)

# Graphical diagnostics


def plot_spikes(spikes, title):
    if spikes is not None:
        pylab.figure(figsize=(15, 5))
        pylab.xlim((0, duration))
        pylab.ylim((0, 11))
        pylab.plot([i[1] for i in spikes], [i[0] for i in spikes], ".")
        pylab.xlabel('Time/ms')
        pylab.ylabel('spikes')
        pylab.title(title)
    else:
        print("No spikes received")


post_spikes = []
weights = []

for i in range(10):
    weights.append(plastic_projections[i].getWeights())
    spikes = post_pops[i].get_data('spikes').segments[0].spiketrains
    for x in spikes[0]:
        post_spikes.append([i+1, x])

plot_spikes(post_spikes, "post-synaptic neuron activity")
pylab.plot(rewards, [0.5 for x in rewards], 'g^')
pylab.plot(punishments, [0.5 for x in punishments], 'r^')
pylab.show()

print("Weights(Initial %s)" % plastic_weights)
for x in weights:
    print(x)

sim.end()