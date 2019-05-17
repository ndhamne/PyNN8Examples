import matplotlib.pyplot as pylab
import math
try:
    import pyNN.spiNNaker as sim
except Exception:
    import spynnaker8 as sim

# ------------------------------------------------------------------
# This example uses the sPyNNaker implementation of pair-based STDP
# To reproduce the eponymous STDP curve first
# Plotted by Bi and Poo (1998)
# ------------------------------------------------------------------
def stdp_param_check(alpha_plus,alpha_minus,w_max,tau_minus,tau_plus):

    min_w_delta = w_max/2.**16
    max_tau_plus_delta = -math.log((min_w_delta/alpha_plus))*tau_plus
    max_tau_minus_delta = -math.log((min_w_delta/alpha_minus))*tau_minus

    return max_tau_plus_delta,max_tau_minus_delta,min_w_delta
# ------------------------------------------------------------------
# Common parameters
# ------------------------------------------------------------------
n = 16
time_between_pairs = 1000
num_pairs = 60
max_w = 1.0/n
start_w = 0.5/n
delta_t = [-100, -60, -40, -30, -20, -10, -1, 1, 10, 20, 30, 40, 60, 100]
start_time = 200
mad = True

w2s =2.
w_max = 1.
start_w = 0.5
tau_plus=16.7
tau_minus=33.7
a_plus = 0.005
a_minus = 0.005

max_tau_plus_delta,max_tau_minus_delta,min_w_delta = stdp_param_check(a_plus,a_minus,w_max,tau_minus,tau_plus)


# Population parameters
model = sim.IF_curr_exp
cell_params = {'cm': 0.25,  # nF
               'i_offset': 0.0,
               'tau_m': 10.0,
               'tau_refrac': 2.0,
               'tau_syn_E': 2.5,
               'tau_syn_I': 2.5,
               'v_reset': -70.0,
               'v_rest': -65.0,
               'v_thresh': -55.4
               }

# SpiNNaker setup
sim.setup(timestep=1.0, min_delay=1.0, max_delay=10.0)

# -------------------------------------------------------------------
# Experiment loop
# -------------------------------------------------------------------
projections = []
sim_time = 0
for t in delta_t:
    # Calculate phase of input spike trains
    # If M.A.D., take into account dendritic delay
    if mad:
        # Pre after post
        if t > 0:
            post_phase = 0
            pre_phase = t + 1
        # Post after pre
        else:
            post_phase = -t
            pre_phase = 1
    # Otherwise, take into account axonal delay
    else:
        # Pre after post
        if t > 0:
            post_phase = 1
            pre_phase = t
        # Post after pre
        else:
            post_phase = 1 - t
            pre_phase = 0

    sim_time = max(sim_time, (num_pairs * time_between_pairs) + abs(t))

    # Neuron populations
    pre_pop = sim.Population(1, model(**cell_params))
    post_pop = sim.Population(1, model, cell_params)

    # Stimulating populations
    pre_times = [i for i in range(pre_phase, sim_time, time_between_pairs)]
    post_times = [i for i in range(post_phase, sim_time, time_between_pairs)]
    pre_stim = sim.Population(
        1, sim.SpikeSourceArray(spike_times=[pre_times]))
    post_stim = sim.Population(
        1, sim.SpikeSourceArray(spike_times=[post_times]))

    weight = 2.0

    # Connections between spike sources and neuron populations
    ee_connector = sim.OneToOneConnector()
    sim.Projection(
        pre_stim, pre_pop, ee_connector, receptor_type='excitatory',
        synapse_type=sim.StaticSynapse(weight=w2s))
    sim.Projection(
        post_stim, post_pop, ee_connector, receptor_type='excitatory',
        synapse_type=sim.StaticSynapse(weight=w2s))

    # Plastic Connection between pre_pop and post_pop
    stdp_model = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(
        #timing_dependence=sim.extra_models.SpikeNearestPairRule(
            tau_plus=tau_plus, tau_minus=tau_minus, A_plus=a_plus, A_minus=a_minus),
        weight_dependence=sim.AdditiveWeightDependence(
            w_min=0.0, w_max=w_max), weight=start_w)

    projections.append(sim.Projection(
        pre_pop, post_pop, sim.OneToOneConnector(),
        synapse_type=stdp_model))

print("Simulating for %us" % (sim_time / 1000))
post_pop.record("spikes")
# Run simulation
sim.run(sim_time)

post_pop_spikes = post_pop.get_data("spikes")
# Get weight from each projection
end_w = [p.get('weight', 'list', with_address=False)[0] for p in projections]

# End simulation on SpiNNaker
sim.end()
pp_spikes = post_pop_spikes.segments[0].spiketrains
# -------------------------------------------------------------------
# Plot curve
# -------------------------------------------------------------------
# Calculate deltas from end weights
delta_w = [(w - start_w) / start_w for w in end_w]

# Plot STDP curve
figure, axis = pylab.subplots()
axis.set_xlabel(r"$(t_{j} - t_{i}/ms)$")
axis.set_ylabel(r"$(\frac{\Delta w_{ij}}{w_{ij}})$",
                rotation="horizontal", size="xx-large")
axis.plot(delta_t, delta_w)
axis.axhline(color="grey", linestyle="--")
axis.axvline(color="grey", linestyle="--")

print "----------Weight scaling + STDP parameters----------"
print "max_tau_plus:{}".format(max_tau_plus_delta), "max_tau_minus:{}".format(max_tau_minus_delta),\
    "min_w_delta:{}".format(min_w_delta)

pylab.show()
