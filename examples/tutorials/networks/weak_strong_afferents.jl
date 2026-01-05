using DrWatson
using Plots
using UnPack
using SpikingNeuralNetworks
using Statistics
SNN.@load_units;


Zerlaut2019_network = (
    # Number of neurons in each population
    Npop = (E = 4000, I = 1000),

    # Parameters for excitatory neurons
    exc = SNN.IFParameter(
        τm = 200pF / 10nS,  # Membrane time constant
        El = -70mV,         # Leak reversal potential
        Vt = -50.0mV,       # Spike threshold
        Vr = -70.0f0mV,     # Reset potential
        R = 1/10nS,        # Membrane resistance
        # a = 4nS,
        # b = 80pA
    ),

    # Parameters for inhibitory neurons
    inh = SNN.IFParameter(
        τm = 200pF / 10nS,  # Membrane time constant
        El = -70mV,         # Leak reversal potential
        Vt = -53.0mV,       # Spike threshold
        Vr = -70.0f0mV,     # Reset potential
        R = 1/10nS,        # Membrane resistance
    ),
    spike_exc = SNN.PostSpike(τabs = 2ms),         # Absolute refractory period
    spike_inh = SNN.PostSpike(τabs = 1ms),         # Absolute refractory period

    # Synaptic properties
    synapse_exc = SNN.SingleExpSynapse(
        τi = 5ms,             # Inhibitory synaptic time constant
        τe = 5ms,             # Excitatory synaptic time constant
        E_i = -80mV,        # Inhibitory reversal potential
        E_e = 0mV,           # Excitatory reversal potential
    ),
    synapse_inh = SNN.SingleExpSynapse(
        τi = 5ms,             # Inhibitory synaptic time constant
        τe = 5ms,             # Excitatory synaptic time constant
        E_i = -80mV,        # Inhibitory reversal potential
        E_e = 0mV,           # Excitatory reversal potential
    ),


    # Connection probabilities and synaptic weights
    connections = (
        E_to_E = (p = 0.05, μ = 2nS, rule = :Fixed), # Excitatory to excitatory
        E_to_I = (p = 0.05, μ = 2nS, rule = :Fixed), # Excitatory to inhibitory
        I_to_E = (p = 0.05, μ = 10nS, rule = :Fixed), # Inhibitory to excitatory
        I_to_I = (p = 0.05, μ = 10nS, rule = :Fixed), # Inhibitory to inhibitory
    ),

    # Parameters for external Poisson input
    afferents = (
        layer = SNN.PoissonLayer(rate = 10Hz, N = 100), # Poisson input layer
        conn = (p = 0.1f0, μ = 4.0nS), # Connection probability and weight
    ),
)

# %% [markdown]
# ## Network Construction
#
# Define a function to create the network based on the configuration parameters.

# %%
# Function to create the network
function network(config)
    @unpack afferents, connections, Npop, spike_exc, spike_inh, exc, inh = config
    @unpack synapse_exc, synapse_inh = config

    # Create neuron populations
    E = SNN.Population(
        exc;
        synapse = synapse_exc,
        spike = spike_exc,
        N = Npop.E,
        name = "E",
    )  # Excitatory population
    I = SNN.Population(
        inh;
        synapse = synapse_inh,
        spike = spike_inh,
        N = Npop.I,
        name = "I",
    )  # Inhibitory population

    # Create external Poisson input
    @unpack layer = afferents
    afferentE = SNN.Stimulus(layer, E, :glu, conn = afferents.conn, name = "noiseE")  # Excitatory input
    afferentI = SNN.Stimulus(layer, I, :glu, conn = afferents.conn, name = "noiseI")  # Inhibitory input

    # Create recurrent connections
    synapses = (
        E_to_E = SNN.SpikingSynapse(E, E, :glu, conn = connections.E_to_E, name = "E_to_E"),
        E_to_I = SNN.SpikingSynapse(E, I, :ge, conn = connections.E_to_I, name = "E_to_I"),
        I_to_E = SNN.SpikingSynapse(I, E, :gi, conn = connections.I_to_E, name = "I_to_E"),
        I_to_I = SNN.SpikingSynapse(I, I, :gi, conn = connections.I_to_I, name = "I_to_I"),
    )
    model = SNN.compose(;
        E,
        I,
        afferentE,
        afferentI,
        synapses...,
        silent = true,
        name = "Balanced network",
    )
    SNN.monitor!(model.pop, [:fire])
    SNN.monitor!(model.stim, [:fire])
    # monitor!(model.pop, [:v], sr=200Hz)
    return SNN.compose(; model..., silent = true)
end

config = SNN.@update Zerlaut2019_network begin
    afferents.layer.rate = 8Hz
end
model = network(config)
model.stim.afferentE
SNN.sim!(; model, duration = 5_000ms, pbar = true)
SNN.raster(model.pop)

νs = exp.(range(log(1), log(50), 20))
frs = []
sttcs = []
plots = map(νs) do input_rate
    config = SNN.@update Zerlaut2019_network begin
        afferents.layer.rate = input_rate * Hz
    end
    model = network(config)
    SNN.sim!(; model, duration = 6_000ms, pbar = true)

    # Firing rate of the network with a fixed afferent rate
    frE, r = SNN.firing_rate(model.pop.E, interval = 3s:5s, pop_average = true)
    sttc = SNN.STTC(model.pop.E, ΔT = 50ms, interval = 3s:5s) |> mean
    push!(frs, mean(frE))
    push!(sttcs, sttc)
end

plot(
    νs,
    frs,
    scale = :log10,
    xlabel = "Afferent rate (Hz)",
    ylabel = "Firing rate (Hz)",
    labels = ["E" "I"],
    lw = 2,
    ylims = (1e-6, 30),
    xlims = (1, 50),
    size = (600, 400),
)
plot!(
    twinx(),
    νs,
    sttcs,
    # xlabel = "Afferent rate (Hz)",
    ylabel = "STTC",
    labels = ["E" "I"],
    lw = 2,
    xlims = (1, 50),
    xscale = :log10,
    size = (600, 400),
    color = :red,
)
plot!(legend = false)

config = SNN.@update Zerlaut2019_network begin
    afferents.layer.rate = 8Hz
end
model = network(config)
model.stim.afferentE
SNN.sim!(; model, duration = 5_000ms, pbar = true)
SNN.raster(model.pop)
sttc = SNN.STTC(model.pop.E, ΔT = 50ms, interval = 0:3s)
heatmap(
    sttc,
    xlabel = "Neuron index",
    ylabel = "Neuron index",
    title = "Spike Time Tiling Coefficient (STTC) Matrix",
    colorbar_title = "STTC",
)

mean(sttc)
