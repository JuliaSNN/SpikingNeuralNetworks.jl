using DrWatson
using Plots
using SpikingNeuralNetworks
SNN.@load_units;
using Random, Statistics, StatsBase
using Statistics, SparseArrays

# %% [markdown]
# Create vectors of dendritic parameters and the Tripod model
N = 1
d1 = [SNN.Dendrite(; create_dendrite(l = rand(250:1:350) * um)...) for n = 1:N]
d2 = [SNN.Dendrite(; create_dendrite(l = rand(250:1:350) * um)...) for n = 1:N]
Tripod_pop = SNN.TripodNeurons(
    N = N,
    d1 = d1,
    d2 = d2,
    soma_syn = Synapse(DuarteGluSoma, MilesGabaSoma),
    dend_syn = Synapse(EyalGluDend, MilesGabaDend),
    NMDA = SNN.EyalNMDA,
    param = SNN.AdExSoma(b = 0.0f0, Vr = -50),
)
# background
N_E = 1000
N_I = 200
ν_E = 50Hz
ν_I = 50Hz
r0 = 10Hz
v0_d1 = -40mV
v0_d2 = -40mV
μ_s = 1.5f0

I = SNN.Poisson(N = N_I, param = SNN.PoissonParameter(rate = ν_E))
E = SNN.Poisson(N = N_E, param = SNN.PoissonParameter(rate = ν_I))
inh_d1 = SNN.CompartmentSynapse(
    I,
    Tripod_pop,
    :d1,
    :inh,
    p = 0.2,
    μ = 1,
    param = SNN.iSTDPParameterPotential(v0 = v0_d1),
)
inh_d2 = SNN.CompartmentSynapse(
    I,
    Tripod_pop,
    :d2,
    :inh,
    p = 0.2,
    μ = 1,
    param = SNN.iSTDPParameterPotential(v0 = v0_d2),
)
inh_s = SNN.CompartmentSynapse(
    I,
    Tripod_pop,
    :s,
    :inh,
    p = 0.1,
    μ = 1,
    param = SNN.iSTDPParameterRate(r = r0),
)
exc_d1 = SNN.CompartmentSynapse(E, Tripod_pop, :d1, :exc, p = 0.2, μ = 15.0)
exc_d2 = SNN.CompartmentSynapse(E, Tripod_pop, :d2, :exc, p = 0.2, μ = 15.0)
exc_s = SNN.CompartmentSynapse(E, Tripod_pop, :s, :exc, p = 0.2, μ = μ_s)

synapses = [inh_d1, inh_d2, inh_s, exc_d1, exc_d2, exc_s]
populations = [Tripod_pop, I, E]

SNN.train!(populations, synapses, duration = 5000ms)
##
SNN.monitor(Tripod_pop, [:fire, :v_d1, :v_s, :v_d2])
SNN.sim!(populations, synapses, duration = 2000ms)
SNN.vecplot(Tripod_pop, [:v_d2, :v_d1, :v_s])
