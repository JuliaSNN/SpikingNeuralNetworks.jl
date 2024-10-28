using DrWatson
using Plots
using Revise
using SpikingNeuralNetworks
SNN.@load_units;

##
# Define neurons and synapses in the network
N = 100
E = SNN.TripodHet(
    N = N,
    soma_syn = Synapse(DuarteGluSoma, MilesGabaSoma),
    dend_syn = Synapse(EyalGluDend, MilesGabaDend),
    NMDA = SNN.EyalNMDA,
    param = SNN.AdExSoma(b = 0.0f0, Vr = -50),
)

E_to_E =
    SNN.CompartmentSynapse(E, E, :d1, :he, μ = 30, param = SNN.vSTDPParameter())
