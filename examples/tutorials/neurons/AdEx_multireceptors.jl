# AdEx neuron with three GABA receptors of different kinetics (MultiReceptorSynapse): one input
# spike per receptor, at 100, 200 and 300 ms.
using SpikingNeuralNetworks
using CairoMakie
SNN.@load_units

receptors = SNN.Receptors(
    SNN.Receptor(E_rev = -70.0f0, g0 = 1 / 2, τr = 0.5f0, τd = 2.0f0, α = 1.0f0, target = :gaba1),
    SNN.Receptor(E_rev = -70.0f0, g0 = 1 / 10, τr = 0.5f0, τd = 10.0f0, α = 1.0f0, target = :gaba2),
    SNN.Receptor(E_rev = -70.0f0, g0 = 1 / 100, τr = 0.5f0, τd = 100.0f0, α = 1.0f0, target = :gaba3),
)
synapse = SNN.MultiReceptorSynapse(syn = receptors)
E = SNN.Population(SNN.AdExParameter(El = -55mV), N = 1, synapse = synapse, spike = SNN.PostSpike())

conn = (p = 1, μ = 1.0f0)
st1 = SNN.Stimulus(SNN.SpikeTimeParameter([[0.1s]]), E, :gaba1, name = "stim1"; conn)
st2 = SNN.Stimulus(SNN.SpikeTimeParameter([[0.2s]]), E, :gaba2, name = "stim2"; conn)
st3 = SNN.Stimulus(SNN.SpikeTimeParameter([[0.3s]]), E, :gaba3, name = "stim3"; conn)

model = SNN.compose(; E, st1, st2, st3, silent = true)
SNN.monitor!(E, [:v, :fire])
SNN.sim!(model; duration = 0.5second)

fig = Figure()
ax = Axis(fig[1, 1], title = "AdEx neuron with multiple GABA receptors", ylabel = "V (mV)", xlabel = "Time (s)")
SNN.vecplot!(ax, E, :v, neurons = 1, add_spikes = true)
save(joinpath(mktempdir(), "AdEx_multireceptors.png"), fig)
