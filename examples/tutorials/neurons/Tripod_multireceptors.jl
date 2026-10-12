# Tripod neuron whose dendrites have three GABA receptors of different kinetics
# (MultiReceptorSynapse): one input spike per receptor on dendrite 1, at 100, 200 and 300 ms.
using SpikingNeuralNetworks
using CairoMakie
SNN.@load_units

receptors = SNN.Receptors(
    SNN.Receptor(E_rev = -70.0f0, g0 = 1 / 2, τr = 0.5f0, τd = 2.0f0, α = 1.0f0, target = :gaba1),
    SNN.Receptor(E_rev = -70.0f0, g0 = 1 / 10, τr = 0.5f0, τd = 10.0f0, α = 1.0f0, target = :gaba2),
    SNN.Receptor(E_rev = -70.0f0, g0 = 1 / 100, τr = 0.5f0, τd = 100.0f0, α = 1.0f0, target = :gaba3),
)
synapse = SNN.MultiReceptorSynapse(syn = receptors)
E = SNN.Population(SNN.TripodParameter(), adex = SNN.AdExParameter(El = -55mV), N = 1,
                   dend_syn = synapse, spike = SNN.PostSpike())

conn = (p = 1, μ = 1.0f0)
st1 = SNN.Stimulus(SNN.SpikeTimeParameter([[0.1s]]), E, :gaba1, :d1, name = "stim1"; conn)
st2 = SNN.Stimulus(SNN.SpikeTimeParameter([[0.2s]]), E, :gaba2, :d1, name = "stim2"; conn)
st3 = SNN.Stimulus(SNN.SpikeTimeParameter([[0.3s]]), E, :gaba3, :d1, name = "stim3"; conn)

model = SNN.compose(; E, st1, st2, st3, silent = true)
SNN.monitor!(E, [:v_d1, :fire])
SNN.sim!(model; duration = 0.5second)

fig = Figure()
ax = Axis(fig[1, 1], title = "Tripod with multiple GABA receptors", ylabel = "V dendrite 1 (mV)", xlabel = "Time (s)")
SNN.vecplot!(ax, E, :v_d1, neurons = 1)
save(joinpath(mktempdir(), "tripod_multireceptors.png"), fig)
