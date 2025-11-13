using SpikingNeuralNetworks

## Example of AdEx neuron with multiple GABA receptors
## This implementation is based on MultiReceptorSynapse, which allows defining an arbitrary number of receptors with different kinetics and reversal potentials.
## However, this is not the most efficient way, if you need multiple receptors and care for performance, consider implementing a custom synapse model. 

receptors = SNN.Receptors(
    SNN.Receptor(
        E_rev = -70.0f0,
        g0 = 1/2,
        τr = 0.5f0,
        τd = 2.0f0,
        α = 1.0f0,
        target = :gaba1,
    ),  #
    SNN.Receptor(
        E_rev = -70.0f0,
        g0 = 1/10,
        τr = 0.5f0,
        τd = 10.0f0,
        α = 1.0f0,
        target = :gaba2,
    ),
    SNN.Receptor(
        E_rev = -70.0f0,
        g0 = 1/100,
        τr = 0.5f0,
        τd = 100.0f0,
        α = 1.0f0,
        target = :gaba3,
    ),
)
synapse = SNN.MultiReceptorSynapse(syn = receptors)

E = SNN.Population(
    SNN.AdExParameter(El = -55mV),
    N = 1,
    synapse = synapse,
    spike = SNN.PostSpike(),
)
model = SNN.compose(E)
SNN.sim!(model; duration = 0.5second)
SNN.reset_time!(model)

conn = (p = 1, μ = 1.0f0)
st1 =
    SNN.SpikeTimeParameter([[0.1s]]) |>
    x -> SNN.Stimulus(x, E, :gaba1, name = "stim1"; conn)
st2 =
    SNN.SpikeTimeParameter([[0.2s]]) |>
    x -> SNN.Stimulus(x, E, :gaba2, name = "stim2"; conn)
st3 =
    SNN.SpikeTimeParameter([[0.3s]]) |>
    x -> SNN.Stimulus(x, E, :gaba3, name = "stim3"; conn)

model = SNN.compose(; E, st1, st2, st3)
SNN.monitor!(E, [:v, :fire])
SNN.sim!(model; duration = 0.5second)
p = SNN.vecplot!(plot(), E, :v, neurons = 1, r = 0.0s:0.5s, add_spikes = true)
plot!(title = "AdEx Neuron with Multiple GABA Receptors", ylims = :auto)
