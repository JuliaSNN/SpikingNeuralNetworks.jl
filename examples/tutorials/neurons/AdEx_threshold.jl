using SpikingNeuralNetworks
import SNNPlots: vecplot, vecplot!, plot, histogram
SNN.@load_units

## Inspired by
# https://brian2.readthedocs.io/en/stable/examples/adaptive_threshold.html

adex_neuron = (
    param = SNN.AdExParameter(Vr = -50mV),
    spike = SNN.PostSpike(At = 10mV, τA = 10ms),
    synapse = SNN.SingleExpSynapse(),
)
E = SNN.Population(; N = 1, adex_neuron...)

SNN.monitor!(E, [:v, :fire, :w, :θ], sr = 1kHz)

# Define the Poisson stimulus parameters 
poisson_exc = SNN.PoissonLayer(
    rate = 15Hz,    # Mean firing rate (Hz) 
    N = 100,
)
conn = (; p = 1.0f0, μ = 2.0f0, σ = 0.1f0, dist = :Normal, rule = :Fixed)
SNN.clear_records!(model)
stim_exc = SNN.Stimulus(poisson_exc, E, :ge; conn, name = "noiseE")

model = SNN.compose(; E = E, stim_exc, silent = true)

SNN.sim!(; model, duration = 20s)
p1 = vecplot(
    E,
    :v,
    add_spikes = true,
    interval = 200ms:1ms:1800ms;
    ylabel = "Membrane potential (mV)",
    ylims = (-80, 10),
)
vecplot!(
    p1,
    E,
    :θ,
    interval = 200ms:1ms:800ms;
    ylabel = "Membrane potential (mV)",
    ylims = (-80, 10),
    c = :red,
    label = "Spike Threshold (mV)",
)
#
v, r = SNN.record(E, :v, range = true)
st = SNN.spiketimes(E)

p2 = histogram(
    v(1, st[1] .- st[1][1]),
    bins = 20,
    xlabel = "V at threshold (mV)",
    ylabel = "Count",
    label = "",
    c = :black,
    legend = false,
)
plot(p1, p2)
