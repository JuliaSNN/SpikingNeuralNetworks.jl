using SpikingNeuralNetworks
import SNNPlots: vecplot, vecplot!, plot, histogram
SNN.@load_units

## Inspired by
# https://brian2.readthedocs.io/en/stable/examples/adaptive_threshold.html

E = SNN.AdEx(; N = 1, param = SNN.AdExParameter(At=14, Vr = -50mV))
SNN.monitor!(E, [:v, :fire, :w, :θ], sr = 1kHz)

# Define the Poisson stimulus parameters 
poisson_exc = SNN.PoissonStimulusLayer(
    500.2Hz,    # Mean firing rate (Hz) 
    p = 1.0f0,  # Probability of connecting to a neuron
    μ = 1.0,  # Synaptic strength (nS)
    N = 10, # Neurons in the Poisson Layer
)

stim_exc = SNN.PoissonLayer(E, :ge, param = poisson_exc, name = "noiseE")

model = SNN.compose(; E = E, stim_exc, silent = true)

SNN.sim!(; model, duration = 20s)
p1 = vecplot(
    E,
    :v,
    add_spikes = true,
    ylabel = "Membrane potential (mV)",
    ylims = (-80, 10),
)
vecplot!(p1,
    E,
    :θ,
    interval = 0:1ms:800ms;
    ylabel = "Membrane potential (mV)",
    ylims = (-80, 10),
)
#
v, r = SNN.record(E, :v, range=true)
st = SNN.spiketimes(E)

p2 = histogram(v[1,st[1].-1ms], 
    bins = 20,
    xlabel = "V at threshold (mV)",
    ylabel = "Count",
    label = "",
    c = :black,
    legend = false,
)
plot(p1, p2)