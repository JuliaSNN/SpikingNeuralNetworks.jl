# AdEx neuron with an adaptive spike threshold (PostSpike with At > 0), driven by Poisson input.
# Inspired by https://brian2.readthedocs.io/en/stable/examples/adaptive_threshold.html
using SpikingNeuralNetworks
using CairoMakie
SNN.@load_units

adex_neuron = (
    param = SNN.AdExParameter(Vr = -50mV),
    spike = SNN.PostSpike(At = 10mV, τA = 10ms),
    synapse = SNN.SingleExpSynapse(),
)
E = SNN.Population(; N = 1, adex_neuron...)
SNN.monitor!(E, [:v, :fire, :w, :θ], sr = 1kHz)

poisson_exc = SNN.PoissonLayer(rate = 15Hz, N = 100)
conn = (; p = 1.0f0, μ = 2.0f0, σ = 0.1f0, dist = :Normal, rule = :Fixed)
stim_exc = SNN.Stimulus(poisson_exc, E, :ge; conn, name = "noiseE")
model = SNN.compose(; E = E, stim_exc, silent = true)
SNN.sim!(; model, duration = 20s)

v, r = SNN.record(E, :v, range = true)
st = SNN.spiketimes(E)

fig = Figure(size = (900, 400))
ax1 = Axis(fig[1, 1], ylabel = "Membrane potential (mV)", xlabel = "Time (s)")
SNN.vecplot!(ax1, E, :v, add_spikes = true, interval = 200ms:1ms:1800ms, color = :black)
SNN.vecplot!(ax1, E, :θ, interval = 200ms:1ms:1800ms, color = :red, label = "threshold")
axislegend(ax1)
ax2 = Axis(fig[1, 2], xlabel = "Threshold at spike (mV)", ylabel = "Count")
θ = SNN.record(E, :θ)
hist!(ax2, θ(1, st[1] .- 1ms), bins = 20, color = :black)
save(joinpath(mktempdir(), "AdEx_threshold.png"), fig)
