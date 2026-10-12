# Integrate-and-fire neuron driven by excitatory and inhibitory Poisson input.
using SpikingNeuralNetworks
using CairoMakie
SNN.@load_units

E = SNN.IF(; N = 1)
rate = 50Hz * 300
stim1 = SNN.PoissonStimulus(E, :ge, param = SNN.PoissonFixed(rate = rate, μ = 1.0f0), neurons = :ALL)
stim2 = SNN.PoissonStimulus(E, :gi, param = SNN.PoissonFixed(rate = rate, μ = 5.1f0), neurons = :ALL)
SNN.monitor!(E, [:v, :fire], sr = 1000Hz)
SNN.monitor!(E, [:ge, :gi], sr = 1000Hz, variables = :synvars)

model = SNN.compose(E, sE = stim1, sI = stim2, silent = true)
SNN.sim!(model = model; duration = 10s)

fig = Figure(size = (800, 600))
ax1 = Axis(fig[1, 1], ylabel = "V (mV)")
SNN.vecplot!(ax1, E, :v, interval = 4s:1ms:5s, add_spikes = true)
ax2 = Axis(fig[2, 1], ylabel = "Conductance (nS)", xlabel = "Time (s)")
SNN.vecplot!(ax2, E, :ge, variables = :synvars, interval = 4s:1ms:5s, label = "exc")
SNN.vecplot!(ax2, E, :gi, variables = :synvars, interval = 4s:1ms:5s, label = "inh")
axislegend(ax2)
save(joinpath(mktempdir(), "IF_neuron.png"), fig)
