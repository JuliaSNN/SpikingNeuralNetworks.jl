using SpikingNeuralNetworks
using CairoMakie
SNN.@load_units

# Create a postsynaptic population
E = SNN.IF(; N = 3, param = SNN.IFParameter())

# Define spike times (ms) and the input neuron of each spike
spiketimes = [100ms, 200ms, 300ms]
neurons = [1, 2, 3]
param = SNN.SpikeTimeParameter(spiketimes, neurons)

# Create a spike time stimulus: input neuron j excites neuron j with weight 10 nS
stim = SNN.SpikeTimeStimulus(E, :ge; param, conn = Matrix(10nS * SNN.SNNModels.LinearAlgebra.I(3)))

# Create a model and simulate
model = SNN.compose(; E, stim, silent = true)
SNN.monitor!(E, [:v, :fire, :syn_curr], sr = 2kHz)
SNN.sim!(; model, duration = 500ms)

# Plot the results
fig = Figure(size = (600, 500))
ax1 = Axis(fig[1, 1], ylabel = "Membrane potential (mV)")
SNN.vecplot!(ax1, E, :v, add_spikes = true)
ax2 = Axis(fig[2, 1], ylabel = "Synaptic current (pA)", xlabel = "Time (s)")
SNN.vecplot!(ax2, E, :syn_curr)
save(joinpath(mktempdir(), "timed_stim.png"), fig)
