using SpikingNeuralNetworks
using Plots
@load_units

# Create a postsynaptic population
E = IF(; N = 3, param = IFParameter())

# Define spike times and neuron indices
spiketimes = [0.1, 0.2, 0.3]  # in seconds
neurons = [1, 2, 3]            # neuron indices
param = SpikeTimeParameter(spiketimes, neurons)

# Create a spike time stimulus
stim = SpikeTimeStimulus(E, :ge, param=param, p=1, μ=1)

# Create a model and simulate
model = compose(; E = E, stim)
monitor!(E, [:v, :fire, :ge], sr = 2kHz)
sim!(; model, duration = 1s)

# Plot the results
p = plot(
    vecplot(E, :v, add_spikes=true, ylabel="Membrane potential (mV)", ylims=(-80, 10), c=:black),
    vecplot(E, :ge, ylabel="Synaptic conductance (nS)", c=:gray, lw=0.4, alpha=0.4),
    layout=(2, 1),
    size=(600, 500),
    xlabel= "Time (s)",
    leftmargin=10Plots.mm,
)