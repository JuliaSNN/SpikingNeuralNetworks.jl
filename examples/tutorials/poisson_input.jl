using SNNPlots
import SNNPlots: vecplot, plot, savefig, gplot
using SpikingNeuralNetworks
SNN.@load_units

if_parameter = SNN.IFParameter(
    R = 0.5GΩ,
    Vt = -50mV,
    ΔT = 2mV,
    El = -70mV,
    τm = 20ms,
    Vr = -55mV,
    E_i = -75mV,
    E_e = 0mV,
)

# Create the IF neuron
# E = SNN.AdEx(; N = 1, 
#     # param=if_parameter,
#     )
E = SNN.IF(; N = 1, param = if_parameter)

# Create an excitatory and inhibitory spike trains

# Define the Poisson stimulus parameters 
poisson_exc = SNN.PoissonStimulusLayer(
    1.2Hz,    # Mean firing rate (Hz) 
    p = 1.0f0,  # Probability of connecting to a neuron
    μ = 1.0,  # Synaptic strength (nS)
    N = 1000, # Neurons in the Poisson Layer
)

poisson_inh = SNN.PoissonStimulusLayer(
    3Hz,       # Mean firing rate (Hz)
    p = 1.0f0,   # Probability of connecting to a neuron
    μ = 4.0,   # Synaptic strength (nS)
    N = 1000,  # Neurons in the Poisson Layer
)

# Create the Poisson layers for excitatory and inhibitory inputs
stim_exc = PoissonLayer(E, :ge, param = poisson_exc, name = "noiseE")
stim_inh = PoissonLayer(E, :gi, param = poisson_inh, name = "noiseI")

# Create the model and run the simulation
model = merge_models(; E = E, stim_exc, stim_inh)
SNN.monitor!(E, [:v, :fire, :w, :ge, :gi], sr = 2kHz)
SNN.monitor!(model.stim, [:fire])
SNN.sim!(; model, duration = 1000ms)

# Plot the results
# gplot is a special function the plots the synaptic currents

SNNPlots.default(palette = :okabe_ito)
p = plot(
    raster(model.stim),
    gplot(
        E,
        v_sym = :v,
        ge_sym = :ge,
        gi_sym = :gi,
        Ee_rev = 0mV,
        Ei_rev = -75mV,
        ylabel = "Synapti current (μA)",
    ),
    vecplot(
        E,
        :v,
        add_spikes = true,
        ylabel = "Membrane potential (mV)",
        ylims = (-80, 10),
        c = :black,
    ),
    layout = (3, 1),
    fgcolorlegend = :transparent,
    size = (800, 900),
    xlabel = "Time (s)",
    leftmargin = 10SNNPlots.Plots.mm,
)
#


# using StatsBase
# v, r = SNN.record(E,:v, range=true)

# aa = autocor(v[1,r], 1:100)
# plot(r[1:100], aa)

savefig(
    p,
    "/home/user/mnt/zeus/User_folders/aquaresi/network_models/src/SpikingNeuralNetworks.jl/docs/src/assets/examples/poisson_input.png",
)

p
