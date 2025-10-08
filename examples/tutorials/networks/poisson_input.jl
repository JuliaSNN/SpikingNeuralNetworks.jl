import SpikingNeuralNetworks: vecplot, plot, savefig, gplot, Plots
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
poisson_exc = SNN.PoissonLayer(
    rate=1Hz,    # Mean firing rate (Hz) 
    N = 1000, # Neurons in the Poisson Layer
)

proj_exc = (
    p = 1,
    μ = 2nS
)
poisson_inh = SNN.PoissonLayer(
    rate = 6Hz,       # Mean firing rate (Hz)
    N = 1000,  # Neurons in the Poisson Layer
)

proj_inh = (p = 1.0f0,   # Probability of connecting to a neuron
            μ = 4.0,   # Synaptic strength (nS)
)

# Create the Poisson layers for excitatory and inhibitory inputs
stim_exc = SNN.Stimulus(poisson_exc, E, :ge, name = "noiseE", conn=proj_exc)
stim_inh = SNN.Stimulus(poisson_inh, E, :gi, name = "noiseI", conn=proj_inh)

# Create the model and run the simulation
model = SNN.compose(; E = E, stim_exc, stim_inh)
SNN.monitor!(E, [:v, :fire, :w, :ge, :gi], sr = 2kHz)
SNN.monitor!(model.stim, [:fire])
SNN.sim!(; model, duration = 1000ms)

# Plot the results
# gplot is a special function the plots the synaptic currents

Plots.default(palette = :okabe_ito)
p = plot(
    SNN.raster(model.stim),
    SNN.gplot(
        E,
        v_sym = :v,
        ge_sym = :ge,
        gi_sym = :gi,
        Ee_rev = 0mV,
        Ei_rev = -75mV,
        ylabel = "Synapti current (μA)",
    ),
    SNN.vecplot(
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
    leftmargin = 10Plots.mm,
)
##


savefig(
    p,
    "/home/user/mnt/helix/User_folders/aquaresi/network_models/src/SpikingNeuralNetworks.jl/docs/src/assets/examples/poisson_input.png",
)

p
