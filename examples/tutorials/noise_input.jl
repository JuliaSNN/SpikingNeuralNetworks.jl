

import SNNPlots: vecplot, plot, Plots
using SpikingNeuralNetworks
using Distributions
SNN.@load_units

SNNPlots.default(palette = :okabe_ito)

if_parameter =
    SNN.IFParameter(R = 0.5GΩ, Vt = -50mV, ΔT = 2mV, El = -70mV, τm = 20ms, Vr = -55mV)

# Create the IF neuron with tonic firing parameters
E = SNN.IF(; N = 1, param = if_parameter)
SNN.monitor!(E, [:v, :fire, :w, :I], sr = 2kHz)

# Create a withe noise input current 
current_param = CurrentNoiseParameter(E.N; I_base = 30pA, I_dist = Normal(00pA, 100pA))
current = CurrentStimulus(E, :I, param = current_param)
model = merge_models(; E = E, I = current)
SNN.sim!(; model, duration = 2000ms)

p = plot(
    vecplot(
        E,
        :v,
        add_spikes = true,
        ylabel = "Membrane potential (mV)",
        ylims = (-80, 10),
        c = :black,
    ),
    vecplot(E, :I, ylabel = "External current (pA)", c = :gray, lw = 0.4, alpha = 0.4),
    layout = (2, 1),
    size = (600, 500),
    xlabel = "Time (s)",
    leftmargin = 10Plots.mm,
)

savefig(
    p,
    "/home/user/mnt/zeus/User_folders/aquaresi/network_models/src/SpikingNeuralNetworks.jl/docs/src/assets/examples/noise_current.png",
)

# Create a populations with 2 IF neurons
E = SNN.IF(; N = 2, param = if_parameter)
SNN.monitor!(E, [:v, :fire, :w, :I], sr = 2kHz)

# Create a withe noise input current 

function sinusoidal_current(variables::Dict, t::Float32, i::Int)
    # Extract the parameters from the variables dictionary
    amplitude = variables[:amplitude]
    frequency = variables[:frequency]
    phase = variables[:shift_phase]

    # Calculate the current value at time t for neuron i
    return amplitude * sin(2 * π * frequency * t + i*phase)
end

variables = Dict(
    :amplitude => 50pA,
    :frequency => 1Hz,
    :shift_phase => π*3/4, # Phase shift for each neuron
)

current_param = SNN.CurrentVariableParameter(variables, sinusoidal_current)
current = CurrentStimulus(E, :I, param = current_param)
model = merge_models(; E = E, I = current)
SNN.sim!(; model, duration = 2000ms)

p = plot(
    vecplot(
        E,
        :v,
        add_spikes = true,
        ylabel = "Membrane potential (mV)",
        ylims = (-80, 10),
    ),
    vecplot(E, :I, ylabel = "External current (pA)", c = :gray, lw = 0.4, alpha = 0.4),
    layout = (2, 1),
    size = (600, 500),
    xlabel = "Time (s)",
    leftmargin = 10Plots.mm,
)

savefig(
    p,
    "/home/user/mnt/zeus/User_folders/aquaresi/network_models/src/SpikingNeuralNetworks.jl/docs/src/assets/examples/variable_current.png",
)
