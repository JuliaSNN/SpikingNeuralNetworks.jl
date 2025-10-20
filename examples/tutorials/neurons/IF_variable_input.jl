# Create a populations with 2 IF neurons
E = SNN.Population(; N = 2, if_neuron...)
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
current_stim = SNN.CurrentStimulus(E, :I, param = current_param)
model = SNN.compose(; E = E, I = current_stim)
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

# savefig(
#     p,
#     "/home/user/mnt/helix/User_folders/aquaresi/network_models/src/SpikingNeuralNetworks.jl/docs/src/assets/examples/variable_current.png",
# )
