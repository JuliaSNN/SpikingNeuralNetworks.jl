using Pkg
Pkg.add(url="https://github.com/JuliaSNN/SpikingNeuralNetworks.jl")
Pkg.add("Distributions")
##
using SpikingNeuralNetworks
using Distributions
SNN.@load_units

# The macro @eval is used to define the new neuron model within the SNNModels module. It is equivalent to add a new file in the SNNModels.jl/src/populations directory. We strongly suggest this approach to avoid complications with the module system.
@eval SNN.SNNModels begin

    """
    Define the neuron model parameters. 
    Parameters are used at integration time to compute the equation update. 
    All parameters are optional. We strongly advise using SI units and default values.
    """
    NeuronParameter
    @snn_kw struct NeuronParameter <: AbstractPopulationParameter
        # adex parameters
        R::Float32 = 1f0GΩ
        Er::Float32 = -70.6f0
        Vt::Float32 = -50.4f0
        up::Float32 = 0.1f0 * ms
        τabs::Float32 = 0.1f0 * ms
        τe::Float32 = 10f0ms
        τi::Float32 = 10f0ms
    end

    """
    Define the neuron model. 
    The neuron model holds the parameters and state variables of the neuron.
    The state variables are used to compute the equation update at integration time and can be recorded.

    The entries:
     - `param::NeuronParameter`: The parameters of the neuron model.
     - `N::Int64`: The number of neurons in the population.
     - `name::String`: The name of the neuron population.
     - `id::String`: A unique identifier for the neuron population.
     - `records::Dict{Symbol, Any}`: A dictionary to store recorded variables.
    are compulsory
    """
    Neuron 

    @snn_kw struct Neuron <: AbstractPopulation
        param::NeuronParameter = NeuronParameter()
        N::Int64 = 10
        name::String= "Neuron"
        id::String = randstring(12)
        v::Vector{Float32} = ones(Float32, N)*-70.6f0  # Initial membrane potential
        ge::Vector{Float32} = zeros(Float32, N)
        gi::Vector{Float32} = zeros(Float32, N)
        fire::Vector{Bool} = falses(N)
        I::Vector{Float32} = zeros(Float32, N)
        records::Dict{Symbol, Any} = Dict{Symbol, Any}()
    end

    """
    Integrate the neuron model.
    The function integrate!(p::Neuron, param::NeuronParameter, dt::Float32) is mandatory for the a Population model
    and is used to update the state variables of the neuron model at each time step.

    The function must only define the integration step, the recordings are handled by the simulation engine. 
    The following present a good practice to implement the integration step:
    - Use the `@unpack` macro to extract the state variables from the neuron model.
    - Use the `@inbounds` macro to skip bounds checking for performance reasons.
    - Update the state variables in a for loop over the number of neurons `N`.
    - Update the state variables using the timestep `dt` and the parameters from `param`. 

    The macro `@inbounds` is used to skip bounds checking for performance reasons. It leads to segment faults if the indices are out of bounds.
    
    The macro `@fastmath` is used to allow the compiler to use fast math operations, which may lead to slight inaccuracies but improves performance. We consider that in the context of biophysical networks this imprecisions are not critical.
    
    """
    integrate!

    function integrate!(p::Neuron, param::NeuronParameter, dt::Float32)
        @unpack N, v, ge, gi, fire, I = p
        @inbounds @fastmath for i in 1:N
            if fire[i]
                v[i] = param.Er 
                fire[i] = false
            else
                v[i] += dt*(param.Er - v[i] + 
                        (ge[i] - gi[i]) * param.R + 
                        I[i] * param.R )
            end
            ge[i] -= ge[i] / param.τe * dt
            gi[i] -= gi[i] / param.τi * dt
            if v[i] >= param.Vt
                fire[i] = true
                v[i] = 20*mV  # Reset membrane potential after firing
            end
        end
    end
    export Neuron, NeuronParameter, integrate!
end

import SpikingNeuralNetworks: NeuronParameter, Neuron, CurrentNoiseParameter, CurrentStimulus, compose, sim!, monitor!, vecplot
# validate_population_model(SNN.Neuron()) # This is only available in SNNModels v1.5.5

param = NeuronParameter()
neuron = Neuron(param=param, N=1)
# Create a withe noise input current 
current_param = CurrentNoiseParameter(neuron.N; I_base = 0pA, I_dist = Normal(-50pA, 100pA))
current_stim = CurrentStimulus(neuron, :I, param = current_param)

monitor!(neuron, [:v, :fire, :ge, :gi, :I], sr = 2kHz)
model = compose(; neuron, current_stim)
sim!(; model, duration = 1000ms, pbar=true)

vecplot(
    neuron,
    :v,
    add_spikes = true,
    ylabel = "Membrane potential (mV)",
    # ylims = (-80, 10),
    c = :black,
)