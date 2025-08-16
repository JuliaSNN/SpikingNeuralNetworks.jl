# Models Extensions

Users can define new concrete types of the three abstract models (`AbstractPopulation`, `AbstractStimulus`, and `AbstractSynapse`) to extend the functionality of the SpikingNeural Networks package.

New populations, stimuli, or synapses models can be added by users by defining new types.

## Adding a Population Model

To add a new population model, users need to define a new concrete type that inherits from `AbstractPopulation`. The new population model should include the following:

1. **Parameters**: Define a new type for the parameters of the population model. This type should inherit from `AbstractPopulationParameter`.
2. **State Variables**: Define the state variables of the population model. These variables should be included in the new type that inherits from `AbstractPopulation`.
3. **Integration Function**: Define a `integrate(population::P, param::T, dt::Float32) where {P<:AbstractPopulation, T<:AbstractPopulationParameter}` function to integrate the population model. This function should update the state variables of the population model at each time step.

### Current-based IF 


```julia src/SpikingNeuralNetworks.jl/examples/tutorials/extensions/neuron_model.jl
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
```

## Adding a Stimulus Model

Thanks to the multidispatching the simulation loop will call the function that matches the `population`, `stimulus`, or `connection` type and its parameter. Thus we don't need to always define a new type, defining a new parameter and a function that specializes for it is sufficient to introduce a new behaviour.

Here we extend the `PoissonStimulus<:AbstractStimulus` adding a new `PoissonRefractoryParameter<:AbstractStimulusParameter`. We use the function `PoissonLayer` to create an input layer that stimulate the postsynaptic population with Poisson distributed spikes with a ΔT absolute refractory period. 

### Poisson Stimulus with refractory time

```julia
using SpikingNeuralNetworks
using Distributions
SNN.@load_units
using ProtoStructs

# The macro @eval is used to define the new neuron model within the SNNModels module. It is equivalent to add a new file in the SNNModels.jl/src/populations directory. We strongly suggest this approach to avoid complications with the module system.
@eval SNN.SNNModels begin

    """
    Define the Poisson refractory stimulus parameters.
    Parameters are used at integration time to compute the equation update.
    All parameters are optional. 
    """
    PoissonRefractoryParameter

    @snn_kw struct PoissonRefractoryParameter <: PoissonStimulusParameter #{R} where {R<:Float32}
    # @proto struct PoissonRefractoryParameter{R = Float32} 
        ΔT::Float32 = 2f0ms  # Absolute refractory period
        N::Int = 100  # Number of neurons
        rate::Float32 = 10Hz
        last_spike::Vector{Float32} = zeros(Float32, N)  # Last spike time for each neuron
        rates::Vector{Float32} = fill(rate, N)  # Firing rate for each neuron
        p::Float32 = 0.1f0  # Fraction of neurons receiving the stimulus
        μ::Float32 = 1f0  # Mean of the weight distribution
        σ::Float32 = 0f0  # Standard deviation of the weight distribution
        active::Vector{Bool} = [true]  # Active neurons
    end

    """
    Generate a Poisson stimulus with an absolute refractory period for a postsynaptic population.
    """
    function stimulate!(
        p::PoissonStimulus,
        param::PoissonRefractoryParameter,
        time::Time,
        dt::Float32,
    )
        @unpack N, randcache, fire, neurons, colptr, W, I, g = p
        @unpack rates, ΔT, last_spike = param
        current_time = get_time(time)
        rand!(randcache)
        @inbounds @simd for j = 1:N
            if (current_time - last_spike[j]) > ΔT && randcache[j] < rates[j] * dt
                fire[j] = true
                last_spike[j] = current_time
                @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                    g[I[s]] += W[s]
                end
            else
                fire[j] = false
            end
        end
    end
    export PoissonRefractoryParameter, stimulate!
end

import SpikingNeuralNetworks: PoissonLayer, PoissonRefractoryParameter, compose, sim!, monitor!, vecplot
# validate_population_model(SNN.Neuron()) # This is only available in SNNModels v1.5.5

neuron_param = SNN.IdentityParam()
neuron = SNN.Identity(; param = neuron_param, N = 1, name = "Identity Neuron")

# Create a withe noise input current
stim_param = PoissonRefractoryParameter(N=1, ΔT = 20ms, p=1)
stim = PoissonLayer(neuron, :g; param=stim_param)

monitor!(neuron, [:g, :fire], sr = 2kHz)
model = compose(; neuron, stim)
sim!(; model, duration = 100000ms, pbar=true)

vecplot(
    neuron,
    :g,
    neurons=1,
    add_spikes = true,
    ylabel = "Membrane potential (mV)",
    xlims = (0, 1000ms),
    # ylims = (-80, 10),
    c = :black,
)

st = SNN.spiketimes(neuron)[1]
diff(st) |> x-> SNNPlots.histogram(x, bins=100)
```