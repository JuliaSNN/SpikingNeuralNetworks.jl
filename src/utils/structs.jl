"""
    AbstractParameter

An abstract type representing a parameter.
"""
abstract type AbstractParameter end

"""
    AbstractConnectionParameter <: AbstractParameter

An abstract type representing a connection parameter.
"""
abstract type AbstractConnectionParameter <: AbstractParameter end

"""
    AbstractPopulationParameter <: AbstractParameter

An abstract type representing a population parameter.
"""
abstract type AbstractPopulationParameter <: AbstractParameter end

"""
    AbstractStimulusParameter <: AbstractParameter

An abstract type representing a stimulus parameter.
"""
abstract type AbstractStimulusParameter <: AbstractParameter end

"""
    AbstractConnection

An abstract type representing a connection. Any struct inheriting from this type must implement:

# Methods
- `forward!(c::Synapse, param::SynapseParameter)`: Propagates the signal through the synapse.
- `plasticity!(c::Synapse, param::SynapseParameter, dt::Float32, T::Time)`: Updates the synapse parameters based on plasticity rules.
"""
abstract type AbstractConnection end

"""
    AbstractPopulation

An abstract type representing a population. Any struct inheriting from this type must implement:

# Methods
- `integrate!(p::NeuronModel, param::NeuronModelParam, dt::Float32)`: Integrates the neuron model over a time step `dt` using the given parameters.
"""
abstract type AbstractPopulation end

"""
    AbstractStimulus

An abstract type representing a stimulus. Any struct inheriting from this type must implement:

# Methods
- `stimulate!(p::Stimulus, param::StimulusParameter, time::Time, dt::Float32)`: Applies the stimulus to the population.
"""
abstract type AbstractStimulus end

"""
    AbstractSparseSynapse <: AbstractConnection

An abstract type representing a sparse synapse connection.
"""
abstract type AbstractSparseSynapse <: AbstractConnection end

"""
    AbstractNormalization <: AbstractConnection

An abstract type representing a normalization connection.
"""
abstract type AbstractNormalization <: AbstractConnection end

"""
    PlasticityVariables

An abstract type representing plasticity variables.
"""
abstract type PlasticityVariables end

"""
    Spiketimes

A type alias for a vector of vectors of Float32, representing spike times.
"""
Spiketimes = Vector{Vector{Float32}}

"""
    EmptyParam

A struct representing an empty parameter.

# Fields
- `type::Symbol`: The type of the parameter, default is `:empty`.
"""
EmptyParam

@snn_kw struct EmptyParam
    type::Symbol = :empty
end

"""
    struct Time
    Time

A mutable struct representing time. 
A mutable struct representing time.

# Fields
- `t::Vector{Float32}`: A vector containing the current time.
- `tt::Vector{Int}`: A vector containing the current time step.
- `dt::Float32`: The time step size.

"""
Time

@snn_kw mutable struct Time{VFT = Vector{Float32}, VIT = Vector{Int32}, FT = Float32}
    t::VFT = [0.0f0]
    tt::VIT = Int32[0]
    dt::FT = 0.125f0
end

export Spiketimes, Time
