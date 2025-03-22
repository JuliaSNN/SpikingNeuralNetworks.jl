abstract type AbstractParameter end
abstract type AbstractConnectionParameter <: AbstractParameter end
abstract type AbstractPopulationParameter <: AbstractParameter end
abstract type AbstractStimulusParameter <: AbstractParameter end
abstract type AbstractConnection end
abstract type AbstractPopulation end
abstract type AbstractStimulus end

abstract type AbstractSparseSynapse <: AbstractConnection end
abstract type AbstractNormalization <: AbstractConnection end
abstract type PlasticityVariables end

Spiketimes = Vector{Vector{Float32}}

@snn_kw struct EmptyParam
    type::Symbol = :empty
end

"""
    struct Time

A mutable struct representing time. 

# Fields
- `t::Vector{Float32}`: A vector containing the current time.
- `tt::Vector{Int}`: A vector containing the current time step.
- `dt::Float32`: The time step size.

"""
Time
@snn_kw mutable struct Time{VFT = Vector{Float32},VIT = Vector{Int32},FT = Float32}
    t::VFT = [0.0f0]
    tt::VIT = Int32[0]
    dt::FT = 0.125f0
end


export Spiketimes, Time
