abstract type AbstractGeneralizedIFParameter <: AbstractPopulationParameter end
abstract type AbstractGeneralizedIF <: AbstractPopulation end

integrate!(p::AbstractPopulation, param::AbstractPopulationParameter, dt::Float32) = nothing
plasticity!(
    p::AbstractPopulation,
    param::AbstractPopulationParameter,
    dt::Float32,
    T::Time,
) = nothing

include("synapse.jl")
include("synapse_parameters.jl")

## Neurons
include("poisson.jl")
include("iz.jl")
include("hh.jl")
include("rate.jl")
include("identity.jl")

## IF
abstract type AbstractIFParameter <: AbstractGeneralizedIFParameter end
include("if/noisy_if.jl")
include("if/if.jl")
include("if/if_current.jl")
include("if/extendedLIF.jl")
include("if/if_CANAHP.jl")

## AdEx
abstract type AbstractAdExParameter <: AbstractGeneralizedIFParameter end
include("adex/adExParameter.jl")
include("adex/adEx.jl")
include("adex/adExMultiTimescale.jl")

function AdExNeuron(; param::T, kwargs...) where {T<:AbstractAdExParameter}
    if isa(param, AdExSynapseParameter)
        for n in eachindex(param.syn)
            param.α[n] = param.syn[n].α
        end
        return AdExSynapse(; param = param, kwargs...)
    elseif isa(param, AdExMultiTimescaleParameter)
        return AdExMultiTimescale(; param = param, kwargs...)
    else
        return AdExSimple(; param = param, kwargs...)
    end
end

## Multicompartment
abstract type AbstractDendriteIF <: AbstractGeneralizedIF end
include("multicompartment/adex_soma.jl")
include("multicompartment/dendrite.jl")
include("multicompartment/tripod.jl")
include("multicompartment/ballandstick.jl")
include("multicompartment/multipod.jl")


export AbstractDendriteIF, AbstractGeneralizedIF, AbstractGeneralizedIFParameter
