abstract type AbstractGeneralizedIFParameter <: AbstractPopulationParameter end
abstract type AbstractGeneralizedIF <: AbstractPopulation end
integrate!(p::AbstractPopulation, param::AbstractPopulationParameter, dt::Float32) = nothing
plasticity!(p::AbstractPopulation, param::AbstractPopulationParameter, dt::Float32, T::Time) = nothing

include("noisy_if.jl")
include("poisson.jl")
include("iz.jl")
include("hh.jl")
include("rate.jl")
include("identity.jl")

abstract type AbstractIFParameter <: AbstractGeneralizedIFParameter end
include("if.jl")
include("if_current.jl")

abstract type AbstractAdExParameter <: AbstractGeneralizedIFParameter end
include("adEx.jl")

abstract type AbstractDendriteIF <: AbstractGeneralizedIF end
include("multicompartment/tripod_params.jl")
include("multicompartment/dendrite.jl")
include("multicompartment/tripod.jl")
include("multicompartment/ballandstick.jl")
