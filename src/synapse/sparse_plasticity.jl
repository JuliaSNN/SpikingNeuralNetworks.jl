abstract type SpikingSynapseParameter <: AbstractConnectionParameter end
struct no_STDPParameter <: SpikingSynapseParameter end

## No plasticity
struct no_PlasticityVariables <: PlasticityVariables end

function plasticityvariables(param::no_STDPParameter, Npre, Npost)
    return no_PlasticityVariables()
end

function plasticity!(
    c::AbstractSparseSynapse,
    param::no_STDPParameter,
    dt::Float32,
    T::Time,
) end
##

include("sparse_plasticity/vSTDP.jl")
include("sparse_plasticity/iSTDP.jl")
include("sparse_plasticity/STP.jl")
include("sparse_plasticity/longshortSP.jl")
include("sparse_plasticity/STDP.jl")

export SpikingSynapse,
    SpikingSynapseParameter,
    no_STDPParameter,
    no_PlasticityVariables,
    plasticityvariables,
    plasticity!
