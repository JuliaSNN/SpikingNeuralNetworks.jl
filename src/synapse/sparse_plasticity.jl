abstract type SpikingSynapseParameter <: AbstractSynapseParameter end
abstract type PlasticityVariables end
struct no_STDPParameter <: SpikingSynapseParameter end

## No plasticity
struct no_PlasticityVariables <: PlasticityVariables end

function get_variables(param::no_STDPParameter, Npre, Npost)
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
include("sparse_plasticity/STDP.jl")

export SpikingSynapse,
    SpikingSynapseParameter,
    no_STDPParameter,
    no_PlasticityVariables,
    get_variables,
    plasticity!
