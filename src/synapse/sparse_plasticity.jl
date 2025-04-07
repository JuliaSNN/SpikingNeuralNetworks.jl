abstract type SpikingSynapseParameter <: AbstractConnectionParameter end
struct no_STDPParameter <: SpikingSynapseParameter end

NoSTDP = no_STDPParameter()

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
include("sparse_plasticity/STDP_structured.jl")

function change_plasticity!(syn, param)
    syn.param = param
    @unpack fireI, fireJ = syn
    Npre, Npost = length(fireJ), length(fireI)
    syn.plasticity = plasticityvariables(param, Npre, Npost)
end

export SpikingSynapse,
    SpikingSynapseParameter,
    no_STDPParameter,
    NoSTDP,
    no_PlasticityVariables,
    plasticityvariables,
    plasticity!
    change_plasticity!
