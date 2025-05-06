abstract type PlasticityParameter end

abstract type LTPVariables <: PlasticityVariables end
abstract type LTPParameter <: PlasticityParameter end
abstract type STPVariables <: PlasticityVariables end
abstract type STPParameter <: PlasticityParameter end

@snn_kw struct NoLTP{VB = Vector{Bool}} <: LTPParameter 
    active::VB = [false]
end
@snn_kw struct NoSTP{VB = Vector{Bool}} <: STPParameter 
    active::VB = [false]
end
@snn_kw struct NoVariables{VB = Vector{Bool}} <: PlasticityVariables 
    active::VB = [false]
end

struct LTP <:LTPParameter end
struct STP <:STPParameter end


plasticityvariables(param::NoLTP, Npre, Npost) =  NoVariables()
plasticityvariables(param::NoSTP, Npre, Npost) =  NoVariables()

function plasticity!(
    c::PT,
    param::SpikingSynapseParameter,
    dt::Float32,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    any(c.STPVars.active) && plasticity!(c, c.STPParam, c.STPVars, dt, T)
    any(c.LTPVars.active) && plasticity!(c, c.LTPParam, c.LTPVars, dt, T)
end

function set_plasticity!(c::AbstractSparseSynapse, param::LTPParameter, state::Bool)
    c.LTPVars.active .= state
end

function set_plasticity!(c::AbstractSparseSynapse, param::STPParameter, state::Bool)
    c.STPVars.active .= state
end


function plasticity!(
    c::AbstractSparseSynapse,
    param::PT,
    variables::NoVariables,
    dt::Float32,
    T::Time,
) where{PT<:PlasticityParameter}
end


## STP
include("sparse_plasticity/STP.jl")

## STDP
abstract type STDPParameter <: LTPParameter end
NoSTDP = NoLTP()
include("sparse_plasticity/vSTDP.jl")
include("sparse_plasticity/iSTDP.jl")
# include("sparse_plasticity/longshortSP.jl")
include("sparse_plasticity/STDP_traces.jl")
include("sparse_plasticity/STDP_structured.jl")

function change_plasticity!(syn; LTP=nothing, STP=nothing)
    @unpack fireI, fireJ = syn
    Npre, Npost = length(fireJ), length(fireI)
    if !isnothing(LTP)
        syn.LTPParam = LTP
        syn.LTPVars = plasticityvariables(param, Npre, Npost)
    end
    if !isnothing(STP)
        syn.STPParam = STP
        syn.STPVars = plasticityvariables(param, Npre, Npost)
    end
end

export SpikingSynapse,
    SpikingSynapseParameter,
    no_STDPParameter,
    NoSTDP,
    no_PlasticityVariables,
    plasticityvariables,
    plasticity!
    change_plasticity!

export LTP, STP, NoLTP, NoSTP
