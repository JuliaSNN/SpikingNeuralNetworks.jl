struct LSSPParameter{ST<:SpikingSynapseParameter,LT<:SpikingSynapseParameter} <:
       SpikingSynapseParameter
    long::LT
    short::ST
end
function LSSPParameter(; long, short)
    return LSSPParameter(long, short)
end

struct LSSPVariables{IT,ST<:PlasticityVariables,LT<:PlasticityVariables} <:
       PlasticityVariables
    ## Plasticity variables
    Npost::IT
    Npre::IT
    long::LT
    short::ST
end

function LSSPVariables(; Npre, Npost, long, short)
    return LSSPVariables(Npre, Npost, long, short)
end


function plasticityvariables(param::LSSPParameter, Npre, Npost)
    ls = plasticityvariables(param.long, Npre, Npost)
    ss = plasticityvariables(param.short, Npre, Npost)
    return LSSPVariables(Npre = Npre, Npost = Npost, long = ls, short = ss)
end

"""
    plasticity!(c::AbstractSparseSynapse, param::LSSPParameter, dt::Float32)

    Perform update of synapses using plasticity rules based on the Long Short Term Plasticity (LSSP) model.
    This function updates pre-synaptic spike traces and post-synaptic traces, and modifies synaptic weights using LSSP rules.
"""
function plasticity!(c::AbstractSparseSynapse, param::LSSPParameter, dt::Float32, T::Time)
    plasticity!(c, param.long, c.plasticity.long, dt, T)
    plasticity!(c, param.short, c.plasticity.short, dt, T)
end#======================================================================================#


## Overwrite the record and monitor functions for LSSP, this is necessary because the LSSPVariables contains two different plasticity variables in a single object.

function record_plast!(
    obj::ST,
    plasticity::LSSPVariables,
    key::Symbol,
    T::Time,
    indices::Dict{Symbol,Vector{Int}},
    name_plasticity::Symbol,
) where {ST<:AbstractSparseSynapse}
    nameof(typeof(plasticity.long)) == name_plasticity &&
        (record_plast!(obj, plasticity.long, key, T, indices, name_plasticity))
    nameof(typeof(plasticity.short)) == name_plasticity &&
        (record_plast!(obj, plasticity.short, key, T, indices, name_plasticity))
end

function has_plasticity_field(plasticity::LSSPVariables, key)
    hasfield(typeof(plasticity), key) ||
        hasfield(typeof(plasticity.long), key) ||
        hasfield(typeof(plasticity.short), key)
end

function monitor_plast(obj, plasticity::LSSPVariables, sym)
    (has_plasticity_field(plasticity.long, sym)) &&
        (monitor_plast(obj, plasticity.long, sym))
    (has_plasticity_field(plasticity.short, sym)) &&
        (monitor_plast(obj, plasticity.short, sym))
end


export LSSPParameter, LSSPVariables, plasticityvariables, plasticity!
