
abstract type AbstractSpikingSynapse <: AbstractSparseSynapse end

function synaptic_target(
    targets::Dict,
    post::T,
    sym::Symbol,
    target,
) where {T<:AbstractPopulation}
    g = zeros(Float32, post.N)
    v_post = zeros(Float32, post.N)
    if isnothing(target)
        g = getfield(post, sym)
        _v = :v
        hasfield(typeof(post), _v) && (v_post = getfield(post, _v))
        push!(targets, :sym => sym)
    elseif typeof(target) == Symbol
        _sym = Symbol("$(sym)_$target")
        _v = Symbol("v_$target")
        g = getfield(post, _sym)
        hasfield(typeof(post), _v) && (v_post = getfield(post, _v))
        push!(targets, :sym => _sym)
    elseif typeof(target) == Int
        if typeof(post) <: AbstractDendriteIF
            _sym = Symbol("$(sym)_d")
            _v = Symbol("v_d")
            g = getfield(post, _sym)[target]
            v_post = getfield(post, _v)[target]
            push!(targets, :sym => Symbol(string(_sym, target)))
        elseif isa(post, AdExMultiTimescale)
            g = getfield(post, sym)[target]
            v_post = getfield(post, :v)
            push!(targets, :sym => Symbol(string(sym, target)))
        end
    end
    return g, v_post
end

function synaptic_target(
    targets::Dict,
    post::T,
    sym::Nothing,
    target,
) where {T<:AbstractPopulation}
    @warn "Synaptic target not instatiated, returning non-pointing arrays"
    g = zeros(Float32, post.N)
    v = zeros(Float32, post.N)
    return g, v
end
synaptic_target(post::T, sym::Symbol, target) where {T<:AbstractPopulation} =  synaptic_target(Dict(), post, sym, target) 

include("empty.jl")
include("normalization.jl")
include("aggregate_scaling.jl")
include("rate_synapse.jl")
include("fl_synapse.jl")
include("fl_sparse_synapse.jl")
include("pinning_synapse.jl")
include("pinning_sparse_synapse.jl")
include("spike_rate_synapse.jl")

struct SpikingSynapseParameter <: AbstractConnectionParameter end
include("sparse_plasticity.jl")
include("spiking_synapse.jl")
