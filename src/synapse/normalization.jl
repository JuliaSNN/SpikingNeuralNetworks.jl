"""
    Abstract type for normalization parameters.
"""
abstract type NormParam end

"""
    MultiplicativeNorm{FT = Int32} <: NormParam

This struct holds the parameters for multiplicative normalization. 
It includes a timescale τ (default 0.0) and an operator (default multiplication).
"""
MultiplicativeNorm

@snn_kw struct MultiplicativeNorm{FT = Int32} <: NormParam
    τ::Float32 = 0.0f0
    operator::Function = *
end

"""
    AdditiveNorm{FT = Float32} <: NormParam

This struct holds the parameters for additive normalization. 
It includes a timescale τ (default 0.0) and an operator (default addition).
"""
AdditiveNorm

@snn_kw struct AdditiveNorm{FT = Float32} <: NormParam
    τ::Float32 = 0.0f0
    operator::Function = +
end

"""
    SynapseNormalization{VFT = Vector{Float32}, VIT = Vector{Int32}, MFT = Matrix{Float32}}

A struct that holds parameters for synapse normalization, including:
- param: Normalization parameter, can be either MultiplicativeNorm or AdditiveNorm.
- t: A vector of integer values representing time points.
- W0: A vector of initial weights before simulation.
- W1: A vector of weights during the simulation.
- μ: A vector of mean synaptic weights.
- records: A dictionary for storing additional data.
"""
SynapseNormalization

@snn_kw struct SynapseNormalization{
    VFT = Vector{Float32},
    VIT = Vector{Int32},
    MFT = Matrix{Float32},
    VST = Vector{<:AbstractSparseSynapse},
} <: AbstractNormalization
    id::String = randstring(12)
    param::NormParam = MultiplicativeNorm()
    synapses::VST
    t::VIT = [0, 1]
    W0::VFT = [0.0f0]
    W1::VFT = [0.0f0]
    μ::VFT = [0.0f0]
    targets::Dict = Dict()
    records::Dict = Dict()
end

"""
    SynapseNormalization(N; param, kwargs...)

Constructor function for the SynapseNormalization struct.
- N: The number of synapses.
- param: Normalization parameter, can be either MultiplicativeNorm or AdditiveNorm.
- kwargs: Other optional parameters.
Returns a SynapseNormalization object with the specified parameters.
"""
function SynapseNormalization(N, synapses; param::NormParam, kwargs...)
    if !isa(N, Int)
        @unpack N = N
    end
    W0 = zeros(Float32, N)
    W1 = zeros(Float32, N)
    μ = zeros(Float32, N)
    targets = Dict()
    posts = [syn.targets[:post] for syn in synapses]
    @assert length(unique(posts)) == 1
    targets[:post] = unique(posts)[1]
    targets[:synapses] = [syn.id for syn in synapses]
    for syn in synapses
        @assert isa(syn, AbstractSparseSynapse)
        @unpack rowptr, W, index = syn
        Is = 1:(length(rowptr)-1)
        @assert length(Is) == N
        for i in eachindex(Is)
            @simd for j ∈ rowptr[i]:(rowptr[i+1]-1) # all presynaptic neurons connected to neuron 
                W0[i] += W[index[j]]
            end
        end
    end
    SynapseNormalization(; @symdict(param, W0, W1, μ, synapses)..., targets, kwargs...)
end



function forward!(c::SynapseNormalization, param::NormParam) end

"""
    plasticity!(c::SynapseNormalization, param::AdditiveNorm, dt::Float32)

Updates the synaptic weights using additive or multiplicative normalization (operator). This function calculates 
the rate of change `μ` as the difference between initial weight `W0` and the current weight `W1`, 
normalized by `W1`. The weights are updated at intervals specified by time constant `τ`.

# Arguments
- `c`: An instance of SynapseNormalization.
- `param`: An instance of AdditiveNorm.
- `dt`: Simulation time step.
"""
function plasticity!(c::SynapseNormalization, param::NormParam, dt::Float32, T::Time)
    tt = get_step(T)
    @unpack τ = param
    if ((tt) % round(Int, τ / dt)) < dt
        plasticity!(c, param)
    end
end

function plasticity!(c::SynapseNormalization, param::NormParam)
    @unpack W1, W0, μ, synapses = c
    @unpack operator = param
    fill!(W1, 0.0f0)
    for syn in synapses
        @unpack rowptr, W, index = syn
        Threads.@threads for i = 1:(length(rowptr)-1) # Iterate over all postsynaptic neuron
            @inbounds @fastmath @simd for j = rowptr[i]:(rowptr[i+1]-1) # all presynaptic neurons of i
                W1[i] += W[index[j]]
            end
        end
    end
    # normalize
    # @fastmath @inbounds @simd 
    @turbo for i in eachindex(μ)
        μ[i] = (W0[i] - operator(W1[i], 0.0f0)) / W1[i] #operator defines additive or multiplicative norm
    end
    # apply
    for syn in synapses
        @unpack rowptr, W, index = syn
        Threads.@threads for i = 1:(length(rowptr)-1) # Iterate over all postsynaptic neuron
            @inbounds @fastmath @simd for j = rowptr[i]:(rowptr[i+1]-1) # all presynaptic neurons connected to neuron i
                W[index[j]] = operator(W[index[j]], μ[i])
            end
        end
    end
end
