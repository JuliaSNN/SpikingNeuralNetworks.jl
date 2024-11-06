
@snn_kw mutable struct SpikeRateSynapse{
    VIT = Vector{Int32},
    VFT = Vector{Float32},
    VBT = Vector{Bool},
} <: AbstractConnection
    id::String = randstring(12)
    param::RateSynapseParameter = RateSynapseParameter()
    colptr::VIT # column pointer of sparse W
    I::VIT      # postsynaptic index of W
    W::VFT  # synaptic weight
    rI::VFT # postsynaptic rate
    # rJ::VFT # presynaptic rate
    Apre::VFT = zero(W) # presynaptic trace
    tpre::VFT = zero(W) # presynaptic spiking time
    fireJ::VBT # presynaptic firing
    g::VFT  # postsynaptic conductance
    targets::Dict = Dict()
    records::Dict = Dict()
end

"""
[Rate Synapse](https://brian2.readthedocs.io/en/2.0b4/resources/tutorials/2-intro-to-brian-synapses.html)
"""
SpikeRateSynapse

function SpikeRateSynapse(pre, post; μ = 0.0, p = 0.0, kwargs...)
    w = μ / √(p * pre.N) * sprandn(post.N, pre.N, p)
    rowptr, colptr, I, J, index, W = dsparse(w)
    rI, fireJ, g = post.r, pre.fire, post.g
    SpikeRateSynapse(; @symdict(colptr, I, W, rI, fireJ, g)..., kwargs...)
end

function forward!(c::SpikeRateSynapse, param::RateSynapseParameter)
    @unpack colptr, I, W, rI, fireJ, g = c
    @unpack lr = param
    # fill!(g, zero(eltype(g)))
    @inbounds for j = 1:(length(colptr)-1)
        if fireJ[j]
            for s = colptr[j]:(colptr[j+1]-1)
                g[I[s]] += W[s]
            end
        end
    end
end

function plasticity!(c::SpikeRateSynapse, param::RateSynapseParameter, dt::Float32, T::Time)
    @unpack colptr, I, W, rI, rJ, g = c
    @unpack lr = param
    @inbounds for j = 1:(length(colptr)-1)
        s_row = colptr[j]:(colptr[j+1]-1)
        rIW = zero(Float32)
        for s in s_row
            rIW += rI[I[s]] * W[s]
        end
        Δ = lr * (rJ[j] - rIW)
        for s in s_row
            W[s] += rI[I[s]] * Δ
        end
    end
end
