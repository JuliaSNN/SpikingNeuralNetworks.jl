@snn_kw struct RateSynapseParameter{FT=Float32}
    lr::FT = 1e-3
end

@snn_kw mutable struct RateSynapse{VIT=Vector{Int32},VFT=Vector{Float32}}
    param::RateSynapseParameter = RateSynapseParameter()
    colptr::VIT # column pointer of sparse W
    I::VIT      # postsynaptic index of W
    W::VFT  # synaptic weight
    rI::VFT # postsynaptic rate
    rJ::VFT # presynaptic rate
    g::VFT  # postsynaptic conductance
    records::Dict = Dict()
end

"""
[Rate Synapse](https://brian2.readthedocs.io/en/2.0b4/resources/tutorials/2-intro-to-brian-synapses.html)
"""
RateSynapse

function RateSynapse(pre, post; σ = 0.0, p = 0.0, kwargs...)
    w = σ / √(p * pre.N) * sprandn(post.N, pre.N, p)
    rowptr, colptr, I, J, index, W = dsparse(w)
    rI, rJ, g = post.r, pre.r, post.g
    RateSynapse(;@symdict(colptr, I, W, rI, rJ, g)..., kwargs...)
end

function forward!(c::RateSynapse, param::RateSynapseParameter)
    @unpack colptr, I, W, rI, rJ, g = c
    @unpack lr = param
    fill!(g, zero(eltype(g)))
    @inbounds for j in 1:(length(colptr) - 1)
        rJj = rJ[j]
        for s = colptr[j]:(colptr[j+1] - 1)
            g[I[s]] += W[s] * rJj
        end
    end
end

function plasticity!(c::RateSynapse, param::RateSynapseParameter, dt::Float32, t::Float32)
    @unpack colptr, I, W, rI, rJ, g = c
    @unpack lr = param
    @inbounds for j in 1:(length(colptr) - 1)
        s_row = colptr[j]:(colptr[j+1] - 1)
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
