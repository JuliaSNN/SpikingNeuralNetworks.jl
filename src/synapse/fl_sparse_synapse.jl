struct FLSparseSynapseParameter
end

@snn_kw mutable struct FLSparseSynapse{VFT=Vector{Float32},FT=Float32}
    param::FLSparseSynapseParameter = FLSparseSynapseParameter()
    colptr::Vector{Int32} # column pointer of sparse W
    I::Vector{Int32}      # postsynaptic index of W
    W::VFT  # synaptic weight
    rI::VFT # postsynaptic rate
    rJ::VFT # presynaptic rate
    g::VFT  # postsynaptic conductance
    P::VFT  # <rᵢrⱼ>⁻¹
    q::VFT  # P * r
    u::VFT # force weight
    w::VFT # output weight
    f::FT = 0 # postsynaptic traget
    z::FT = 0.5randn()  # output z ≈ f
    records::Dict = Dict()
end

"""
[Force Learning Sparse Synapse](http://www.theswartzfoundation.org/docs/Sussillo-Abbott-Coherent-Patterns-August-2009.pdf)
"""
FLSparseSynapse

function FLSparseSynapse(pre, post; σ = 1.5, p = 0.0, α = 1, kwargs...)
    w = σ * 1 / √(p * pre.N) * sprandn(post.N, pre.N, p)
    rowptr, colptr, I, J, index, W = dsparse(w)
    rI, rJ, g = post.r, pre.r, post.g
    P = α .* (I .== J)
    q = zeros(post.N)
    u = 2rand(post.N) - 1
    w = 1 / √post.N * (2rand(post.N) - 1)
    FLSparseSynapse(;@symdict(colptr, I, W, rI, rJ, g, P, q, u, w)..., kwargs...)
end

function forward!(c::FLSparseSynapse, param::FLSparseSynapseParameter)
    @unpack W, rI, rJ, g, P, q, u, w, f, z = c
    z = dot(w, rI)
    g .= z .* u
    fill!(q, zero(Float32))
    @inbounds for j in 1:(length(colptr) - 1)
        rJj = rJ[j]
        for s = colptr[j]:(colptr[j+1] - 1)
            i = I[s]
            q[i] += P[s] * rJj
            g[i] += W[s] * rJj
        end
    end
end

function plasticity!(c::FLSparseSynapse, param::FLSparseSynapseParameter, dt::Float32, t::Float32)
    C = 1 / (1 + dot(q, rI))
    BLAS.axpy!(C * (f - z), q, w)
    @inbounds for j in 1:(length(colptr) - 1)
        for s in colptr[j]:(colptr[j+1] - 1)
            P[s] += -C * q[I[s]] * q[j]
        end
    end
end
