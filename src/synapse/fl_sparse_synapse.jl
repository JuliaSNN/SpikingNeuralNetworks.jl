struct FLSparseSynapseParameter end

@snn_kw mutable struct FLSparseSynapse{VFT = Vector{Float32},FT = Float32} <:
                       AbstractConnection
    name::String = "FLSparseSynapse"
    id::String = randstring(12)
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
    targets::Dict = Dict()
    records::Dict = Dict()
end

"""
[Force Learning Sparse Synapse](http://www.theswartzfoundation.org/docs/Sussillo-Abbott-Coherent-Patterns-August-2009.pdf)
"""
FLSparseSynapse

function FLSparseSynapse(pre, post; μ = 1.5, p = 0.0, α = 1, kwargs...)
    w = μ * 1 / √(p * pre.N) * sprandn(post.N, pre.N, p)
    rowptr, colptr, I, J, index, W = dsparse(w)
    rI, rJ, g = post.r, pre.r, post.g
    P = α .* (I .== J)
    q = zeros(post.N)
    u = 2rand(post.N) - 1
    w = 1 / √post.N * (2rand(post.N) - 1)

    targets = Dict{Symbol,Any}(
        :fire => pre.id,
        :post => post.id,
        :pre => pre.id,
        :type=>:FLSynapseSparse,
    )
    @views g, v_post = synaptic_target(targets, post, :g, nothing)

    FLSparseSynapse(;
        @symdict(colptr, I, W, rI, rJ, g, P, q, u, w)...,
        kwargs...,
        targets = targets,
    )
end

function forward!(c::FLSparseSynapse, param::FLSparseSynapseParameter)
    @unpack W, rI, rJ, g, P, q, u, w, f, z = c
    c.z = dot(w, rI)
    g .= c.z .* u
    fill!(q, zero(Float32))
    @inbounds for j = 1:(length(colptr)-1)
        rJj = rJ[j]
        for s = colptr[j]:(colptr[j+1]-1)
            i = I[s]
            q[i] += P[s] * rJj
            g[i] += W[s] * rJj
        end
    end
end

function plasticity!(
    c::FLSparseSynapse,
    param::FLSparseSynapseParameter,
    dt::Float32,
    T::Time,
)
    @unpack rI, P, q, w, f, z = c
    C = 1 / (1 + dot(q, rI))
    BLAS.axpy!(C * (f - z), q, w)
    @inbounds for j = 1:(length(colptr)-1)
        for s = colptr[j]:(colptr[j+1]-1)
            P[s] += -C * q[I[s]] * q[j]
        end
    end
end
