struct PINningSparseSynapseParameter end

@snn_kw mutable struct PINningSparseSynapse{VIT = Vector{Int32},VFT = Vector{Float32}} <:
                       AbstractConnection
    name::String = "PINningSparseSynapse"
    id::String = randstring(12)
    param::PINningSparseSynapseParameter = PINningSparseSynapseParameter()
    colptr::VIT # column pointer of sparse W
    I::VIT      # postsynaptic index of W
    W::VFT  # synaptic weight
    rI::VFT # postsynaptic rate
    rJ::VFT # presynaptic rate
    g::VFT  # postsynaptic conductance
    P::VFT  # <rᵢrⱼ>⁻¹
    q::VFT  # P * r
    f::VFT  # postsynaptic traget
    targets::Dict = Dict()
    records::Dict = Dict()
end

"""
[PINing Sparse Synapse](https://www.ncbi.nlm.nih.gov/pubmed/26971945)
"""
PINningSparseSynapse

function PINningSparseSynapse(pre, post; μ = 1.5, p = 0.0, α = 1, kwargs...)
    w = μ / √(p * pre.N) * sprandn(post.N, pre.N, p)
    rowptr, colptr, I, J, index, W = dsparse(w)
    rI, rJ = post.r, pre.r
    P = α .* (I .== J)
    f, q = zeros(post.N), zeros(post.N)

    targets = Dict{Symbol,Any}(
        :fire => pre.id,
        :post => post.id,
        :pre => pre.id,
        :type=>:PinningSparseSynapse,
    )
    @views g, v_post = synaptic_target(targets, post, :g, nothing)

    PINningSparseSynapse(;
        @symdict(colptr, I, W, rI, rJ, g, P, q, f)...,
        kwargs...,
        targets = targets,
    )
end

function forward!(c::PINningSparseSynapse, param::PINningSparseSynapseParameter)
    @unpack colptr, I, W, rJ, g, P, q = c
    fill!(q, zero(Float32))
    fill!(g, zero(Float32))
    @inbounds for j = 1:(length(colptr)-1)
        rJj = rJ[j]
        for s = colptr[j]:(colptr[j+1]-1)
            i = I[s]
            g[i] += W[s] * rJj
            q[i] += P[s] * rJj
        end
    end
end

function plasticity!(
    c::PINningSparseSynapse,
    param::PINningSparseSynapseParameter,
    dt::Float32,
    T::Time,
)
    @unpack colptr, I, W, rI, g, P, q, f = c
    C = 1 / (1 + dot(q, rI))
    @inbounds for j = 1:(length(colptr)-1)
        for s = colptr[j]:(colptr[j+1]-1)
            i = I[s]
            P[s] += -C * q[i] * q[j]
            W[s] += C * (f[i] - g[i]) * q[j]
        end
    end
end
