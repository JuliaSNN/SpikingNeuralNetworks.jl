struct PINningSynapseParameter end

@snn_kw mutable struct PINningSynapse{MFT = Matrix{Float32},VFT = Vector{Float32}} <:
                       AbstractConnection
    name::String = "PINningSynapse"
    id::String = randstring(12)
    param::PINningSynapseParameter = PINningSynapseParameter()
    W::MFT  # synaptic weight
    rI::VFT # postsynaptic rate
    rJ::VFT # presynaptic rate
    g::VFT  # postsynaptic conductance
    P::MFT  # <rᵢrⱼ>⁻¹
    q::VFT  # P * r
    f::VFT  # postsynaptic traget
    targets::Dict = Dict()
    records::Dict = Dict()
end

"""
[PINing Sparse Synapse](https://www.ncbi.nlm.nih.gov/pubmed/26971945)
"""
PINningSynapse

function PINningSynapse(pre, post; μ = 1.5, p = 0.0, α = 1, kwargs...)
    rI, rJ = post.r, pre.r
    W = μ * 1 / √pre.N * randn(post.N, pre.N) # normalized recurrent weight
    P = α * I(post.N) # initial inverse of C = <rr'>
    f, q = zeros(post.N), zeros(post.N)
    targets = Dict{Symbol,Any}(
        :fire => pre.id,
        :post => post.id,
        :pre => pre.id,
        :type=>:PinningSynapse,
    )
    @views g, v_post = synaptic_target(targets, post, :g, nothing)

    PINningSynapse(; @symdict(W, rI, rJ, g, P, q, f)..., kwargs..., targets = targets)
end

function forward!(c::PINningSynapse, param::PINningSynapseParameter)
    @unpack W, rI, rJ, g, P, q = c
    mul!(q, P, rJ)
    mul!(g, W, rJ)
end

function plasticity!(
    c::PINningSynapse,
    param::PINningSynapseParameter,
    dt::Float32,
    T::Time,
)
    @unpack W, rI, g, P, q, f = c
    C = 1 / (1 + dot(q, rI))
    BLAS.ger!(C, f - g, q, W)
    BLAS.ger!(-C, q, q, P)
end
