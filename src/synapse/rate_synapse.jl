@snn_kw struct RateSynapseParameter{FT = Float32} <: AbstractConnectionParameter
    lr::FT = 1e-3
end

@snn_kw mutable struct RateSynapse{VIT = Vector{Int32},VFT = Vector{Float32}} <:
                       AbstractConnection
    name::String="RateSynapse"
    id::String = randstring(12)
    param::RateSynapseParameter = RateSynapseParameter()
    colptr::VIT # column pointer of sparse W
    I::VIT      # postsynaptic index of W
    W::VFT  # synaptic weight
    rI::VFT # postsynaptic rate
    rJ::VFT # presynaptic rate
    g::VFT  # postsynaptic conductance
    targets::Dict = Dict()
    records::Dict = Dict()
end

"""
[Rate Synapse](https://brian2.readthedocs.io/en/2.0b4/resources/tutorials/2-intro-to-brian-synapses.html)
"""
RateSynapse

function RateSynapse(pre, post; μ = 0.0, p = 0.0, kwargs...)
    w = μ / √(p * pre.N) * sprandn(post.N, pre.N, p)
    rowptr, colptr, I, J, index, W = dsparse(w)
    rI, rJ = post.r, pre.r
    targets = Dict{Symbol,Any}(
        :fire => pre.id,
        :post => post.id,
        :pre => pre.id,
        :type=>:RateSynapse,
    )
    @views g, v_post = synaptic_target(targets, post, :g, nothing)

    RateSynapse(; @symdict(colptr, I, W, rI, rJ, g)..., kwargs..., targets = targets)
end

function forward!(c::RateSynapse, param::RateSynapseParameter)
    @unpack colptr, I, W, rI, rJ, g = c
    @unpack lr = param
    # fill!(g, zero(eltype(g)))
    @inbounds for j = 1:(length(colptr)-1)
        rJj = rJ[j]
        for s = colptr[j]:(colptr[j+1]-1)
            g[I[s]] += W[s] * rJj
        end
    end
end

function plasticity!(c::RateSynapse, param::RateSynapseParameter, dt::Float32, T::Time)
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
