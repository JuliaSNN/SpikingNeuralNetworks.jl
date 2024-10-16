abstract type AbstractSpikingSynapse <: AbstractSparseSynapse end

@snn_kw mutable struct SpikingSynapse{
    VIT = Vector{Int32},
    VFT = Vector{Float32},
    VBT = Vector{Bool},
} <: AbstractSpikingSynapse
    param::SpikingSynapseParameter = no_STDPParameter()
    plasticity::PlasticityVariables = no_PlasticityVariables()
    rowptr::VIT # row pointer of sparse W
    colptr::VIT # column pointer of sparse W
    I::VIT      # postsynaptic index of W
    J::VIT      # presynaptic index of W
    index::VIT  # index mapping: W[index[i]] = Wt[i], Wt = sparse(dense(W)')
    W::VFT  # synaptic weight
    fireI::VBT # postsynaptic firing
    fireJ::VBT # presynaptic firing
    v_post::VFT
    g::VFT  # rise conductance
    αs::VFT = []
    receptors::VIT = []
    records::Dict = Dict()
end

function SpikingSynapse(pre, post, sym; σ = 0.0, p = 0.0, w = nothing, kwargs...)
    if isnothing(w)
        w = σ * sprand(post.N, pre.N, p) # Construct a random sparse matrix with dimensions post.N x pre.N and density p
    else
        w = sparse(w)
    end
    w[diagind(w)] .= 0
    @assert size(w) == (post.N, pre.N)

    rowptr, colptr, I, J, index, W = dsparse(w)
    fireI, fireJ, v_post = post.fire, pre.fire, post.v
    g = getfield(post, sym)

    # set the paramter for the synaptic plasticity
    param = haskey(kwargs, :param) ? kwargs[:param] : no_STDPParameter()
    plasticity = get_variables(param, pre.N, post.N)

    # Construct the SpikingSynapse instance
    return SpikingSynapse(;
        param = param,
        plasticity = plasticity,
        g = g,
        @symdict(rowptr, colptr, I, J, index, W, fireI, fireJ, v_post)...,
        kwargs...,
    )
end

function forward!(c::SpikingSynapse, param::SpikingSynapseParameter)
    @unpack colptr, I, W, fireJ, g, αs = c
    @inbounds for j ∈ eachindex(fireJ) # loop on presynaptic neurons
        if fireJ[j] # presynaptic fire
            @inbounds @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                g[I[s]] += W[s]
            end
        end
    end
end
