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
    ρ::VFT  # short-term plasticity
    fireI::VBT # postsynaptic firing
    fireJ::VBT # presynaptic firing
    v_post::VFT
    g::VFT  # rise conductance
    αs::VFT = []
    receptors::VIT = []
    records::Dict = Dict()
end

@snn_kw mutable struct SpikingSynapseDelay{
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
    ρ::VFT  # short-term plasticity
    fireI::VBT # postsynaptic firing
    fireJ::VBT # presynaptic firing
    v_post::VFT
    g::VFT  # rise conductance
    αs::VFT = []
    delayspikes::VIT = []
    delaytime::VIT = []
    receptors::VIT = []
    records::Dict = Dict()
end



function SpikingSynapse(pre, post, sym; delay_dist=nothing, μ=1.0, σ = 0.0, p = 0.0, w = nothing, dist=Normal, kwargs...)
    if isnothing(w)
        w = rand(dist(μ, σ), post.N, pre.N) # Construct a random dense matrix with dimensions post.N x pre.N
        w[[n for n in eachindex(w[:]) if rand() > p]] .= 0
        w[w .< 0] .= 0 
        w = sparse(w)
    else
        w = sparse(w)
    end
    (pre == post) && (w[diagind(w)] .= 0) # remove autapses if pre == post
    @assert size(w) == (post.N, pre.N)

    rowptr, colptr, I, J, index, W = dsparse(w)
    ρ = copy(W)
    # short term plasticity
    ρ .= 1.0


    fireI, fireJ, v_post = post.fire, pre.fire, post.v
    g = getfield(post, sym)

    # set the paramter for the synaptic plasticity
    param = haskey(kwargs, :param) ? kwargs[:param] : no_STDPParameter()
    plasticity = get_variables(param, pre.N, post.N)

    if isnothing(delay_dist)
        # Construct the SpikingSynapse instance
        return SpikingSynapse(;
            ρ = ρ,
            param = param,
            plasticity = plasticity,
            g = g,
            @symdict(rowptr, colptr, I, J, index, W, fireI, fireJ, v_post)...,
            kwargs...,
        )

    # delay spike time
    else
        delayspikes = fill(-1, length(W))
        delaytime = round.(Int,rand(delay_dist, length(W))/0.1)
        return SpikingSynapseDelay(;
            param = param,
            ρ = ρ,
            plasticity = plasticity,
            delayspikes = delayspikes,
            delaytime = delaytime,
            g = g,
            @symdict(rowptr, colptr, I, J, index, W, fireI, fireJ, v_post)...,
            kwargs...,
        )
    end
end

function forward!(c::SpikingSynapse, param::SpikingSynapseParameter)
    @unpack colptr, I, W, fireJ, g, ρ, αs = c
    @inbounds for j ∈ eachindex(fireJ) # loop on presynaptic neurons
        if fireJ[j] # presynaptic fire
            @inbounds @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                g[I[s]] += W[s] * ρ[s]
            end
        end
    end
end



function forward!(c::SpikingSynapseDelay, param::SpikingSynapseParameter)
    @unpack colptr, I, W, fireJ, g, ρ, αs = c
    @unpack delayspikes, delaytime = c
    # Threads.@threads 
    for j ∈ eachindex(fireJ) # loop on presynaptic neurons
        if fireJ[j] # presynaptic fire
            @inbounds @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                delayspikes[s] = delaytime[s]
                delayspikes[s] -= 1
            end
        end
    end
    @inbounds for j ∈ eachindex(fireJ) # loop on presynaptic neurons
        if fireJ[j] # presynaptic fire
            @inbounds @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                g[I[s]] += W[s] * ρ[s]
            end
        end
    end
end