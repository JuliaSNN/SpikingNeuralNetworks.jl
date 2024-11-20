abstract type AbstractSpikingSynapse <: AbstractSparseSynapse end

@snn_kw mutable struct SpikingSynapse{
    VIT = Vector{Int32},
    VFT = Vector{Float32},
    VBT = Vector{Bool},
} <: AbstractSpikingSynapse
    id::String = randstring(12)
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
    targets::Dict = Dict()
    records::Dict = Dict()
end

@snn_kw mutable struct SpikingSynapseDelay{
    VIT = Vector{Int32},
    VFT = Vector{Float32},
    VBT = Vector{Bool},
} <: AbstractSpikingSynapse
    id::String = randstring(12)
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
    delayspikes::VIT = []
    delaytime::VIT = []
    targets::Dict = Dict()
    records::Dict = Dict()
end


function CompartmentSynapse(    pre,    post,    target::Symbol,    sym::Symbol;     kwargs...)
    SpikingSynapse(pre, post, sym, target; kwargs...)
end

function SpikingSynapse(pre, post, sym, target=nothing; delay_dist=nothing, μ=1.0, σ = 0.0, p = 0.0, w = nothing, dist=Normal, kwargs...)

    # set the synaptic weight matrix
    if isnothing(w)
        # if w is not defined, construct a random sparse matrix with `dist` with `μ` and `σ`. 
        w = rand(dist(μ, σ), post.N, pre.N) # Construct a random dense matrix with dimensions post.N x pre.N
        w[[n for n in eachindex(w[:]) if rand() > p]] .= 0
        w[w .< 0] .= 0 
        w = sparse(w)
    else
        # if w is defined, convert it to a sparse matrix
        w = sparse(w)
    end
    (pre == post) && (w[diagind(w)] .= 0) # remove autapses if pre == post
    @assert size(w) == (post.N, pre.N)
    targets = Dict{Symbol,Any}(:fire => pre.id, :g => post.id)
    # get the sparse representation of the synaptic weight matrix
    rowptr, colptr, I, J, index, W = dsparse(w)

    # get the presynaptic and postsynaptic firing
    fireI, fireJ = post.fire, pre.fire

    # get the conductance and membrane potential of the target compartment if multicompartment model

    g = zeros(Float32, post.N)
    v_post = zeros(Float32, post.N)
    if !isnothing(sym)
        _sym = isnothing(target) ? sym : Symbol("$(sym)_$target")
        _v   = isnothing(target) ? :v : Symbol("v_$target")
        g = getfield(post, _sym)
        hasfield(typeof(post), _v) && (v_post = getfield(post, _v))
        push!(targets, :sym => _sym)
    end

    # set the paramter for the synaptic plasticity
    param = haskey(kwargs, :param) ? kwargs[:param] : no_STDPParameter()
    plasticity = plasticityvariables(param, pre.N, post.N)

    # short term plasticity
    ρ = copy(W)
    ρ .= 1.0

    if isnothing(delay_dist)
        # Construct the SpikingSynapse instance
        return SpikingSynapse(;
            ρ = ρ,
            param = param,
            plasticity = plasticity,
            g = g,
            targets = targets,
            @symdict(rowptr, colptr, I, J, index, W, fireI, fireJ, v_post)...,
            kwargs...,
        )

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
            targets = targets,
            @symdict(rowptr, colptr, I, J, index, W, fireI, fireJ, v_post)...,
            kwargs...,
        )
    end
end

function forward!(c::SpikingSynapse, param::SpikingSynapseParameter)
    @unpack colptr, I, W, fireJ, g, ρ = c
    @inbounds for j ∈ eachindex(fireJ) # loop on presynaptic neurons
        if fireJ[j] # presynaptic fire
            @inbounds @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                g[I[s]] += W[s] * ρ[s]
            end
        end
    end
end



function forward!(c::SpikingSynapseDelay, param::SpikingSynapseParameter)
    @unpack colptr, I, W, fireJ, g, ρ = c
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