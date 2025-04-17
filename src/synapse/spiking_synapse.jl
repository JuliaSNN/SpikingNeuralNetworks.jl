
@snn_kw mutable struct SpikingSynapse{
    VIT = Vector{Int32},
    VFT = Vector{Float32},
    VBT = Vector{Bool},
} <: AbstractSpikingSynapse
    id::String = randstring(12)
    name::String = "SpikingSynapse"
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
    name::String = "SpikingSynapseDelay"
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


function CompartmentSynapse(pre, post, target::Symbol, sym::Symbol; kwargs...)
    SpikingSynapse(pre, post, sym, target; kwargs...)
end

function SpikingSynapse(
    pre,
    post,
    sym,
    target = nothing;
    delay_dist = nothing,
    μ = 1.0,
    σ = 0.0,
    p = 0.0,
    w = nothing,
    dist::Symbol = :Normal,
    kwargs...,
)

    # set the synaptic weight matrix
    w = sparse_matrix(w, pre.N, post.N, dist, μ, σ, p)
    # remove autapses if pre == post
    (pre == post) && (w[diagind(w)] .= 0)
    # get the sparse representation of the synaptic weight matrix
    rowptr, colptr, I, J, index, W = dsparse(w)
    # get the presynaptic and postsynaptic firing
    fireI, fireJ = post.fire, pre.fire

    # get the conductance and membrane potential of the target compartment if multicompartment model
    targets = Dict{Symbol,Any}(:fire => pre.id, :post => post.id, :pre=> pre.id, :type=>:SpikingSynapse)
    @views g, v_post =  synaptic_target(targets, post, sym, target)

    # set the paramter for the synaptic plasticity
    param = haskey(kwargs, :param) ? kwargs[:param] : no_STDPParameter()
    plasticity = plasticityvariables(param, pre.N, post.N)

    # short term plasticity
    ρ = copy(W)
    ρ .= 1.0

    # Network targets

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
        delaytime = round.(Int, rand(delay_dist, length(W)) / 0.1)
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
