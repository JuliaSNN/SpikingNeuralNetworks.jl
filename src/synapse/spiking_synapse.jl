@snn_kw mutable struct SpikingSynapse{
    VIT = Vector{Int32},
    VFT = Vector{Float32},
    VBT = Vector{Bool},
} <: AbstractSpikingSynapse
    id::String = randstring(12)
    name::String = "SpikingSynapse"
    param::SpikingSynapseParameter = SpikingSynapseParameter()
    LTPParam::LTPParameter = NoLTP()
    STPParam::STPParameter = NoSTP()
    LTPVars::PlasticityVariables = NoPlasticityVariables()
    STPVars::PlasticityVariables = NoPlasticityVariables()
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
    param::SpikingSynapseParameter = SpikingSynapseParameter()
    LTPParam::SpikingSynapseParameter = NoLTP()
    STPParam::SpikingSynapseParameter = noSTP()
    LTPVars::PlasticityVariables = NoPlasticityVariables()
    STPVars::PlasticityVariables = NoPlasticityVariables()
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
    dt = 0.125f0,
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
    targets = Dict{Symbol,Any}(
        :fire => pre.id,
        :post => post.id,
        :pre => pre.id,
        :type=>:SpikingSynapse,
    )
    @views g, v_post = synaptic_target(targets, post, sym, target)

    # set the paramter for the synaptic plasticity
    LTPParam = haskey(kwargs, :LTPParam) ? kwargs[:LTPParam] : NoLTP()
    LTPVars = plasticityvariables(LTPParam, pre.N, post.N)
    STPParam = haskey(kwargs, :STPParam) ? kwargs[:STPParam] : NoSTP()
    STPVars = plasticityvariables(STPParam, pre.N, post.N)

    # short term plasticity
    ρ = copy(W)
    ρ .= 1.0

    # Network targets

    if isnothing(delay_dist)
        # Construct the SpikingSynapse instance
        return SpikingSynapse(;
            ρ = ρ,
            g = g,
            targets = targets,
            @symdict(rowptr, colptr, I, J, index, W, fireI, fireJ, v_post)...,
            LTPVars, 
            STPVars, 
            LTPParam,
            STPParam,
            kwargs...,
        )

    else
        delayspikes = fill(-1, length(W))
        delaytime = round.(Int, rand(delay_dist, length(W))/dt )

        return SpikingSynapseDelay(;
            ρ = ρ,
            delayspikes = delayspikes,
            delaytime = delaytime,
            g = g,
            targets = targets,
            @symdict(rowptr, colptr, I, J, index, W, fireI, fireJ, v_post)...,
            LTPVars, 
            STPVars, 
            LTPParam,
            STPParam,
            kwargs...,
        )
    end
end

function update_plasticity!(c::SpikingSynapse; LTP=nothing, STP=nothing)
    if !isnothing(LTP)
        c.LTPParam = LTP
        c.LTPVars = plasticityvariables(c.LTPParam, length(c.fireJ), length(c.fireI))
    end
    if !isnothing(STP)
        c.STPParam = STP
        c.STPVars = plasticityvariables(c.STPParam, length(c.fireJ), length(c.fireI))
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
            end
        end
    end
    delayspikes .-= 1 # decrement the delay on 1 timestep
    @inbounds for j ∈ eachindex(fireJ) # loop on presynaptic neurons
        @inbounds @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
            if delayspikes[s] == 0
                delayspikes[s] = -1
                g[I[s]] += W[s] * ρ[s]
            end
        end
    end
end

export SpikingSynapse, SpikingSynapseDelay,  update_plasticity!
