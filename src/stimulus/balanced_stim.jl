@snn_kw struct BalancedStimulusParameter{VFT}
    kIE::Float32 = 1.0
    β::Float32 = 1.0
    τ::Float32 = 50.0
    r0::Float32 = 1kHz
    wIE::Float32 = 1.0
end

BSParam = BalancedStimulusParameter

@snn_kw struct BalancedStimulus{VFT = Vector{Float32},VBT = Vector{Bool},VIT = Vector{Int}, IT = Int32} <:

                       AbstractStimulus
    id::String = randstring(12)
    param::BalancedStimulusParameter
    N::IT = 100
    N_pre::IT = 5
    cells::VIT
    ##
    ge::VFT # target conductance for exc
    gi::VFT # target conductance for inh
    colptr::VIT
    rowptr::VIT
    I::VIT
    J::VIT
    index::VIT
    r::VFT
    noise::VFT
    W::VFT
    fire::VBT = zeros(Bool, N)
    ##
    randcache::VFT = rand(N) # random cache
    records::Dict = Dict()
    targets::Dict = Dict()
end


"""
    BalancedStimulus(post::T, sym::Symbol, r::Union{Function, Float32}, cells=[]; N_pre::Int=50, p_post::R=0.05f0, μ::R=1.f0, param=BalancedParameter()) where {T <: AbstractPopulation, R <: Number}

Constructs a BalancedStimulus object for a spiking neural network.

# Arguments
- `post::T`: The target population for the stimulus.
- `sym::Symbol`: The symbol representing the synaptic conductance or current.
- `r::Union{Function, Float32}`: The firing rate of the stimulus. Can be a constant value or a function of time.
- `cells=[]`: The indices of the cells in the target population that receive the stimulus. If empty, cells are randomly selected based on the probability `p_post`.
- `N::Int=200`: The number of Balanced neurons cells.
- `N_pre::Int=5`: The number of presynaptic connected.
- `p_post::R=0.05f0`: The probability of connection between presynaptic and postsynaptic cells.
- `μ::R=1.f0`: The scaling factor for the synaptic weights.
- `param=BalancedParameter()`: The parameters for the Balanced distribution.

# Returns
A `BalancedStimulus` object.
"""
function BalancedStimulus(post::T, sym_e::Symbol, sym_i::Symbol, target = nothing; cells=[], μ=1.f0, param::Union{BalancedStimulusParameter,R}) where {T <: AbstractPopulation, R <: Real}

    if cells == :ALL
        cells = 1:post.N
    end 
    if isempty(cells)
        for i in 1:post.N
            if rand() < p_post
                push!(cells, i)
            end
        end
    end
    w = zeros(Float32, length(cells), length(cells))
    for i in 1:length(cells)
        w[i, i] = 1
    end
    w = μ* sparse(w)
    rowptr, colptr, I, J, index, W = dsparse(w)

    if isnothing(target) 
        ge = getfield(post, sym_e)
        gi = getfield(post, sym_i)
    else
        ge = getfield(post, Symbol("$(sym_e)_$target"))
        gi = getfield(post, Symbol("$(sym_i)_$target"))
    end
    targets = Dict(:pre => :Balanced, :g => post.id, :compartment=>target)

    if typeof(param) <: Real
        r = param
        param = BSParam(rate = (x,y)->r, r* param.kIE)
    end

    r= ones(Float32, post.N)
    noise = zeros(Float32, post.N)

    # Construct the SpikingSynapse instance
    return BalancedStimulus(;
        param = param,
        N = length(cells),
        N_pre = 1,
        cells = cells,
        targets = targets,
        r = r,
        noise = noise,
        ge = ge,
        gi = gi,
        @symdict(rowptr, colptr, I, J, index, W)...,
    )
end


"""
    stimulate!(p::BalancedStimulus, param::BalancedParameter, time::Time, dt::Float32)

Generate a Balanced stimulus for a postsynaptic population.
"""
function stimulate!(p::BalancedStimulus, param::BalancedStimulusParameter, time::Time, dt::Float32)
    @unpack N, N_pre, randcache, fire, cells, colptr, W, I, ge, gi = p

    ## Inhomogeneous Poisson process
    @unpack r0, β, τ, kIE, wIE = param
    @unpack noise, r = p
    Erate::Float32 = r0
    Irate::Float32 = r0 * kIE

    # Inhibitory spike
    rand!(randcache)
    @inbounds @simd for j = 1:N
        if randcache[j] < Irate/N_pre * dt
            fire[j] = true
        else
            fire[j] = false
        end
    end
    for j = 1:N # loop on presynaptic cells
        if fire[j] # presynaptic fire
            @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                gi[cells[I[s]]] += W[s] * wIE
            end
        end
    end

    R(x::Float32, v0::Float32=0.f0) = x > 0.f0 ? x : v0

    # Excitatory spike
    rand!(randcache)
    @inbounds @simd for j = 1:N
        re::Float32 = randcache[j] - 0.5f0
        cc::Float32 = 1.0f0 - dt / τ
        noise[j] = ( noise[j] - re) * cc + re
        Erate = r0 *  R(noise[j] * β, 1.f0) * R(r[j], 1.f0 )
        @assert Erate >= 0
        r[j] += (1 - r[j] + (r0 - Erate)/r0) * dt
        # @info "$j: Erate: $Erate Hz, r: $(r[j]), r0: $r0, noise: $(noise[j]), β"
        if rand() < Erate  * dt
            fire[j] = true
            @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                ge[cells[I[s]]] += W[s] * (Erate*dt > 1.0f0 ? Erate*dt : 1.0f0)
            end
        else
            fire[j] = false
        end
    end

end

export BalancedStimuli, stimulate!, BSParam, BalancedStimulusParameter
