abstract type PoissonStimulusParameter end
@snn_kw struct PoissonStimulusVariable{VFT} <: PoissonStimulusParameter
    variables::Dict{Symbol,Any}
    rate::Function
    active::Vector{Bool} = [true]
end

@snn_kw struct PoissonStimulusFixed{R=Float32} <: PoissonStimulusParameter
    rate::Vector{R}
    active::Vector{Bool} = [true]
end

@snn_kw struct PoissonStimulusInterval{R = Float32} <: PoissonStimulusParameter
    rate::Vector{R} 
    intervals::Vector{Vector{R}}
    active::Vector{Bool} = [true]
end

PSParam = PoissonStimulusVariable

@snn_kw struct PoissonStimulus{
    VFT = Vector{Float32},
    VBT = Vector{Bool},
    VIT = Vector{Int},
    IT = Int32,
} <:
               AbstractStimulus
    id::String = randstring(12)
    name::String = "Poisson"
    param::PoissonStimulusParameter
    N::IT = 100
    N_pre::IT = 5
    cells::VIT
    ##
    g::VFT # target conductance for soma
    colptr::VIT
    rowptr::VIT
    I::VIT
    J::VIT
    index::VIT
    W::VFT
    fire::VBT = zeros(Bool, N)
    ##
    randcache::VFT = rand(N) # random cache
    records::Dict = Dict()
    targets::Dict = Dict()
end


"""
    PoissonStimulus(post::T, sym::Symbol, r::Union{Function, Float32}, cells=[]; N_pre::Int=50, p_post::R=0.05f0, μ::R=1.f0, param=PoissonParameter()) where {T <: AbstractPopulation, R <: Number}

Constructs a PoissonStimulus object for a spiking neural network.

# Arguments
- `post::T`: The target population for the stimulus.
- `sym::Symbol`: The symbol representing the synaptic conductance or current.
- `r::Union{Function, Float32}`: The firing rate of the stimulus. Can be a constant value or a function of time.
- `cells=[]`: The indices of the cells in the target population that receive the stimulus. If empty, cells are randomly selected based on the probability `p_post`.
- `N::Int=200`: The number of Poisson neurons cells.
- `N_pre::Int=5`: The number of presynaptic connected.
- `p_post::R=0.05f0`: The probability of connection between presynaptic and postsynaptic cells.
- `μ::R=1.f0`: The scaling factor for the synaptic weights.
- `param=PoissonParameter()`: The parameters for the Poisson distribution.

# Returns
A `PoissonStimulus` object.
"""
function PoissonStimulus(
    post::T,
    sym::Symbol,
    target = nothing;
    cells = [],
    disjoint = nothing,
    N::Int = 100,
    N_pre::Int = 50,
    p_post = 0.05f0,
    μ = 1.0f0,
    param::Union{PoissonStimulusParameter,R},
    kwargs...,
) where {T<:AbstractPopulation,R<:Real}

    ## select the cells that receive the stimulus
    if cells == :ALL
        cells = 1:post.N
    end
    if isempty(cells)
        for i = 1:post.N
            if !isnothing(disjoint) && (i in disjoint)
                continue
            elseif rand() < p_post
                push!(cells, i)
            end
        end
    end

    ## construct the connectivity matrix
    w = zeros(Float32, length(cells), N)
    for i = 1:length(cells)
        pre = rand(1:N, N_pre)
        w[i, pre] .= 1
    end
    w = μ * sparse(w)

    # normalize the strength of the synapses to each postsynaptic cell
    # w = SNN.dropzeros(w .* μ ./sum(w, dims=2))

    rowptr, colptr, I, J, index, W = dsparse(w)
    if isnothing(target)
        g = getfield(post, sym)
        targets = Dict(:pre => :Poisson, :g => post.id, :sym => :soma)
    elseif typeof(target) == Symbol
        sym = Symbol("$(sym)_$target")
        g = getfield(post, sym)
        targets = Dict(:pre => :Poisson, :g => post.id, :sym => target)
    elseif typeof(target) == Int
        sym = Symbol("$(sym)_d")
        g = getfield(post, sym)[target]
        targets = Dict(:pre => :Poisson, :g => post.id, :sym => Symbol(string(sym, target)))
    end

    if typeof(param) <: Real
        param = PoissonStimulusFixed(fill(param, N), [true])
    end

    # Construct the SpikingSynapse instance
    return PoissonStimulus(;
        param = param,
        N = N,
        N_pre = N_pre,
        cells = cells,
        targets = targets,
        g = g,
        @symdict(rowptr, colptr, I, J, index, W)...,
        kwargs...,
    )
end


"""
    stimulate!(p::PoissonStimulus, param::PoissonParameter, time::Time, dt::Float32)

Generate a Poisson stimulus for a postsynaptic population.
"""
function stimulate!(
    p::PoissonStimulus,
    param::PoissonStimulusFixed,
    time::Time,
    dt::Float32,
)
    @unpack N, N_pre, randcache, fire, cells, colptr, W, I, g = p
    @unpack rate = param
    rand!(randcache)
    @inbounds @simd for j = 1:N
        if randcache[j] < rate[j] * dt / N_pre
            fire[j] = true
            @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                g[cells[I[s]]] += W[s]
            end
        else
            fire[j] = false
        end
    end
end

function stimulate!(
    p::PoissonStimulus,
    param::PoissonStimulusInterval,
    time::Time,
    dt::Float32,
)
    @unpack active = param
    if !active[1]
        return
    end
    @unpack N, N_pre, randcache, fire, cells, colptr, W, I, g = p
    @unpack rate, intervals = param
    for int in intervals
        if !(get_time(time) > int[1] && get_time(time) < int[end])
            return
        end
    end
    # @info "Stimulating at $(get_time(time))"
    rand!(randcache)
    @inbounds @simd for j = 1:N
        if randcache[j] < rate[j] * dt / N_pre
            fire[j] = true
            @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                g[cells[I[s]]] += W[s]
            end
        else
            fire[j] = false
        end
    end
    # for j = 1:N # loop on presynaptic cells
    #     if fire[j] # presynaptic fire
    #     end
    # end
end

function stimulate!(
    p::PoissonStimulus,
    param::PoissonStimulusVariable,
    time::Time,
    dt::Float32,
)
    @unpack active = param
    if !active[1]
        return
    end
    @unpack N, N_pre, randcache, fire, cells, colptr, W, I, g = p
    myrate::Float32 = param.rate(get_time(time), param)
    myrate *= dt / N_pre
    rand!(randcache)
    @inbounds @simd for j = 1:N
        if randcache[j] < myrate
            fire[j] = true
        else
            fire[j] = false
        end
    end
    for j = 1:N # loop on presynaptic cells
        if fire[j] # presynaptic fire
            @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                g[cells[I[s]]] += W[s]
            end
        end
    end
end

function OrnsteinUhlenbeckProcess(x::Float32, param::PSParam)
    X::Float32 = param.variables[:X]
    θ::Float32 = param.variables[:θ]
    μ::Float32 = param.variables[:μ]
    σ::Float32 = param.variables[:σ]
    dt::Float32 = param.variables[:dt]

    ξ = rand(Normal())
    X = X + θ * (μ - X) * dt + σ * ξ * dt
    X = X > 0.0f0 ? X : 0.0f0

    param.variables[:X] = X
    return X
end

function SinWaveNoise(x::Float32, param::PSParam)
    X::Float32 = param.variables[:X]
    θ::Float32 = param.variables[:θ]
    σ::Float32 = param.variables[:σ]
    dt::Float32 = param.variables[:dt]
    ν::Float32 = param.variables[:ν]
    μ::Float32 = param.variables[:μ]

    W = σ * rand(Normal()) * sqrt(dt)
    X = X + θ * (μ - X) * dt - W
    param.variables[:X] = X

    Y = sin(x * 2π * ν)
    return X * 0.1 + Y * μ
end




export PoissonStimulus,
    stimulate!,
    PSParam,
    PoissonStimulusParameter,
    PoissonStimulusVariable,
    PoissonStimulusFixed,
    PoissonStimulusInterval
