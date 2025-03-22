abstract type PoissonStimulusParameter end

"""
    PoissonStimulusVariable

    Poisson stimulus with rate defined with a function.
    
    # Fields
    - `variables::Dict{Symbol,Any}`: A dictionary containing the variables for the function.
    - `rate::Function`: A function defining the rate of the Poisson stimulus.
    - `active::Vector{Bool}`: A vector of booleans indicating if the stimulus is active.
"""
PoissonStimulusVariable
@snn_kw struct PoissonStimulusVariable{VFT} <: PoissonStimulusParameter
    variables::Dict{Symbol,Any}
    rate::Function
    active::Vector{Bool} = [true]
end

"""
    PoissonStimulusFixed

    Poisson stimulus with fixed rate. The rate arrives to all the cells targeted
    by the stimulus.

    # Fields
    - `rate::Vector{R}`: A vector containing the rate of the Poisson stimulus.
    - `active::Vector{Bool}`: A vector of booleans indicating if the stimulus is active.
"""
PoissonStimulusFixed

@snn_kw struct PoissonStimulusFixed{R = Float32} <: PoissonStimulusParameter
    rate::Vector{R}
    active::Vector{Bool} = [true]
end

"""
    PoissonStimulusLayer

    Poisson stimulus with rate defined for each cell in the layer. Each neuron of the 'N' Poisson population fires with 'rate'.
    The connectivity is defined by the parameter 'ϵ'. Thus, the number of presynaptic cells connected to the postsynaptic cells is 'N*ϵ'. Each post-synaptic cell receives rate: 'rate * N * ϵ'.

    # Fields
    - `rate::Vector{R}`: A vector containing the rate of the Poisson stimulus.
    - `N::Int32`: The number of cells in the layer.
    - `ϵ::Float32`: The fraction of presynaptic cells connected to the postsynaptic cells.
    - `active::Vector{Bool}`: A vector of booleans indicating if the stimulus is active.
"""
PoissonStimulusLayer
@snn_kw struct PoissonStimulusLayer{R = Float32} <: PoissonStimulusParameter
    rate::Vector{R}
    N::Int32
    ϵ::Float32
    active::Vector{Bool} = [true]
end

"""
    PoissonStimulusInterval

    Poisson stimulus with rate defined for each cell in the layer. Each neuron of the 'N' Poisson population fires with 'rate' in the intervals defined by 'intervals'.
    
    # Fields
    - `rate::Vector{R}`: A vector containing the rate of the Poisson stimulus.
    - `intervals::Vector{Vector{R}}`: A vector of vectors containing the intervals in which the Poisson stimulus is active.
    - `active::Vector{Bool}`: A vector of booleans indicating if the stimulus is active.
"""
PoissonStimulusInterval
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
} <: AbstractStimulus
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


function PoissonStimulus(
    post::T,
    sym::Symbol,
    target = nothing;
    cells = :ALL,
    N::Int = 100,
    p_post = 1.0,
    N_pre = 50,
    μ = 1.0f0,
    param::Union{PoissonStimulusParameter,R},
    kwargs...,
) where {T<:AbstractPopulation,R<:Real}

    if typeof(param) <: Real
        param = PoissonStimulusFixed(fill(param, N), [true])
        N_pre = round(Int, N * 0.05)
    elseif typeof(param) == PoissonStimulusLayer
        N = param.N
        N_pre = round(Int, N * param.ϵ)
    end

    ## select a subset of cells that receive the stimulus
    cells = cell == :ALL ? eachindex(post.N) : cells
    for i = 1:post.N
        (rand() < p_post) && (push!(cells, i))
    end

    ## construct the connectivity matrix
    w = zeros(Float32, length(cells), N)
    for i = 1:length(cells)
        pre = rand(1:N, round(Int, N_pre))
        w[i, pre] .= 1
    end
    w = μ * sparse(w)

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
    param::PoissonStimulusLayer,
    time::Time,
    dt::Float32,
)
    @unpack N, N_pre, randcache, fire, cells, colptr, W, I, g = p
    @unpack rate = param
    rand!(randcache)
    @inbounds @simd for j = 1:N
        if randcache[j] < rate[j] * dt
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
    PoissonStimulusInterval,
    PoissonStimulusLayer
