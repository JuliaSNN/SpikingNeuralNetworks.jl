@snn_kw mutable struct PoissonStimulusParameter{VFT}
    variables::Dict{Symbol, Any}=Dict{Symbol, Any}()
    rate::Function
end

PSParam = PoissonStimulusParameter

@snn_kw mutable struct PoissonStimulus{VFT = Vector{Float32},VBT = Vector{Bool},VIT = Vector{Int}, IT = Int32, GT = gtype} <:

                       AbstractStimulus
    param::PoissonStimulusParameter
    N::IT = 100
    N_pre::IT = 5
    cells::VIT
    ##
    g::VFT # target conductance for soma
    g_d::GT # target conductance for dendrites
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
function PoissonStimulus(post::T, sym::Symbol; cells=[], N::Int=200,N_pre::Int=5, p_post::R=0.05f0, receptors::Vector{Int}=[1], μ::R=1.f0, param::Union{PoissonStimulusParameter,R2}) where {T <: AbstractPopulation, R <: Real, R2<:Real}

    if cells == :ALL
        cells = 1:post.N
    end 
    if isempty(cells)
        for i in  1:post.N
            if rand() < p_post
                push!(cells, i)
            end
        end
    end
    w = zeros(Float32, length(cells), N)
    for i in 1:length(cells)
        pre = rand(1:N, N_pre)
        w[i, pre] .= 1
    end
    w = μ* sparse(w)

    # normalize the strength of the synapses to each postsynaptic cell
    # w = SNN.dropzeros(w .* μ ./sum(w, dims=2))

    rowptr, colptr, I, J, index, W = dsparse(w)
    if typeof(post) <: AbstractDendriteIF
        g_d = view(getfield(post, sym), :, receptors)
        g = []
    else
        g = getfield(post, sym)
        a = zeros(Float32, 2,2)
        g_d = @view(a[:,receptors])
    end

    if typeof(param) <: Real
        r = param
        param = PSParam(rate = (x,y)->r)
    end

    # Construct the SpikingSynapse instance
    return PoissonStimulus(;
        param = param,
        N = N,
        N_pre = N_pre,
        cells = cells,
        g = g,
        g_d = g_d,
        @symdict(rowptr, colptr, I, J, index, W)...,
    )
end


"""
    stimulate!(p::PoissonStimulus, param::PoissonParameter, time::Time, dt::Float32)

Generate a Poisson stimulus for a postsynaptic population.
"""
function stimulate!(p::PoissonStimulus, param::PoissonStimulusParameter, time::Time, dt::Float32)
    @unpack N, N_pre, randcache, fire, cells, colptr, W, I, g, g_d = p
    myrate::Float32 = param.rate(get_time(time), param)
    rand!(randcache)
    @inbounds @simd for j = 1:N
        if randcache[j] < myrate/N_pre * dt
            fire[j] = true
        else
            fire[j] = false
        end
    end
    if isempty(g) 
        for j = 1:N # loop on presynaptic cells
            if fire[j] # presynaptic fire
                @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                    g_d[cells[I[s]],:] .+= W[s]
                end
            end
        end
    else
        for j = 1:N
            if fire[j] # presynaptic fire
                @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                    g[cells[I[s]]] += W[s]
                end
            end
        end
    end
end

export PoissonStimuli, stimulate!, PSParam, PoissonStimulusParameter
