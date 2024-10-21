
@snn_kw mutable struct PoissonStimulus{VFT = Vector{Float32},VBT = Vector{Bool},VIT = Vector{Int}, IT = Int32, GT = gtype} <:

                       AbstractStimulus
    param::PoissonParameter = PoissonParameter()
    N::IT = 100
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
    rate::Function # rate function
    randcache::VFT = rand(N) # random cache
    records::Dict = Dict()
end


"""
    PoissonStimulus(post::T, sym::Symbol, r::Union{Function, Float32}, cells=[]; N_pre::Int=50, p_post::R=0.05f0, σ::R=1.f0, param=PoissonParameter()) where {T <: AbstractPopulation, R <: Number}

Constructs a PoissonStimulus object for a spiking neural network.

# Arguments
- `post::T`: The target population for the stimulus.
- `sym::Symbol`: The symbol representing the synaptic conductance or current.
- `r::Union{Function, Float32}`: The firing rate of the stimulus. Can be a constant value or a function of time.
- `cells=[]`: The indices of the cells in the target population that receive the stimulus. If empty, cells are randomly selected based on the probability `p_post`.
- `N_pre::Int=50`: The number of presynaptic cells.
- `p_post::R=0.05f0`: The probability of connection between presynaptic and postsynaptic cells.
- `σ::R=1.f0`: The scaling factor for the synaptic weights.
- `param=PoissonParameter()`: The parameters for the Poisson distribution.

# Returns
A `PoissonStimulus` object.
"""
function PoissonStimulus(post::T, sym::Symbol, r::Union{Function}; cells=[], N_pre::Int=200, p_post::R=0.05f0, σ::R=1.f0, param=PoissonParameter()) where {T <: AbstractPopulation, R <: Number}

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
    w = ceil.(sprand(length(cells), N_pre, 0.2)) # Construct a random sparse matrix with dimensions post.N x pre.N and density p

    # normalize the strength of the synapses to each postsynaptic cell
    w = SNN.dropzeros(w .* σ ./sum(w, dims=2))

    rowptr, colptr, I, J, index, W = dsparse(w)
    if typeof(post) <: AbstractDendriteIF
        g_d = view(getfield(post, sym), :, [1])
        g = []
    else
        g = getfield(post, sym)

        a = zeros(Float32, 2,2)
        g_d = @view(a[:,[1,2]])
    end

    (isa(r, Number)) && (r = (t::Time) -> r)

    # Construct the SpikingSynapse instance
    return PoissonStimulus(;
        param = param,
        N = N_pre,
        cells = cells,
        g = g,
        g_d = g_d,
        @symdict(rowptr, colptr, I, J, index, W)...,
        rate = r,
    )
end

"""
[Poisson Neuron](https://www.cns.nyu.edu/~david/handouts/poisson.pdf)
"""
Poisson

function stimulate!(p::PoissonStimulus, param::PoissonParameter, time::Time, dt::Float32)
    @unpack N, randcache, fire, rate, cells, colptr, W, I, g, g_d = p
    myrate = rate(get_time(time))
    rand!(randcache)
    @inbounds @simd for j = 1:N
        if randcache[j] < myrate * dt
            fire[j] = true
        end
    end
    if isempty(g) 
        for j = 1:N # loop on presynaptic cells
            if fire[j] # presynaptic fire
                @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                    g_d[cells[I[s]]] += W[s]
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

export PoissonStimuli, stimulate!
