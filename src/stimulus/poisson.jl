
@snn_kw mutable struct PoissonStimulus{VFT = Vector{Float32},VBT = Vector{Bool},VIT = Vector{Int}, IT = Int32} <:
                       AbstractStimulus
    param::PoissonParameter = PoissonParameter()
    N::IT = 100
    cells::VIT
    ##
    g::VFT # target conductance
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


function PoissonStimulus(post::T, sym::Symbol, r::Union{Function, Float32}, cells=[]; N_pre::Int=50, p_post::R=0.05f0, σ::R=1.f0, param=PoissonParameter()) where {T <: AbstractPopulation, R <: Number}

    if isempty(cells)
        for i in  1:post.N
            if rand() < p_post
                push!(cells, i)
            end
        end
    end
    w = (N_pre*p_post) * ceil.(sprand(length(cells), N_pre, p_post)) # Construct a random sparse matrix with dimensions post.N x pre.N and density p
    w = SNN.dropzeros(w .* σ ./sum(w, dims=2))

    rowptr, colptr, I, J, index, W = dsparse(w)
    g = getfield(post, sym)

    (isa(r, Number)) && (r = (t::Time) -> r)

    # Construct the SpikingSynapse instance
    return PoissonStimulus(;
        param = param,
        N = N_pre,
        cells = cells,
        g = g,
        @symdict(rowptr, colptr, I, J, index, W)...,
        rate = r,
    )
end

"""
[Poisson Neuron](https://www.cns.nyu.edu/~david/handouts/poisson.pdf)
"""
Poisson

function stimulate!(p::PoissonStimulus, param::PoissonParameter, time::Time, dt::Float32)
    @unpack N, randcache, fire, rate, cells, colptr, W, I, g = p
    rand!(randcache)
    myrate::Float32 = rate(time)
    @inbounds for j = 1:N
        if randcache[j] < myrate * dt
            fire[j] = true
            @inbounds for i ∈ eachindex(cells) # loop on postsynaptic cells
                @inbounds @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                    g[cells[I[s]]] += W[s]
                end
            end
        end
    end
end

export PoissonStimuli, stimulate!
