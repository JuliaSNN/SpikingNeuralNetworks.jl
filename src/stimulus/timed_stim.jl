@snn_kw struct SpikeTimeStimulusParameter{VFT=Vector{Float32}, VVIT=Vector{Vector{Int}}} <: AbstractStimulusParameter
    spiketimes::VFT
    neurons::VVIT
end 


@snn_kw struct SpikeTimeStimulus{FT=Float32, VFT = Vector{Float32}, DT=Distribution{Univariate, Continuous}, VIT = Vector{Int}} <: AbstractStimulus
    N::Int
    name::String = "SpikeTimes"
    id::String = randstring(12)
    param::SpikeTimeStimulusParameter
    rowptr::VIT # row pointer of sparse W
    colptr::VIT # column pointer of sparse W
    I::VIT      # postsynaptic index of W
    J::VIT      # presynaptic index of W
    index::VIT  # index mapping: W[index[i]] = Wt[i], Wt = sparse(dense(W)')
    W::VFT  # synaptic weight
    g::VFT  # rise conductance
    next_spike::VFT = [0]
    next_index::VIT = [0]
    fireJ::VIT = zeros(Int, N)
    records::Dict = Dict()
    targets::Dict = Dict()
end

function SpikeTimeStimulus(N, post::T, sym::Symbol, target = nothing; p::R=0.05f0,  μ=1.0, σ = 0.0, w = nothing,dist=Normal, param::SpikeTimeStimulusParameter) where {T <: AbstractPopulation, R <: Real}
    # set the synaptic weight matrix
    if isnothing(w)
        # if w is not defined, construct a random sparse matrix with `dist` with `μ` and `σ`. 
        w = rand(dist(μ, σ), post.N, N) # Construct a random dense matrix with dimensions post.N x pre.N
        w[[n for n in eachindex(w[:]) if rand() > p]] .= 0
        w[w .< 0] .= 0 
        w = sparse(w)
    else
        # if w is defined, convert it to a sparse matrix
        w = sparse(w)
    end
    @assert size(w) == (post.N, N)

    rowptr, colptr, I, J, index, W = dsparse(w)

    if isnothing(target) 
        g = getfield(post, sym)
    else
        g = getfield(post, Symbol("$(sym)_$target"))
    end

    next_spike = zeros(Float32, 1)
    next_index = zeros(Int, 1)
    next_spike[1] = param.spiketimes[1]
    next_index[1] = 1

    return SpikeTimeStimulus(;
        N = N,
        param = param,
        next_spike = next_spike,
        next_index = next_index,
        g = g,
        @symdict(rowptr, colptr, I, J, index, W)...,
    )
end


# """
#     stimulate!(p::CurrentStimulus, param::CurrentStimulus, time::Time, dt::Float32)

# Generate a Poisson stimulus for a postsynaptic population.
# """
function stimulate!(s::SpikeTimeStimulus, param::SpikeTimeStimulusParameter, time::Time, dt::Float32)
    @unpack colptr, I, W, fireJ, g, next_spike, next_index = s
    @unpack spiketimes, neurons = param
    if next_spike[1] < get_time(time)
        @inbounds for j ∈ neurons[next_index[1]] # loop on presynaptic neurons
            @inbounds @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                g[I[s]] += W[s]
            end
        end
        if next_index[1] < length(spiketimes)
            next_index[1] += 1
            next_spike[1] = spiketimes[next_index[1]]
        else
            next_spike[1] = Inf
        end
    end
end



export SpikeTimeStimulusParameter, SpikeTimeStimulusParameter, SpikeTimeStimulus, stimulate!