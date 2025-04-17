@snn_kw struct SpikeTimeStimulusParameter{VFT = Vector{Float32},VIT = Vector{Int}} <:
               AbstractStimulusParameter
    spiketimes::VFT=[]
    neurons::VIT=[]
end

SpikeTimeParameter(; neurons, spiketimes) = SpikeTimeStimulusParameter(spiketimes, neurons)

function SpikeTimeParameter(spiketimes::VFT, neurons::Vector{Int}) where {VFT<:Vector}
    @assert length(spiketimes) == length(neurons) "spiketimes and neurons must have the same length"
    order = sort(1:length(spiketimes), by = x -> spiketimes[x])
    return SpikeTimeStimulusParameter(Float32.(spiketimes[order]), neurons[order])
end

# spiketimes2TimeStimulus()

function SpikeTimeParameter(spiketimes::Spiketimes)
    neurons = Int[]
    times = Float32[]
    for i in eachindex(spiketimes)
        for t in spiketimes[i]
            push!(neurons, i)
            push!(times, t)
        end
    end
    order = sort(1:length(times), by = x -> times[x])
    return SpikeTimeStimulusParameter(Float32.(times[order]), neurons[order])
end

@snn_kw struct SpikeTimeStimulus{
    FT = Float32,
    VFT = Vector{Float32},
    VBT = Vector{Bool},
    DT = Distribution{Univariate,Continuous},
    VIT = Vector{Int},
} <: AbstractStimulus
    N::Int
    name::String = "SpikeTime"
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
    fire::VBT = falses(N)
    records::Dict = Dict()
    targets::Dict = Dict()
end

function SpikeTimeStimulus(
    post::T,
    sym::Symbol,
    target = nothing;
    p::R = 0.05f0,
    μ = 1.0,
    σ = 0.0,
    w = nothing,
    dist = Normal,
    N = nothing,
    param::SpikeTimeStimulusParameter,
    kwargs...,
) where {T<:AbstractPopulation,R<:Real}
    # set the synaptic weight matrix
    N = isnothing(N) ? max_neuron(param) : N
    w = sparse_matrix(w, N, post.N, dist, μ, σ, p)
    rowptr, colptr, I, J, index, W = dsparse(w)

    targets = Dict(:pre => :SpikeTimeStim, :post => post.id)
    g, _ = synaptic_target(targets, post, sym, target)

    next_spike = zeros(Float32, 1)
    next_index = zeros(Int, 1)
    next_spike[1] = isempty(param.spiketimes) ? Inf : param.spiketimes[1]
    next_index[1] = 1

    return SpikeTimeStimulus(;
        N = N,
        param = param,
        next_spike = next_spike,
        next_index = next_index,
        g = g,
        targets = targets,
        @symdict(rowptr, colptr, I, J, index, W)...,
        kwargs...,
    )
end

function SpikeTimeStimulusIdentity(
    post::T,
    sym::Symbol,
    target = nothing;
    param::SpikeTimeStimulusParameter,
    kwargs...,
) where {T<:AbstractPopulation}
    w = LinearAlgebra.I(post.N)
    return SpikeTimeStimulus(post, sym, target; w = w, param = param, kwargs...)
end

# """
#     stimulate!(p::CurrentStimulus, param::CurrentStimulus, time::Time, dt::Float32)

# Generate a Poisson stimulus for a postsynaptic population.
# """
function stimulate!(
    s::SpikeTimeStimulus,
    param::SpikeTimeStimulusParameter,
    time::Time,
    dt::Float32,
)
    @unpack colptr, I, W, fire, g, next_spike, next_index = s
    @unpack spiketimes, neurons = param
    fill!(fire, false)
    while next_spike[1] <= get_time(time)
        j = neurons[next_index[1]] # loop on presynaptic neurons
        fire[j] = true
        @inbounds @simd for s ∈ colptr[j]:(colptr[j+1]-1)
            g[I[s]] += W[s]
        end
        if next_index[1] < length(spiketimes)
            next_index[1] += 1
            next_spike[1] = spiketimes[next_index[1]]
        else
            next_spike[1] = Inf
        end
    end
end

function next_neuron(p::SpikeTimeStimulus)
    @unpack next_spike, next_index, param = p
    if next_index[1] < length(param.spiketimes)
        return param.neurons[next_index[1]]
    else
        return []
    end
end


function shift_spikes!(spiketimes::Spiketimes, delay::Number)
    for n in eachindex(spiketimes)
        spiketimes[n] .+= delay
    end
    return spiketimes
end

function shift_spikes!(param::SpikeTimeStimulusParameter, delay::Number)
    param.spiketimes .+= delay
    return param
end

function shift_spikes!(stimulus::SpikeTimeStimulus, delay::Number)
    stimulus.param.spiketimes .+= delay
    stimulus.next_index[1] = 1
    stimulus.next_spike[1] = stimulus.param.spiketimes[1]
    return stimulus
end

function update_spikes!(stim, spikes, start_time = 0.0f0)
    empty!(stim.param.spiketimes)
    empty!(stim.param.neurons)
    append!(stim.param.spiketimes, spikes.spiketimes .+ start_time)
    append!(stim.param.neurons, spikes.neurons)
    stim.next_index[1] = 1
    stim.next_spike[1] = stim.param.spiketimes[1]
    return stim
end


max_neuron(param::SpikeTimeStimulusParameter) = maximum(param.neurons)



export SpikeTimeStimulusParameter,
    SpikeTimeStimulusParameter,
    SpikeTimeStimulus,
    SpikeTimeStimulusIdentity,
    SpikeTimeParameter,
    stimulate!,
    next_neuron,
    max_neuron,
    shift_spikes!,
    update_spikes!,
    SpikeTime
