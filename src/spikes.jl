using RollingFunctions

function init_spiketimes(N)
    _s = Vector{Vector{Float32}}()
    for i in 1:N
        push!(_s, Vector{Float32}())
    end
    return Spiketimes(_s)
end

"""
    spiketimes(p, interval = nothing, indices = nothing)

Compute the spike times of a population.

Arguments:
- `p`: The network parameters.
- `interval`: The time interval within which to compute the spike times. If `nothing`, the interval is set to (0, firing_time[end]).
- `indices`: The indices of the neurons for which to compute the spike times. If `nothing`, spike times are computed for all neurons.

Returns:
- `spiketimes`: A vector of vectors containing the spike times of each neuron.
"""
function spiketimes(p::T, interval = nothing, indices = nothing) where T<:AbstractNeuron
    if isnothing(indices)
        spiketimes  =init_spiketimes(p.N)
        indices = 1:p.N
    else
        spiketimes = init_spiketimes(length(indices))
    end

    firing_time = p.records[:fire][:time]
    neurons = p.records[:fire][:neurons]

    if isnothing(interval)
        interval = (0, firing_time[end])
    end
    tt0, tt1 = findfirst(x -> x > interval[1], firing_time), findlast(x -> x < interval[2], firing_time)
    for tt in tt0:tt1
        for n in neurons[tt]
            push!(spiketimes[n], firing_time[tt])
        end
    end
    return spiketimes
end

function spiketimes(P) where T<:AbstractNeuron
    collect(Iterators.flatten(map(keys(P)) do k
        p = getfield(P, k)
        spiketimes(p)
        end
    ))
end

function population_indices(P, type="ˆ")
    n = 1
    indices = Dict{Symbol, Vector{Int}}()
    for k in keys(P)
        !occursin(string(type), string(k)) && continue
        p = getfield(P, k)
        indices[k] = n:(n + p.N - 1)
        n += p.N
    end
    return dict2ntuple(sort(indices))
end

function filter_populations(P, type)
    n = 1
    indices = Dict{Symbol, Any}()
    for k in keys(P)
        !occursin(string(type), string(k)) && continue
        p = getfield(P, k)
        indices[k] = p
    end
    return dict2ntuple(sort(indices))
end

"""
    alpha_function(t::T; t0::T, τ::T) where T <: AbstractFloat

    Alpha function for convolution of spiketimes. Evaluate the alpha function at time t, with time of peak t0 and time constant τ.
"""
function alpha_function(t::T; t0::T, τ::T) where {T<:Float32}
    return @fastmath 1 / τ * SNN.exp32(1 - (t - t0) / τ) * Θ(1.0 * (t - t0))
end
"""
    Θ(x::Float64)

    Heaviside function
"""
Θ(x::Float64) = x > 0.0 ? x : 0.0

"""
    convolve(spiketime::Vector{Float32}; interval::AbstractRange, τ = 100)

    Convolve one neuron spiketimes with alpha function to have an approximate rate.

    Parameters
    ----------
    spiketime: Vector{Float32}
        Vector of spiketimes in milliseconds
    interval: AbstractRange
        Time interval to evaluate the rate
    τ: Float32
        Time constant of the alpha function
    Return
    ------
    rate: Vector{Float32}
        Vector of rates in Hz
"""
function convolve(spiketime::Vector{Float32}; interval::AbstractRange, τ = 100.0f0, f::Function = alpha_function)
    rate = zeros(Float32, length(interval))
    @inbounds for i in eachindex(interval)
        v = 0
        t = Float32(interval[i])
        τ = Float32(τ)
        @simd for t0 in spiketime
            @fastmath if (t > t0 && ((t - t0) / τ) < 10)
                v += f(t, t0 = t0, τ = τ)
            end
        end
        rate[i] = v
    end
    return rate ## Hz
end


"""
    merge_spiketimes(spikes::Vector{Spiketimes}; )

    Merge spiketimes from different simulations. 
    This function is not thread safe, it is not recommended to use it in parallel.
    Parameters
    ----------
    spikes: Vector{Spiketimes}
        Vector of spiketimes from different simulations
    Return
    ------
    neurons: Spiketimes
        Single vector of spiketimes 
"""
function merge_spiketimes(spikes::Vector{Spiketimes};)
    neurons = [Vector{Float32}() for _ = 1:length(spikes[1])]
    neuron_ids = collect(1:length(spikes[1]))
    sub_indices = k_fold(neuron_ids, Threads.nthreads())
    sub_neurons = [neuron_ids[x] for x in sub_indices]
    Threads.@threads for p in eachindex(sub_indices)
        for spiketimes in spikes
            for (n, id) in zip(sub_indices[p], sub_neurons[p])
                push!(neurons[n], spiketimes[id]...)
            end
        end
    end
    return sort!.(neurons)
end

"""
    firing_rate(
        spiketimes::Spiketimes,
        interval::AbstractVector = [],
        sampling = 20,
        τ = 25,
        ttf = -1,
        tt0 = -1,
        cache = true,
        pop::Union{Symbol,Vector{Int}}= :ALL,
    )

Calculate the firing rates for a population or an individual neuron.

# Arguments
- `spiketimes`: Spiketimes object.
- `interval`: Time interval vector (default is an empty vector).
- `sampling`: Sampling rate (default is 20ms).
- `τ`: Time constant for convolution (default is 25ms).
- `ttf`: Final time point (default is -1, which means until the end of the simulation time).
- `tt0`: Initial time point (default is -1, which means from the start of the simulation time based on the sampling rate).
- `cache`: If true, uses cached data (default is true).
- `pop`: Either :ALL for all populations or a Vector of Integers specifying specific neuron indices. Default is :ALL.

# Returns
A tuple containing:
- `rates`: A vector of firing rates for each neuron in the chosen population.
- `interval`: The time interval over which the firing rates were calculated.

# Examples
"""
function firing_rate(
    spiketimes::Spiketimes,
    interval::AbstractVector = [];
    sampling = 20ms,
    τ = 25ms,
    ttf = -1,
    tt0 = -1,
    cache = true,
    pop::Union{Symbol,Vector{Int}}= :ALL,
)
    if isempty(interval)
        tt0 = tt0 > 0 ? tt0 : 0.f0
        ttf = ttf > 0 ? ttf : 1000.f0 
        interval = tt0:sampling:ttf
    end
    spiketimes = pop == :ALL ? spiketimes : spiketimes[pop]
    rates = tmap(
        n -> convolve(spiketimes[n], interval = interval, τ = τ),
        eachindex(spiketimes),
    )
    # rates = vcat(rates'...)
    return rates, interval
end


function get_isi(spiketimes::Spiketimes)
    return diff.(spiketimes)
end

get_isi(spiketimes::NNSpikes, pop::Symbol) = read(spiketimes, pop) |> x -> diff.(x)

function get_CV(spikes::Spiketimes)
    intervals = get_isi(spikes;)
    cvs = sqrt.(var.(intervals) ./ (mean.(intervals) .^ 2))
    cvs[isnan.(cvs)] .= -0.0
    return cvs
end

"""
get_spikes_in_interval(spiketimes::Spiketimes, interval::AbstractRange)

Return the spiketimes in the selected interval

# Arguments
spiketimes::Spiketimes: Vector with each neuron spiketime
interval: 2 dimensional array with the start and end of the interval

"""
function get_spikes_in_interval(
    spiketimes::Spiketimes,
    interval,
    margin = [0,0];
    collapse::Bool = false,
)
    neurons = [Vector{Float32}() for x = 1:length(spiketimes)]
    @inbounds @fastmath for n in eachindex(neurons)
        ff = findfirst(x -> x > interval[1] + margin[1], spiketimes[n])
        ll = findlast(x -> x <= interval[2] + margin[2], spiketimes[n])
        if !isnothing(ff) && !isnothing(ll)
            @views append!(neurons[n], spiketimes[n][ff:ll])
        end
    end
    return neurons
end

function get_spikes_in_intervals(
    spiketimes::Spiketimes,
    intervals::Vector{Vector{Float32}};
    margin = 0,
    floor = true,
)
    st = tmap(intervals) do interval
        get_spikes_in_interval(spiketimes, interval, margin)
    end
    (floor) && (interval_standard_spikes!(st, intervals))
    return st
end

function find_interval_indices(
    intervals::AbstractVector{T},
    interval::Vector{T},
) where {T<:Real}
    x1 = findfirst(intervals .>= interval[1])
    x2 = findfirst(intervals .>= interval[2])
    return x1:x2
end


"""
    interval_standard_spikes(spiketimes, interval)

Standardize the spiketimes to the interval [0, interval_duration].
Return a copy of the 'Spiketimes' vector. 
"""
function interval_standard_spikes(spiketimes, interval)
    zerod_spiketimes = deepcopy(spiketimes)
    for i in eachindex(spiketimes)
        zerod_spiketimes[i] .-= interval[1]
    end
    return Spiketimes(zerod_spiketimes)
end

function interval_standard_spikes!(
    spiketimes::Vector{Spiketimes},
    intervals::Vector{Vector{Float32}},
)
    @assert length(spiketimes) == length(intervals)
    for i in eachindex(spiketimes)
        interval_standard_spikes!(spiketimes[i], intervals[i])
    end
end

function interval_standard_spikes!(spiketimes, interval::Vector{Float32})
    for i in eachindex(spiketimes)
        spiketimes[i] .-= interval[1]
    end
    return spiketimes
end


"""
    CV_isi2(intervals::Vector{Float32})

    Return the local coefficient of variation of the interspike intervals
    Holt, G. R., Softky, W. R., Koch, C., & Douglas, R. J. (1996). Comparison of discharge variability in vitro and in vivo in cat visual cortex neurons. Journal of Neurophysiology, 75(5), 1806–1814. https://doi.org/10.1152/jn.1996.75.5.1806
"""
function CV_isi2(intervals::Vector{Float32})
    ISI = diff(intervals)
    CV2 = Float32[]
    for i in eachindex(ISI)
        i == 1 && continue
        x = 2(abs(ISI[i] - ISI[i-1]) / (ISI[i] + ISI[i-1]))
        push!(CV2, x)
    end
    _cv = mean(CV2)

    # _cv = sqrt(var(intervals)/mean(intervals)^2)
    return isnan(_cv) ? 0.0 : _cv
end

function isi_cv(spikes::Vector{NNSpikes}; kwargs...)
    spiketimes = merge_spiketimes(spikes; kwargs...)
    @unpack tt = spikes[end]
    return CV_isi2.(spiketimes)
end

get_isi_cv(x::Spiketimes) = CV_isi2.(x)

"""
    get_st_order(spiketimes::Spiketimes)
"""
function get_st_order(spiketimes::T) where {T<:Vector{}}
    ii = sort(eachindex(1:length(spiketimes)), by = x -> spiketimes[x])
    return ii
end

function get_st_order(spiketimes::Spiketimes, pop::Vector{Int}, intervals)
    @unpack spiketime = get_spike_statistics(spiketimes[pop], intervals)
    ii = sort(eachindex(pop), by = x -> spiketime[x])
    return pop[ii]
end

function get_st_order(
    spiketimes::Spiketimes,
    populations::Vector{Vector{Int}},
    intervals::Vector{Vector{T}},
    unique_pop::Bool = false,
) where {T<:Real}
    return [get_st_order(spiketimes, population, intervals) for population in populations]
end

"""
    relative_time(spiketimes::Spiketimes, start_time)

Return the spiketimes relative to the start_time of the interval
"""
function relative_time!(spiketimes::Spiketimes, start_time)
    neurons = 1:length(spiketimes)
    for n in neurons
        spiketimes[n] = spiketimes[n] .- start_time
    end
    return spiketimes
end

get_spike_count(x::Spiketimes) = length.(x)


"""
    firing_rate_average(P; dt=0.1ms)

Calculates and returns the average firing rates of neurons in a network.

# Arguments:
- `P`: A structure containing neural data, with a key `:fire` in its `records` field which stores spike information for each neuron.
- `dt`: An optional parameter specifying the time interval (default is 0.1ms).

# Returns:
An array of floating point values representing the average firing rate for each neuron.

# Usage:# Notes:
Each row of `P.records[:fire]` represents a neuron, and each column represents a time point. The value in a cell indicates whether that neuron has fired at that time point (non-zero value means it has fired).
The firing rate of a neuron is calculated as the total number of spikes divided by the total time span.
"""
function firing_rate_average(P; dt = 0.1ms)
    @assert haskey(P.records, :fire)
    spikes = hcat(P.records[:fire]...)
    time_span = size(spikes, 2) / 1000 * dt
    rates = Vector{Float32}()
    for spike in eachrow(spikes)
        push!(rates, sum(spike) / time_span)
    end
    return rates
end

"""
    firing_rate(P, τ; dt=0.1ms)

Calculate the firing rate of neurons.

# Arguments
- `P`: A struct or object containing neuron information, including records of when each neuron fires.
- `τ`: The time window over which to calculate the firing rate.

# Keywords
- `dt`: The time step for calculation (default is 0.1 ms).

# Returns
A 2D array with firing rates. Each row corresponds to a neuron and each column to a time point.

# Note
This function assumes that the firing records in `P` are stored as columns corresponding to different time points. 
The result is normalized by `(dt/1000)` to account for the fact that `dt` is typically in milliseconds.

"""
function firing_rate(P, τ; dt = 0.1ms)
    spikes = hcat(P.records[:fire]...)
    time_span = round(Int, size(spikes, 2) * dt)
    rates = zeros(P.N, time_span)
    L = round(Int, time_span - τ) * 10
    my_spikes = Matrix{Int}(spikes)
    @fastmath @inbounds for s in axes(spikes, 1)
        T = round(Int, τ / dt)
        rates[s, round(Int, τ)+1:end] = trolling_mean((@view my_spikes[s,:]), T)[1:10:L] ./ (dt / 1000)
    end
    return rates
end

function rolling_mean(a, n::Int)
    @assert 1<=n<=length(a)
    out = similar(a, length(a)-n+1)
    out[1] = sum(a[1:n])
    for i in eachindex(out)[2:end]
        out[i] = out[i-1]-a[i-1]+a[i+n-1]
    end
    return out ./n
end

function trolling_mean(a, n::Int)
    @assert 1<=n<=length(a)
    nseg=Threads.nthreads()
    if nseg*n >= length(a)
        return rolling_mean(a,n)
    else
        out = similar(a, length(a)-n+1)
        lseg = (length(out)-1)÷nseg+1
        segments = [(i*lseg+1, min(length(out),(i+1)*lseg)) for i in 0:nseg-1]
        for (start, stop) in segments
            out[start] = sum(a[start:start+n-1])
            for i in start+1:stop
                out[i] = out[i-1]-a[i-1]+a[i+n-1]
            end
        end
        return out ./n
    end
end


"""
    spiketimes_from_bool(P, τ; dt = 0.1ms)

This function takes in the records of a neural population `P` and time constant `τ` to calculate spike times for each neuron.

# Arguments
- `P`: A data structure containing the recorded data of a neuronal population.
- `τ`: A time constant parameter.

# Keyword Arguments
- `dt`: The time step used for the simulation, defaults to 0.1 milliseconds.

# Returns
- `spiketimes`: An object of type `SNN.Spiketimes` which contains the calculated spike times of each neuron.

# Examples
```
julia
spiketimes = spike_times(population_records, time_constant)
```
"""
function spiketimes_from_bool(P; dt = 0.1ms)
    spikes = hcat(P.records[:fire]...)
    _spiketimes = Vector{Vector{Float32}}()
    for (n, z) in enumerate(eachrow(spikes))
        push!(_spiketimes, findall(z) * dt)
    end
    return SNN.Spiketimes(_spiketimes)
end
