using RollingFunctions
using Interpolations
using StatsBase
using DSP

function init_spiketimes(N)
    _s = Vector{Vector{Float32}}()
    for i = 1:N
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

Returns:
- `spiketimes`: A vector of vectors containing the spike times of each neuron.
"""
function spiketimes(
    p::T;
    interval = nothing,
    kwargs...,
) where {T<:Union{AbstractPopulation,AbstractStimulus}}
    _spiketimes = init_spiketimes(p.N)

    firing_time = p.records[:fire][:time]
    neurons = p.records[:fire][:neurons]

    if length(firing_time) < 2
        # @warn "No spikes in population"
        return _spiketimes
    end
    if isnothing(interval)
        interval = (0, firing_time[end]+1)
    end
    tt0, tt1 = findfirst(x -> x > interval[1], firing_time),
    findlast(x -> x < interval[end], firing_time)
    if isnothing(tt0) || isnothing(tt1)
        return _spiketimes
    end
    for tt = tt0:tt1
        for n in neurons[tt]
            push!(_spiketimes[n], firing_time[tt])
        end
    end
    return _spiketimes
end



"""
    spiketimes(Ps; kwargs...)

    Return the spiketimes of each population in single vector of Spiketimes.
"""
function spiketimes(Ps; kwargs...)
    st = Vector{Vector{Float32}}[]
    for p in Ps
        _st = spiketimes(p; kwargs...)
        st = vcat(st, _st)
    end
    return Spiketimes(st)
end

"""
    spiketimes_split(Ps; kwargs...)

    Return the spiketimes of each population in a vector of Spiketimes.
"""
function spiketimes_split(Ps; kwargs...)
    st_ps = Vector{Vector{Vector{Float32}}}()
    names = Vector{String}()
    for p in Ps
        _st = spiketimes(p; kwargs...)
        push!(st_ps, Spiketimes(_st))
        push!(names, p.name)
    end
    return st_ps, names
end


"""
    spikecount(model, Trange, neurons)

    Return the total number of spikes of the neurons in the selected interval
"""
function spikecount(
    pop::T,
    Trange::Q,
    neurons::Vector{Int},
) where {T<:AbstractPopulation,Q<:AbstractVector}
    return length.(spiketimes(pop, interval = Trange)[neurons]) |> sum
end

export spikecount

# spikecount(x::Spiketimes) = length.(x)

# function alpha_function(t::T; t0::T, τ::T) where {T<:Float32}
#     return SNN.exp64(- (t - t0) / τ) * Θ((t - t0))
# end

# """
#     Θ(x::Float64)

#     Heaviside function
# """
# Θ(x::Float32) = x > 0.0 ? x : 0.0


"""
    alpha_function(t::T; t0::T, τ::T) where T <: AbstractFloat

    Alpha function for convolution of spiketimes. Evaluate the alpha function at time t, with time of peak t0 and time constant τ.
"""
function alpha_function(t::Float32, τ::Float32)
    if t >= 0
        return (t / τ) * exp(1 - t / τ)
    else
        return 0.0
    end
end

function get_alpha_kernel(τ, interval)
    kernel_length = τ * 10  # Length of the kernel in ms
    bin_width = step(interval) # kernel window in ms
    kernel_time = Float32.(0.0:bin_width:kernel_length)
    τ = Float32(τ)
    alpha_kernel = [alpha_function(t, τ) for t in kernel_time]
    alpha_kernel ./= sum(alpha_kernel) * bin_width
    return alpha_kernel
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


function k_fold(vector, k, do_shuffle = false)
    if do_shuffle
        ns = shuffle(1:length(vector))
    else
        ns = 1:length(vector)
    end
    b = length(ns) ÷ k
    indices=Vector{Vector{Int}}()
    for i = 1:(k-1)
        push!(indices, ns[((i-1)*b+1):(b*i)])
    end
    push!(indices, ns[(1+b*(k-1)):end])
    return indices
end


function merge_spiketimes(spikes::Spiketimes;)
    sort(vcat(spikes...))
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
    spiketimes::Spiketimes;
    interval::AbstractVector = [],
    τ = 25ms,
    interpolate = true,
    pop_average = false,
    time_average = false,
    neurons = :ALL,
    kwargs...
)
    # Check if the interval is empty and create an interval
    interval = _retrieve_interval(interval; kwargs...)
    neurons = neurons == :ALL ? eachindex(spiketimes) : neurons
    rates = nothing
    if time_average
        return time_average_fr(spiketimes, interval, pop_average), interval
    end

    if length(spiketimes) < 1
        rates = zeros(Float32, 0, length(interval))
    elseif all(isempty.(spiketimes))
        rates = zeros(Float32, length(spiketimes[neurons]), length(interval))
    else
        spiketimes = spiketimes[neurons]
        alpha_kernel = get_alpha_kernel(τ, interval)
        rates = map(eachindex(spiketimes)) do n
            spike_train, _ =
                bin_spiketimes(spiketimes[n]; interval = interval, do_sparse = false)
            conv(spike_train, alpha_kernel)[1:length(interval)] .* s # times
        end
        rates = hcat(rates...)'
    end

    if interpolate
        interp = get_interpolator(rates)
        rates = Interpolations.scale(
            Interpolations.interpolate(rates, interp),
            1:length(spiketimes),
            interval,
        )
    else
        rates = copy(hcat(rates...)')
    end

    if pop_average
        rates = mean(rates, dims = 1)[1, :]
    end
    return rates, interval
end

function _retrieve_interval(interval;
    sampling = 20ms,
    ttf = -1,
    tt0 = -1,
    kwargs...)
    if isempty(interval)
        max_time =
            all(isempty.(spiketimes)) ? 1.0f0 : maximum(Iterators.flatten(spiketimes))
        tt0 = tt0 > 0 ? tt0 : 0.0f0
        ttf = ttf > 0 ? ttf : max_time
        interval = tt0:sampling:ttf
    end
    return interval
end

function time_average_fr(spiketimes, interval, pop_average)
    rates = sum.(length.(spiketimes))./(interval[end] - interval[1])./Hz
    if pop_average
        rates = mean(rates)
        isnan(rates) && (rates = 0.0f0)
    else
        rates
    end
    return rates
end

firing_rate(P, interval::T; kwargs...) where {T<:AbstractRange} =
    firing_rate(P; interval, kwargs...)

function firing_rate(
    population::T;
    kwargs...,
) where {T<:Union{AbstractPopulation,AbstractStimulus}}
    return firing_rate(SNN.spiketimes(population); kwargs...)
end

function firing_rate(populations; mean_pop = false, kwargs...)
    spiketimes_pop, names_pop = spiketimes_split(populations)
    fr_pop = []
    interval = nothing
    for n in eachindex(spiketimes_pop)
        rates, interval = firing_rate(spiketimes_pop[n]; pop_average = mean_pop, kwargs...)
        push!(fr_pop, rates)
    end
    return fr_pop, interval, names_pop
end

function average_firing_rate(
    spiketimes::Spiketimes;
    interval::AbstractVector = [],
    sampling = 20ms,
    τ = 25ms,
    ttf = -1,
    tt0 = -1,
    cache = true,
)
    rates, interval = firing_rate(
        spiketimes;
        interval = interval,
        sampling = sampling,
        τ = τ,
        ttf = ttf,
        tt0 = tt0,
        cache = cache,
        pop = pop,
        interpolate = false,
    )
    return mean.(rates)
end

function average_firing_rate(populations; interval)
    spiketimes = SNN.spiketimes(populations)
    return sort(vcat(spiketimes...)) |> x -> fit(Histogram, x, interval).weights,
    interval[1:(end-1)]
end

"""
    autocor(spiketimes::Spiketimes; interval = 0:1:1000)

Calculate the cross-correlation of two spike trains.

# Arguments
- `spike_times1`: A vector of spike times for the first neuron.
- `spike_times2`: A vector of spike times for the second neuron.
- `bin_width`: The width of the time bins in milliseconds.
- `max_lag`: The maximum time lag in milliseconds.

# Returns
- `lags`: The time lags in milliseconds.
- `auto_corr`: The auto-correlation for each time lag.
"""
function compute_cross_correlogram(
    spike_times1::Vector{Float32},
    spike_times2::Vector{Float32} = Float32[];
    bin_width = 1.0ms,
    max_lag = 100.0,
    shift_predictor = false,
)
    # Create a binary spike train
    # bin_width = 1000/sr

    spike_train1, interval = bin_spiketimes(spike_times1; max_lag, bin_width)
    if !isempty(spike_times2)
        if shift_predictor
            spike_times2 = spike_times2 .+ 1000.0
        end
        spike_train2, _ = bin_spiketimes(spike_times2;)
    else
        spike_train2 = spike_train1
    end

    # Compute the auto-correlation (cross-correlogram with itself)
    _corr = xcorr(spike_train1, spike_train2)

    # Compute time lags
    bins = length(_corr) ÷ 2
    lags = ((-bins):bins) .* bin_width

    # Trim the lags and auto-correlation to the specified max_lag
    lag_mask = abs.(lags) .<= max_lag
    lags = lags[lag_mask]
    _corr = _corr[lag_mask]

    isempty(spike_times2) && (auto_corr[length(lags)÷2+1] = 0)

    return lags, _corr
end

"""
    compute_covariance_density(t_post, t_pre, T; τ=200ms, sr=50Hz)

Compute the covariance density of spike trains `t_post` and `t_pre` over a time interval `T`.
The function returns the covariance density vectors for positive and negative time lags.

# Arguments
- `spike_times1`:: Vector{Float32}: A vector of spike times for the first neuron.
- `spike_times2`:: Vector{Float32}: A vector of spike times for the second neuron.
- `bin_width`:: Float32: The width of the time bins in milliseconds.
- `max_lag`:: Float32: The maximum time lag in milliseconds.

# Returns
- `lags`:: Vector{Float32}: The time lags in milliseconds.
- `covariance_density`:: Vector{Float32}: The covariance density for each time lag.
"""
function compute_covariance_density(
    spike_times1::Vector{Float32},
    spike_times2::Vector{Float32};
    bin_width = 1ms,
    max_lag = 200ms,
)
    # Compute the cross-correlogram
    lags, cross_corr =
        compute_cross_correlogram(spike_times1, spike_times2; bin_width, max_lag)
    spike_train1, _ = bin_spiketimes(spike_times1; max_lag, bin_width)
    spike_train2, _ = bin_spiketimes(spike_times2; max_lag, bin_width)

    # Compute mean firing rates
    λ_x = mean(spike_train1) / bin_width
    λ_y = mean(spike_train2) / bin_width

    # Compute covariance density
    covariance_density = cross_corr .- (λ_x * λ_y * bin_width * length(spike_train1))

    return lags, covariance_density
end


"""
    bin_spiketimes(spiketimes, interval, sr)

Given a list of spike times `spiketimes`, an interval `[start, end]`, and a sampling rate `sr`,
this function counts the number of spikes that fall within each time bin of width `1/sr` within the interval.
The function returns a sparse matrix `count` containing the spike counts for each bin, and an array `r`
containing the time points corresponding to the center of each bin.

# Arguments
- `spiketimes`: A 1-dimensional array of spike times.
- `interval`: A 2-element array specifying the start and end times of the interval.
- `sr`: The sampling rate, i.e., the number of time bins per second.

# Returns
- `count`: A sparse matrix containing the spike counts for each time bin.
- `r`: An array of time points corresponding to the center of each time bin.
"""
function bin_spiketimes(
    spike_times::Vector{Float32};
    interval::AbstractRange,
    max_lag = 500ms,
    bin_width = 1.0ms,
    do_sparse = true,
)
    # interval =
    #     isempty(interval) ? (0.0:bin_width:(maximum(spike_times)+max_lag)) : interval
    bin_width = step(interval)
    spike_train = zeros(length(interval))
    st = sort(spike_times) .- first(interval)
    for i in findall(x -> x > 0 && x < length(interval)*bin_width, st)
        index = floor(Int, st[i] / bin_width) + 1
        if index <= length(spike_train)
            spike_train[index] += 1.0
        end
    end
    if do_sparse
        return SNN.sparse(spike_train), interval
    else
        return spike_train, interval
    end
end

function bin_spiketimes(spike_times::Spiketimes; kwargs...)
    sample, r = bin_spiketimes(spike_times[1]; kwargs..., do_sparse = false)
    bin_array = zeros(length(spike_times), length(sample))
    for n in eachindex(spike_times)
        bin_array[n, :] = bin_spiketimes(spike_times[n]; kwargs...)[1]
    end
    return bin_array, r
end

bin_spiketimes(P::AbstractPopulation; kwargs...) = bin_spiketimes(spiketimes(P); kwargs...)
bin_spiketimes(P::AbstractStimulus; kwargs...) = bin_spiketimes(spiketimes(P); kwargs...)
bin_spiketimes(P, interval::T; kwargs...) where {T<:AbstractRange} = bin_spiketimes(P; interval, kwargs...)

function bin_spiketimes(populations; interval, kwargs...)
    st_pops, names_pop = spiketimes_split(populations)
    ss = map(st->bin_spiketimes(st; interval, kwargs...)[1],eachindex(st_pops)) 
    return ss, interval, names_pop
end




function isi(spiketimes::Spiketimes)
    return diff.(spiketimes)
end

function isi(spiketimes::Vector{Float32})
    return diff(spiketimes)
end

function isi(pop::T; interval = nothing) where {T<:AbstractPopulation}
    return spiketimes(pop; interval = interval) |> isi
end

# isi(spiketimes::NNSpikes, pop::Symbol) = read(spiketimes, pop) |> x -> diff.(x)

function CV(spikes::Spiketimes)
    intervals = isi(spikes;)
    cvs = sqrt.(var.(intervals) ./ (mean.(intervals) .^ 2))
    cvs[isnan.(cvs)] .= -0.0
    return cvs
end

"""
spikes_in_interval(spiketimes::Spiketimes, interval::AbstractRange)

Return the spiketimes in the selected interval

# Arguments
spiketimes::Spiketimes: Vector with each neuron spiketime
interval: 2 dimensional array with the start and end of the interval

"""
function spikes_in_interval(
    spiketimes::Spiketimes,
    interval,
    margin = [0, 0];
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

function spikes_in_intervals(
    spiketimes::Spiketimes,
    intervals::Vector{Vector{Float32}};
    margin = 0,
    floor = true,
)
    st = tmap(intervals) do interval
        spikes_in_interval(spiketimes, interval, margin)
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

# function isi_cv(spikes::Vector{NNSpikes}; kwargs...)
#     spiketimes = merge_spiketimes(spikes; kwargs...)
#     @unpack tt = spikes[end]
#     return CV_isi2.(spiketimes)
# end

isi_cv(x::Spiketimes) = CV_isi2.(x)

"""
    st_order(spiketimes::Spiketimes)
"""
function st_order(spiketimes::T) where {T<:Vector{}}
    ii = sort(eachindex(1:length(spiketimes)), by = x -> spiketimes[x])
    return ii
end

function st_order(spiketimes::Spiketimes, pop::Vector{Int}, intervals)
    @unpack spiketime = spike_statistics(spiketimes[pop], intervals)
    ii = sort(eachindex(pop), by = x -> spiketime[x])
    return pop[ii]
end

function st_order(
    spiketimes::Spiketimes,
    populations::Vector{Vector{Int}},
    intervals::Vector{Vector{T}},
    unique_pop::Bool = false,
) where {T<:Real}
    return [st_order(spiketimes, population, intervals) for population in populations]
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
# function firing_rate(P, τ; dt = 0.1ms)
#     spikes = hcat(P.records[:fire]...)
#     time_span = round(Int, size(spikes, 2) * dt)
#     rates = zeros(P.N, time_span)
#     L = round(Int, time_span - τ) * 10
#     my_spikes = Matrix{Int}(spikes)
#     @fastmath @inbounds for s in axes(spikes, 1)
#         T = round(Int, τ / dt)
#         rates[s, round(Int, τ)+1:end] =
#             trolling_mean((@view my_spikes[s, :]), T)[1:10:L] ./ (dt / 1000)
#     end
#     return rates
# end

function resample_spikes(X, Y)
    if length(X) > 200_000
        s = ceil(Int, length(X) / 200_000)
        points = Vector{Int}(eachindex(X))
        points = sample(points, 200_000, replace = false)
        X = X[points]
        Y = Y[points]
        @warn "Subsampling raster plot, 1 out of $s spikes"
    end
    return X, Y
end

function rolling_mean(a, n::Int)
    @assert 1 <= n <= length(a)
    out = similar(a, length(a) - n + 1)
    out[1] = sum(a[1:n])
    for i in eachindex(out)[2:end]
        out[i] = out[i-1] - a[i-1] + a[i+n-1]
    end
    return out ./ n
end

function trolling_mean(a, n::Int)
    @assert 1 <= n <= length(a)
    nseg = Threads.nthreads()
    if nseg * n >= length(a)
        return rolling_mean(a, n)
    else
        out = similar(a, length(a) - n + 1)
        lseg = (length(out) - 1) ÷ nseg + 1
        segments = [(i * lseg + 1, min(length(out), (i + 1) * lseg)) for i = 0:(nseg-1)]
        for (start, stop) in segments
            out[start] = sum(a[start:(start+n-1)])
            for i = (start+1):stop
                out[i] = out[i-1] - a[i-1] + a[i+n-1]
            end
        end
        return out ./ n
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

"""
    sample_spikes(N, rate::Vector, interval::R; dt=0.125f0) where {R <: AbstractRange}

Generate sample spike times for N neurons from a rate vector.
The function generates spike times for each neuron based on the rate vector and the time interval. The spike times are generated such that during the interval of the rate, the number of spikes is Poisson distributed with the rate.

# Arguments
- `N`: The number of neurons.
- `rate::Vector`: The vector with the rate to be sampled.
- `interval::R`: The time interval over which the rate is recorded.
- `dt=0.125f0`: The time step size.


# Returns
An array of spike times for each neuron.

"""
function sample_spikes(
    N,
    rate::Vector,
    interval::R;
    rate_factor = 1.0f0,
    dt = 0.125f0,
) where {R<:AbstractRange}
    spiketimes = Vector{Float32}[[] for _ = 1:N]
    @assert length(rate) == length(interval)
    steps = step(interval) / dt
    t = dt + Float32(interval[1])
    for i in eachindex(interval)
        r = rate[i] * Hz
        for _ = 1:steps
            for n = 1:N
                if rand() < r * dt * rate_factor
                    push!(spiketimes[n], t)
                end
            end
            t = Float32(t + dt)
        end
    end
    spiketimes
end

function sample_inputs(
    N,
    rate::Matrix{F},
    interval::R;
    dt = 0.125f0,
    rate_factor = 1.0f0,
    seed = nothing,
) where {R<:AbstractRange, F<:Real}
    !isnothing(seed) && (Random.seed!(seed))
    inputs = Vector{Float32}[]
    for i = 1:size(rate, 1)
        for n in sample_spikes(N, rate[i, :], interval; dt = dt, rate_factor = rate_factor)
            push!(inputs, n)
        end
    end
    inputs
end

function sample_inputs(
    N,
    spikes::BitMatrix,
    interval::R;
    rate_factor,
    dt = 0.125f0,
    seed = nothing,
) where {R<:AbstractRange}
    !isnothing(seed) && (Random.seed!(seed))
    inputs = Vector{Float32}[]
    @assert size(spikes, 2) == length(interval)
    for i = 1:size(spikes, 1)
        st_n = findall(spikes[i, :])
        isnothing(st_n) && push!(inputs, Float32[])
        push!(inputs, interval[st_n])
    end
    inputs
end



export spiketimes,
    spiketimes_from_bool,
    merge_spiketimes,
    convolve,
    alpha_function,
    autocorrelogram,
    bin_spiketimes,
    compute_covariance_density,
    isi,
    CV,
    CV_isi2,
    firing_rate,
    average_firing_rate,
    firing_rate_average,
    firing_rate,
    firing_rate_average,
    spikes_in_interval,
    spikes_in_intervals,
    find_interval_indices,
    interval_standard_spikes,
    interval_standard_spikes!,
    relative_time!,
    st_order,
    isi_cv,
    CV_isi2
sample_spikes, sample_inputs
