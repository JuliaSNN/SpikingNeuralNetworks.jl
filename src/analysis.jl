using RollingFunctions

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
    spiketimes(P, τ; dt = 0.1ms)

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
function spiketimes(P; dt = 0.1ms)
    spikes = hcat(P.records[:fire]...)
    _spiketimes = Vector{Vector{Float32}}()
    for (n, z) in enumerate(eachrow(spikes))
        push!(_spiketimes, findall(z) * dt)
    end
    return SNN.Spiketimes(_spiketimes)
end
