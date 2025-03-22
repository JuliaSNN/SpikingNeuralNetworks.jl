"""
    sim!(
        P::Vector{TN},
        C::Vector{TS};
        dt = 0.1f0,
        duration = 10.0f0,
        pbar = false,
    ) where {TN <: AbstractPopulation, TS<:AbstractConnection }

Simulates the spiking neural network for a specified duration by repeatedly calling `sim!` function.

**Arguments**
- `P::Vector{TN}`: Vector of neurons in the network.
- `C::Vector{TS}`: Vector of synapses in the network.
- `dt::Float32`: Time step for the simulation. Default value is `0.1f0`.
- `duration::Float32`: Duration of the simulation. Default value is `10.0f0`.
- `pbar::Bool`: Flag indicating whether to display a progress bar during the simulation. Default value is `false`.

**Details**
- The function creates a range of time steps from `0.0f0` to `duration-dt` with a step size of `dt`.
- If `pbar` is `true`, the function creates a progress bar using the `ProgressBar` function with the time step range. Otherwise, it uses the time step range directly.
- The function iterates over the time steps and calls the `sim!` function with `P`, `C`, and `dt`.

"""
function sim!(
    P::Vector{TP},
    C::Vector{TC} = [EmptySynapse()],
    S::Vector{TS} = [EmptyStimulus()];
    dt = 0.125f0,
    duration = 10.0f0,
    pbar = false,
    time = Time(),
) where {TP<:AbstractPopulation,TC<:AbstractConnection,TS<:AbstractStimulus}
    dt = Float32(dt)
    duration = Float32(duration)
    dts = 0.0f0:dt:(duration-dt)
    pbar = pbar ? ProgressBar(dts) : dts
    for t in pbar
        sim!(P, C, S, dt, time)
    end
    return time
end



"""
    train!(
        P::Vector{TN},
        C::Vector{TS};
        dt = 0.1ms,
        duration = 10ms,
    ) where {TN <: AbstractPopulation, TS<:AbstractConnection }

Trains the spiking neural network for a specified duration by repeatedly calling `train!` function.

**Arguments**
- `P::Vector{TN}`: Vector of neurons in the network.
- `C::Vector{TS}`: Vector of synapses in the network.
- `dt::Float32`: Time step for the training. Default value is `0.1ms`.
- `duration::Float32`: Duration of the training. Default value is `10ms`.

**Details**
- The function converts `dt` to `Float32` if it is not already.
- The function creates a progress bar using the `ProgressBar` function with a range of time steps from `0.0f0` to `duration-dt` with a step size of `dt`.
- The function iterates over the time steps and calls the `train!` function with `P`, `C`, and `dt`.

"""
function train!(
    P::Vector{TP},
    C::Vector{TC} = [EmptySynapse()],
    S::Vector{TS} = [EmptyStimulus()];
    dt = 0.125ms,
    duration = 10ms,
    time = Time(),
    pbar = false,
) where {TP<:AbstractPopulation,TC<:AbstractConnection,TS<:AbstractStimulus}
    dt = Float32(dt)
    dts = 0.0f0:dt:(duration-dt)
    pbar = pbar ? ProgressBar(dts) : dts
    for t in pbar
        train!(P, C, S, dt, time)
    end
    return time
end


function train!(args...; model, kwargs...)
    pop =
        haskey(model, :pop) ? Vector{AbstractPopulation}(collect(model.pop)) :
        Vector{AbstractPopulation}([])
    syn =
        haskey(model, :syn) ? Vector{AbstractConnection}(collect(model.syn)) :
        Vector{AbstractConnection}([])
    stim =
        haskey(model, :stim) ? Vector{AbstractStimulus}(collect(model.stim)) :
        Vector{AbstractStimulus}([])
    for arg in args
        if isa(arg, AbstractPopulation)
            push!(pop, arg)
        elseif isa(arg, AbstractConnection)
            push!(syn, arg)
        elseif isa(arg, AbstractStimulus)
            push!(stim, arg)
        else
            error("Invalid argument type: $(typeof(arg))")
        end
    end
    mytime = train!(pop, syn, stim; time = model.time, kwargs...)
    update_time!(model.time, mytime)
    return get_time(model.time)
end

function sim!(args...; model, kwargs...)
    pop =
        haskey(model, :pop) ? Vector{AbstractPopulation}(collect(model.pop)) :
        Vector{AbstractPopulation}([])
    syn =
        haskey(model, :syn) ? Vector{AbstractConnection}(collect(model.syn)) :
        Vector{AbstractConnection}([])
    stim =
        haskey(model, :stim) ? Vector{AbstractStimulus}(collect(model.stim)) :
        Vector{AbstractStimulus}([])
    for arg in args
        if isa(arg, AbstractPopulation)
            push!(pop, arg)
        elseif isa(arg, AbstractConnection)
            push!(syn, arg)
        elseif isa(arg, AbstractStimulus)
            push!(stim, arg)
        else
            error("Invalid argument type: $(typeof(arg))")
        end
    end
    mytime = sim!(pop, syn, stim; time = model.time, kwargs...)
    update_time!(model.time, mytime)
    return get_time(model.time)
    # sim!(collect(model.pop), collect(model.syn), collect(model.stim); kwargs...)
end

function initialize!(; kwargs...)
    train!(; kwargs...)
end

#########

function sim!(
    P::Vector{TP},
    C::Vector{TC},
    S::Vector{TS},
    dt::Float32,
    T::Time,
) where {TP<:AbstractPopulation,TC<:AbstractConnection,TS<:AbstractStimulus}
    record_zero!(P, C, S, T)
    update_time!(T, dt)
    for s in S
        stimulate!(s, getfield(s, :param), T, dt)
        record!(s, T)
    end
    for p in P
        integrate!(p, getfield(p, :param), dt)
        record!(p, T)
    end
    for c in C
        forward!(c, getfield(c, :param))
        record!(c, T)
    end
end

function train!(
    P::Vector{TP},
    C::Vector{TC},
    S::Vector{TS},
    dt::Float32,
    T::Time,
) where {TP<:AbstractPopulation,TC<:AbstractConnection,TS<:AbstractStimulus}
    record_zero!(P, C, S, T)
    update_time!(T, dt)
    for s in S
        stimulate!(s, getfield(s, :param), T, dt)
        record!(s, T)
    end
    for p in P
        integrate!(p, p.param, dt)
        plasticity!(p, p.param, dt, T)
        record!(p, T)
    end
    for c in C
        forward!(c, c.param)
        ## I moved the control in the plasticity function for the sake of cleaner code
        plasticity!(c, c.param, dt, T)
        record!(c, T)
    end
end

function record_zero!(P, C, S, T)
    get_time(T) > 0.0f0 && return
    for p in P
        record!(p, T)
    end
    for c in C
        record!(c, T)
    end
    for s in S
        record!(s, T)
    end
end

export sim!, train!
