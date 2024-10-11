"""
    sim!(P::Vector{TN}, C::Vector{TS}, dt::Float32) where {TN <: AbstractNeuron, TS<:AbstractSynapse }

Simulates the spiking neural network by iterating over the populations and synapses in the network and updating their states.

**Arguments**
- `P::Vector{TN}`: Vector of neurons in the network.
- `C::Vector{TS}`: Vector of synapses in the network.
- `dt::Float32`: Time step for the simulation.

**Details**
- The function then calls the `integrate!` function on `p` with its parameters and the time step `dt`.
- Finally, the function calls the `record!` function on `p` to record its state.

- For each synapse `c` in `C`, the function checks if `c` has a field `t` and increments its value by 1 if it exists.
- The function then calls the `forward!` function on `c` with its parameters.
- Finally, the function calls the `record!` function on `c` to record its state.

"""
function sim!(
    P::Vector{TN},
    C::Vector{TS},
    dt::Float32,
) where {TN<:AbstractNeuron,TS<:AbstractSynapse}
    # Threads.@threads 
    for p in P
        hasfield(typeof(p), :t) && (p.t[1] += 1)
        integrate!(p, getfield(p, :param), dt)
        record!(p)
    end
    # Threads.@threads 
    for c in C
        hasfield(typeof(c), :t) && (c.t[1] += 1)
        forward!(c, getfield(c, :param))
        record!(c)
    end
end

"""
    sim!(
        P::Vector{TN},
        C::Vector{TS};
        dt = 0.1f0,
        duration = 10.0f0,
        pbar = false,
    ) where {TN <: AbstractNeuron, TS<:AbstractSynapse }

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
    P::Vector{TN},
    C::Vector{TS} = [EmptySynapse()];
    dt = 0.1f0,
    duration = 10.0f0,
    pbar = false,
) where {TN<:AbstractNeuron,TS<:AbstractSynapse}
    dt = Float32(dt)
    duration = Float32(duration)
    dts = 0.0f0:dt:(duration-dt)
    pbar = pbar ? ProgressBar(dts) : dts
    for t in pbar
        sim!(P, C, dt)
    end
end


"""
    train!(P::Vector{TN}, C::Vector{TS}, dt::Float32) where {TN <: AbstractNeuron, TS<:AbstractSynapse }

Trains the spiking neural network by iterating over the neurons and synapses in the network and updating their states.

**Arguments**
- `P::Vector{TN}`: Vector of neurons in the network.
- `C::Vector{TS}`: Vector of synapses in the network.
- `dt::Float32`: Time step for the training.

**Details**
- For each neuron `p` in `P`, the function checks if `p` has a field `t` and increments its value by 1 if it exists.
- The function then calls the `integrate!` function on `p` with its parameters and the time step `dt`.
- Finally, the function calls the `record!` function on `p` to record its state.

- For each synapse `c` in `C`, the function checks if `c` has a field `t` and increments its value by 1 if it exists.
- The function then calls the `forward!` function on `c` with its parameters.
- If `c` has a field `t` and its second element is approximately equal to 0, the function continues to the next iteration.
- Otherwise, the function calls the `plasticity!` function on `c` with its parameters and the time step `dt`.
- Finally, the function calls the `record!` function on `c` to record its state.

"""
function train!(
    P::Vector{TN},
    C::Vector{TS},
    dt::Float32,
) where {TN<:AbstractNeuron,TS<:AbstractSynapse}
    for p in P
        hasfield(typeof(p), :t) && (p.t[1] += 1)
        integrate!(p, p.param, dt)
        record!(p)
    end
    for c in C
        hasfield(typeof(c), :t) && (c.t[1] += 1)
        hasfield(typeof(c), :t) && (c.t[1] += 1)
        forward!(c, c.param)
        hasfield(typeof(c), :t) && (c.t[2] ≈ 0 && continue)
        plasticity!(c, c.param, dt)
        record!(c)
    end
end

"""
    train!(
        P::Vector{TN},
        C::Vector{TS};
        dt = 0.1ms,
        duration = 10ms,
    ) where {TN <: AbstractNeuron, TS<:AbstractSynapse }

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
    P::Vector{TN},
    C::Vector{TS};
    dt = 0.1ms,
    duration = 10ms,
) where {TN<:AbstractNeuron,TS<:AbstractSynapse}
    dt = Float32(dt)
    pbar = ProgressBar(0.0f0:dt:(duration-dt))
    # pbar = 0.0f0:dt:(duration-dt)
    for t in pbar
        train!(P, C, dt)
    end
end

function train!(;model, dt = 0.1ms, duration = 10ms)
    train!(collect(model.pop), collect(model.syn); dt = dt, duration = duration)
end

function sim!(;model, dt = 0.1ms, duration = 10ms)
    sim!(collect(model.pop), collect(model.syn); dt = dt, duration = duration)
end



export sim!, train!
