"""
    LIF
A leaky-integrate-fire neuron described by the following differential equation
``\\frac{\\mathrm{d}v}{\\mathrm{d}t} = \\frac{R}{\\tau} I - \\lambda``
# Fields
- `voltage::VT`: membrane potential
- `current::VT`: injected (unprocessed) current
- `lastt::IT`: the last time this neuron processed a spike
- `τm::VT`: membrane time constant
- `vreset::VT`: reset voltage potential
- `R::VT`: resistive constant (typically = 1)
"""
mutable struct LIF{VT<:Real, IT<:Integer} <: AbstractCell
    # required fields
    voltage::VT
    current::VT

    # model specific fields
    lastt::IT
    τm::VT
    vreset::VT
    R::VT
end

Base.show(io::IO, ::MIME"text/plain", neuron::LIF) =
    print(io, """LIF:
                     voltage: $(neuron.voltage)
                     current: $(neuron.current)
                     τm:      $(neuron.τm)
                     vreset:  $(neuron.vreset)
                     R:       $(neuron.R)""")
Base.show(io::IO, neuron::LIF) =
    print(io, "LIF(voltage = $(neuron.voltage), current = $(neuron.current))")

"""
    LIF(τm, vreset, vth, R = 1.0)
Create a LIF neuron with zero initial voltage and empty current queue.
"""
LIF(τm::Real, vreset::Real, R::Real = 1.0) = LIF{Float32, Int}(vreset, 0, 0, τm, vreset, R)

isactive(neuron::LIF, t::Integer; dt::Real = 1.0) = (neuron.current > 0)
getvoltage(neuron::LIF) = neuron.voltage

"""
    excite!(neuron::LIF, current)
    excite!(neurons::AbstractArray{<:LIF}, current)
Excite a `neuron` with external `current`.
"""
excite!(neuron::LIF, current) = (neuron.current += current)
excite!(neurons::T, current) where T<:AbstractArray{<:LIF} = (neurons.current .+= current)

"""
    spike!(neuron::LIF, t::Integer; dt::Real = 1.0)
    spike!(neurons::AbstractArray{<:LIF}, spikes; dt::Real = 1.0)
Record a output spike from the threshold function with the `neuron` body.
Sets `neuron.lastt`.
"""
spike!(neuron::LIF, t::Integer; dt::Real = 1.0) = (t > 0) && (neuron.voltage = neuron.vreset)
function spike!(neurons::T, spikes; dt::Real = 1.0) where T<:AbstractArray{<:LIF}
    map!((x, y, s) -> (s > 0) ? y : x, neurons.voltage, neurons.voltage, neurons.vreset, spikes)
end

"""
    lif(t::Real, I, V; R, tau)
    lif!(t::AbstractArray{<:Real}, I::AbstractArray{<:Real}, V::AbstractArray{<:Real}; R::AbstractArray{<:Real}, tau::AbstractArray{<:Real})
    lif!(t::CuVector{<:Real}, I::CuVector{<:Real}, V::CuVector{<:Real}; R::CuVector{<:Real}, tau::CuVector{<:Real})
Evaluate a leaky integrate-and-fire neuron.
Use `CuVector` instead of `Vector` to evaluate on GPU.
# Fields
- `t`: time since last evaluation in seconds
- `I`: external current
- `V`: current membrane potential
- `vrest`: resting membrane potential
- `R`: resistance constant
- `tau`: time constant
"""
function lif(t::Real, I, V; vrest, R, tau)
    V = V * exp(-t / tau) + vrest
    V += I * (R / tau)

    return V
end
function lif!(t::AbstractArray{<:Real}, I::AbstractArray{<:Real}, V::AbstractArray{<:Real}; vrest::AbstractArray{<:Real}, R::AbstractArray{<:Real}, tau::AbstractArray{<:Real})
    # account for leakage
    @. V = V * exp(-t / tau) + vrest

    # apply update step
    @. V += I * (R / tau)

    return V
end
function lif!(t::CuVector{<:Real}, I::CuVector{<:Real}, V::CuVector{<:Real}; vrest::CuVector{<:Real}, R::CuVector{<:Real}, tau::CuVector{<:Real})
    # account for leakage
    @. V = V * exp(-t / tau) + vrest

    # apply update step
    @. V += I * (R / tau)

    return V
end

"""
    evaluate!(neuron::LIF, t::Integer; dt::Real = 1.0)
    (neuron::LIF)(t::Integer; dt::Real = 1.0)
    evaluate!(neurons::AbstractArray{<:LIF}, t::Integer; dt::Real = 1.0)
Evaluate the neuron model at time `t`. Return the resulting membrane potential.
"""
function evaluate!(neuron::LIF, t::Integer; dt::Real = 1.0)
    neuron.voltage = lif((t - neuron.lastt) * dt, neuron.current, neuron.voltage; vrest = 0, R = neuron.R, tau = neuron.τm)
    neuron.lastt = t
    neuron.current = 0

    return neuron.voltage
end
(neuron::LIF)(t::Integer; dt::Real = 1.0) = evaluate!(neuron, t; dt = dt)
function evaluate!(neurons::T, t::Integer; dt::Real = 1.0) where T<:AbstractArray{<:LIF}
    lif!((t .- neurons.lastt) .* dt, neurons.current, neurons.voltage;
         vrest = zero(neurons.voltage), R = neurons.R, tau = neurons.τm)
    neurons.lastt .= t
    neurons.current .= 0

    return neurons.voltage
end

"""
    reset!(neuron::LIF)
    reset!(neurons::AbstractArray{<:LIF})
Reset `neuron` by setting the membrane potential to `neuron.vreset`.
"""
function reset!(neuron::LIF)
    neuron.lastt = 0
    neuron.voltage = neuron.vreset
    neuron.current = 0
end
function reset!(neurons::T) where T<:AbstractArray{<:LIF}
    neurons.lastt .= 0
    neurons.voltage .= neurons.vreset
    neurons.current .= 0
end
