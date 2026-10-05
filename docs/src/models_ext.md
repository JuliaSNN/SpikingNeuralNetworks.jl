# Model Extensions

New populations, connections, stimuli and plasticity rules are added by defining new concrete
subtypes of the abstract types of `SNNModels` and the methods that the simulation loop calls for
them. Because `sim!` and `train!` dispatch on the type of each component and on the type of its
`param` field, it is often enough to define a new *parameter* type and the method that specialises on
it, without a new component type.

## The simulation loop

At every time step `sim!` and `train!` do the following (see `src/utils/main.jl` in SNNModels):

<!-- norun -->
```julia
record_zero!(P, C, S, T)                  # only at t = 0: store the initial state in the records
update_time!(T, dt)
for s in S                                # stimuli
    stimulate!(s, s.param, T, dt)
    record!(s, T)
end
for p in P                                # populations
    update_traces!(p, p.param, dt, T)     # train! only
    integrate!(p, p.param, dt)
    plasticity!(p, p.param, dt, T)        # train! only
    record!(p, T)
end
for c in C                                # connections
    update_traces!(c, c.param, dt, T)     # train! only
    forward!(c, c.param, dt, T)
    plasticity!(c, c.param, dt, T)        # train! only
    record!(c, T)
end
```

`sim!` never calls `update_traces!` or `plasticity!`: the weights and the short-term plasticity
variables of a model stay fixed under `sim!`, even if its synapses carry plasticity rules.

## Interface of each component

**Populations** (`<: AbstractPopulation`, parameter `<: AbstractPopulationParameter`)

- Required fields: `N`, `param`, `id`, `name`, `records::Dict` (checked by
  `validate_population_model`; `monitor!` and `record!` store the recordings in `records`).
  A field `fire::Vector{Bool}` is needed to record spikes (`monitor!(p, [:fire])`) and to be the
  presynaptic population of a `SpikingSynapse`. Any other field (`v`, `I`, ...) can be recorded
  with `monitor!`.
- `integrate!(p, param, dt::Float32)`: advance the state by one step.
- `synaptic_target(targets::Dict, post, sym::Symbol, target)`: return `(g, v_post)`, the vector
  into which connections and stimuli add their input and the membrane potential used by
  voltage-dependent plasticity. Needed only if the population receives `SpikingSynapse`s or spiking
  stimuli; there is no generic fallback. Stimuli that write a current (`CurrentStimulus`) read the
  field `sym` directly.
- Optional: `Population(param::MyParameter; kwargs...)` so that `SNN.Population(; param, ...)`
  builds the new type, and `plasticity!`/`update_traces!` methods; the generic fallbacks for
  `AbstractPopulationParameter` do nothing.

**Connections** (`<: AbstractConnection`, parameter `<: AbstractConnectionParameter`)

- Required fields: `param`, `id`, `name`, `records::Dict` (checked by `validate_synapse_model`).
- `forward!(c, param, dt, T)` or the two-argument form `forward!(c, param)`: the generic
  four-argument method calls the two-argument one. This generic method, and the no-op
  `update_traces!` fallback, exist only for parameters that are subtypes of
  `AbstractConnectionParameter`; with any other parameter type `sim!` and `train!` fail with a
  `MethodError`.
- `plasticity!(c, param, dt::Float32, T::Time)`: called by `train!` for every connection; there is
  no generic fallback, so define it (it can return `nothing`) if the model is ever run with `train!`.
  `update_traces!` has a no-op fallback.

**Stimuli** (`<: AbstractStimulus`, parameter `<: AbstractStimulusParameter`)

- Required fields: `param`, `id`, `name`, `records::Dict` (checked by `validate_stimulus_model`).
- `stimulate!(s, param, T::Time, dt::Float32)`.
- Optional: a `Stimulus(param::MyParameter, post, sym; kwargs...)` method.

**Plasticity rules** for `SpikingSynapse` (`<: LTPParameter` or `<: STPParameter`)

- `plasticityvariables(param, Npre, Npost)`: return the state of the rule, a subtype of
  `PlasticityVariables` with an `active::Vector{Bool}` field (the rule runs only if
  `any(active)`).
- `plasticity!(c, param, vars, dt, T)` (applied after `forward!`) and, if the rule needs it,
  `update_traces!(c, param, vars, dt, T)` (applied before `forward!`). Long-term rules modify
  `c.W`, short-term rules the efficacy `c.ρ`.

The existing types in SNNModels are the best templates: see the [Model catalogue](catalogue/index.md).

## Adding a population model: current-based IF

The new types are defined inside `SNNModels` with `@eval`, which is equivalent to adding a file to
`SNNModels.jl/src/populations` and gives access to the internal macros (`@snn_kw`) and types.

A docstring cannot be attached directly to an `@snn_kw struct`; write the docstring above the bare
name of the type and define the struct after it, as below. `@snn_kw` generates a keyword constructor whose defaults may refer to earlier fields. Field types
must be plain names: a parametric type written in place, such as `Vector{Float32}`, is not accepted
by the macro (it fails with `Cannot convert an object of type Expr to an object of type Symbol`).
Declare vector types as type parameters with a default (`Neuron{VFT = Vector{Float32}}`), as the
library models do, or use the aliases `VBT = Vector{Bool}` and `VIT = Vector{Int}` defined in
SNNModels.

```julia
using SpikingNeuralNetworks
using Distributions
SNN.@load_units

@eval SNN.SNNModels begin
    """
        NeuronParameter

    Parameters of a current-based leaky integrate-and-fire neuron.
    """
    NeuronParameter

    @snn_kw struct NeuronParameter <: AbstractPopulationParameter
        R::Float32 = 1GΩ
        El::Float32 = -70.6mV
        Vt::Float32 = -50.4mV
        Vr::Float32 = -70.6mV
        τm::Float32 = 20ms
        τe::Float32 = 10ms
        τi::Float32 = 10ms
    end

    """
        Neuron

    Population of current-based leaky integrate-and-fire neurons. `N`, `param`, `name`, `id`
    and `records` are compulsory; `fire` is needed to record spikes and to project spikes.
    """
    Neuron

    @snn_kw struct Neuron{VFT = Vector{Float32}} <: AbstractPopulation
        param::NeuronParameter = NeuronParameter()
        N::Int = 10
        name::String = "Neuron"
        id::String = randstring(12)
        v::VFT = fill(param.El, N)
        ge::VFT = zeros(Float32, N)
        gi::VFT = zeros(Float32, N)
        fire::VBT = zeros(Bool, N)
        I::VFT = zeros(Float32, N)
        records::Dict = Dict()
    end

    # one forward Euler step; recordings are handled by the simulation loop
    function integrate!(p::Neuron, param::NeuronParameter, dt::Float32)
        @unpack N, v, ge, gi, fire, I = p
        @unpack R, El, Vt, Vr, τm, τe, τi = param
        @inbounds for i in 1:N
            v[i] += dt / τm * (El - v[i] + R * (ge[i] - gi[i] + I[i]))
            ge[i] -= dt * ge[i] / τe
            gi[i] -= dt * gi[i] / τi
            fire[i] = v[i] >= Vt
            fire[i] && (v[i] = Vr)
        end
    end

    # where SpikingSynapse and spiking stimuli deliver their input
    function synaptic_target(targets::Dict, post::Neuron, sym::Symbol, target = nothing)
        push!(targets, :sym => sym)
        return getfield(post, sym), post.v
    end
end

neuron = SNN.SNNModels.Neuron(N = 1)
current_param = SNN.CurrentNoise(neuron.N; I_base = 20pA, I_dist = Normal(0pA, 20pA))
current_stim = SNN.Stimulus(current_param, neuron, :I)
input = SNN.Stimulus(SNN.PoissonLayer(rate = 10Hz, N = 100), neuron, :ge; conn = (p = 1, μ = 1pA))

SNN.monitor!(neuron, [:v, :fire, :ge, :I], sr = 2kHz)
model = SNN.compose(; neuron, current_stim, input, silent = true)
SNN.sim!(; model, duration = 1000ms)
SNN.spiketimes(neuron)
```

## Adding a stimulus model: Poisson input with refractory period

Here a new parameter type, subtype of `PoissonStimulusParameter`, is enough: the existing
`PoissonStimulus` container and its `Stimulus` constructor are reused, and only a `stimulate!`
method for the new parameter is added. The stimulus adds `μ` to the target of each neuron when
the neuron's Poisson source emits a spike, with an absolute refractory period `ΔT` between
spikes.

```julia
using SpikingNeuralNetworks
SNN.@load_units

@eval SNN.SNNModels begin
    @snn_kw struct PoissonRefractory{VFT = Vector{Float32}} <: PoissonStimulusParameter
        rate::Float32 = 10Hz
        ΔT::Float32 = 2ms                       # absolute refractory period
        μ::Float32 = 1.0f0                      # increment per spike
        N::Int = 100                            # size of the target population
        last_spike::VFT = fill(-Inf32, N)
        active::VBT = [true]
    end

    function stimulate!(p::PoissonStimulus, param::PoissonRefractory, time::Time, dt::Float32)
        param.active[1] || return
        @unpack rate, ΔT, μ, last_spike = param
        t = get_time(time)
        @inbounds for n in p.neurons
            if t - last_spike[n] > ΔT && rand() < rate * dt
                p.g[n] += μ
                last_spike[n] = t
            end
        end
    end
end

neuron = SNN.Identity(N = 1, name = "Identity Neuron")
stim = SNN.Stimulus(SNN.SNNModels.PoissonRefractory(N = 1, rate = 100Hz, ΔT = 20ms), neuron, :g)
SNN.monitor!(neuron, [:fire])
model = SNN.compose(; neuron, stim, silent = true)
SNN.sim!(; model, duration = 2000ms)
isi = diff(SNN.spiketimes(neuron)[1])
@assert minimum(isi) > 20ms
```
