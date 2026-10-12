# Running simulations

```@meta
CurrentModule = SNNModels
```

After this page you can run a model with or without plasticity, continue a run, choose the time
step and the floating-point precision, use several threads, and save a model to disk.

## `sim!` and `train!`

```@example simulation
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100)
EE = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.1, μ = 0.5nS), LTPParam = SNN.STDPGerstner())
drive = SNN.Stimulus(SNN.PoissonFixed(rate = 2kHz), E, :ge)
model = SNN.compose(; E, EE, drive, silent = true)

SNN.sim!(model, 500ms)                  # weights frozen
SNN.train!(model, 500ms)                # STDP changes EE.W
SNN.sim!(; model, duration = 1s)       # keyword form
SNN.get_time(model.time)                # 2000.0 (ms): runs continue from the current time
```

`sim!` and `train!` run the same step; only `train!` applies plasticity. Both accept the model
positionally (`sim!(model, duration)`) or as a keyword (`sim!(; model, duration)`). The time step
is the keyword `dt` (default `0.125ms`). Further keywords: `pbar = true` (progress bar), `perturbation!` (a function called before each step),
`neurons_per_thread` and `flush_subnormals` (below).

One step does, in this order:

<!-- norun -->
```julia
update_time!(T, dt)                       # the clock advances first
for s in stimuli;     stimulate!(s, s.param, T, dt); record!(s, T); end
for p in populations
    update_traces!(p, p.param, dt, T)     # train! only
    integrate!(p, p.param, dt)
    plasticity!(p, p.param, dt, T)        # train! only
    record!(p, T)
end
for c in connections
    update_traces!(c, c.param, dt, T)     # train! only
    forward!(c, c.param, dt, T)
    plasticity!(c, c.param, dt, T)        # train! only
    record!(c, T)
end
```

Input written by a connection in one step is integrated by its target in the next. The spike
list of a population is built once per step and shared by its connections and recorder, so a
custom component must not write `pop.fire` between `integrate!` and the end of the step. A step
that throws is a fatal error: build the model again.

## Precision

Models run in `Float32` by default. Build the populations with `VFT = Vector{Float64}` for a
`Float64` model; connections and stimuli take the precision of the population they write into,
and `compose` gives the model a clock of the same type. `float_type` returns the precision of any
component.

```@example simulation
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100, VFT = Vector{Float64})
EE = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.1, μ = 0.5nS),
                        LTPParam = SNN.STDPGerstner(FT = Float64))
model = SNN.compose(; E, EE, silent = true)
SNN.train!(model, 100ms)
SNN.float_type(EE), eltype(model.time.t)    # (Float64, Float64)
```

A rule built in `Float32` is converted to the precision of its synapse but keeps its
`Float32`-rounded values; build it with `FT = Float64` for exact `Float64` parameters. Mixing a
`Float32` parameter object into a `Float64` population is an `ArgumentError`. A few components are
`Float32` only (see [Known limitations](release_notes.md#Known-limitations)).

## Threads

Start Julia with several threads (`julia -t 8`) and set the number of neurons per chunk:

```@example simulation
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 4000)
drive = SNN.Stimulus(SNN.PoissonFixed(rate = 2kHz), E, :ge)
model = SNN.compose(; E, drive, silent = true)

SNN.sim!(model, 100ms; neurons_per_thread = 512)    # this run only
SNN.set_neurons_per_thread!(512)                    # every following run
SNN.neurons_per_thread()                            # 512
SNN.set_neurons_per_thread!(0)                      # off (the default)
nothing # hide
```

Neuron updates, `forward!` and the event-driven STDP, iSTDP and STP updates of `SpikingSynapse`
are split by neuron range into at most 8 chunks. Results are bit-identical for every setting
and thread count. `set_optimal_neurons_per_thread!(model)` times a copy of the model for several
settings and keeps the fastest; `with_neurons_per_thread(f, n)` runs `f()` with a temporary
setting.

## Subnormal numbers

`sim!` and `train!` flush subnormal floating-point results to zero (`flush_subnormals = true`),
which keeps decaying variables of silent neurons and synapses from slowing the run down. Values
above `floatmin` are unaffected. Pass `flush_subnormals = false` for IEEE gradual underflow.

## Saving and loading

```@example simulation
using SpikingNeuralNetworks
E = SNN.IF(N = 10, name = "E")
model = SNN.compose(; E, silent = true)
dir = mktempdir()
SNN.save_model(; model, path = dir, name = "toy", info = (seed = 1,))
loaded = SNN.load_model(dir, "toy", (seed = 1,))
loaded.model.pop.E.N                    # 10
```

`info` is a `NamedTuple` of metadata that becomes part of the folder name. Files saved with
version 1 cannot be loaded (see [What changes from v1 to v2](index.md#What-changes-from-v1-to-v2)).

## API

```@autodocs
Modules = [SNNModels]
Pages   = ["utils/threading.jl", "utils/threading_tuner.jl", "utils/subnormals.jl"]
```
