# Recordings

```@meta
CurrentModule = SNNModels
```

Any field of a population, connection or stimulus can be recorded during a simulation.
Recording is set up with `monitor!` before `sim!`/`train!`, and the data are read back with
`getvariable` (raw samples), `record` (interpolated in time), `spiketimes` and
`firing_rate`.

The examples on this page share the network below: an AdEx excitatory population and an IF
inhibitory population with recurrent connections, driven by Poisson input. The
excitatory recurrent connections have short-term plasticity (STP) and the inhibitory to
excitatory connections have inhibitory STDP (`iSTDPRate`). The code blocks of this page are
meant to be run in sequence.

<!-- run: sequential -->

```julia
using SpikingNeuralNetworks
using Statistics
SNN.@load_units

E = SNN.AdEx(; N = 800, param = SNN.AdExParameter(; El = -50mV))
I = SNN.IF(; N = 200, param = SNN.IFParameter())
EE = SNN.SpikingSynapse(E, E, :he; conn = (μ = 2, p = 0.02), STPParam = SNN.MarkramSTPParameter())
EI = SNN.SpikingSynapse(E, I, :ge; conn = (μ = 30, p = 0.02))
IE = SNN.SpikingSynapse(I, E, :hi; conn = (μ = 50, p = 0.02), LTPParam = SNN.iSTDPRate(r = 5Hz))
II = SNN.SpikingSynapse(I, I, :gi; conn = (μ = 10, p = 0.02))
input_E = SNN.Stimulus(SNN.PoissonFixed(rate = 2kHz), E, :ge)
input_I = SNN.Stimulus(SNN.PoissonFixed(rate = 2kHz), I, :ge)
model = SNN.compose(; E, I, EE, EI, IE, II, input_E, input_I)
```

## Monitoring variables

`monitor!(obj, keys; sr)` registers the fields `keys` of `obj` for recording. `obj` can be
a single component or a collection such as `model.pop`. The sampling rate `sr` defaults to
`1000Hz` for a single component and to `200Hz` when a collection is passed; the sampling
period is `floor(1 / (sr * dt))` simulation steps. `:fire` is special: it records every
spike, independently of `sr`.

```julia
SNN.monitor!(E, [:ge, :gi], sr = 200Hz)
SNN.monitor!(model.pop, :v, sr = 200Hz)
SNN.monitor!(model.pop, :fire)
SNN.sim!(; model, duration = 2s)
```

Recording buffers are allocated just before the time loop, sized from the simulation
duration (at least 1 s) or from the `monitor_time` keyword of `monitor!`; if a run is longer,
the buffer grows automatically (with a one-time warning). For spikes, `monitor_rate`
(default 20 Hz) sets the expected maximum firing rate used for the initial allocation.

A tuple `(key, indices)` restricts the recording of a vector field to some neurons, e.g.
`SNN.monitor!(E, [(:v, 1:10)])`. For `:fire` only the spikes of the listed neurons are
recorded, with their original indices. (In SNNModels 1.8.4 the indices were ignored for
`:fire`.)

!!! note
    It is not possible to pause a recording while keeping the variable monitored. To start
    again from scratch, use `clear_records!(obj)` (deletes the data, keeps the set-up) or
    `clear_monitor!(obj)` (removes the set-up as well).

## Reading variables

`getvariable(obj, key)` returns the raw samples, with time as the last dimension
(neurons x samples for a vector field). When the model time is 0 at the start, the first
sample is taken at `t = 0`.

`record(obj, key)` returns the recording as an interpolation object over continuous time:
it is evaluated with call syntax, `v(i, t)`, at any time `t` (in ms) within the recorded
range (`i` and `t` can be ranges), and `record(obj, key, range = true)` also returns the
time axis `r`.

Samples are taken at the global steps that are multiples of the sampling period
(`round(1/(sr*dt))` steps), and `r` runs from the first to the last sample actually taken, so it
is the exact sample-time axis also when a variable is monitored from a later time (as `:W`
below, monitored from 2 s on: samples at 2.1, 2.2, ..., 4 s at 10 Hz).

!!! note "Changed after SNNModels 1.8.4"
    In SNNModels 1.8.4 `r` ran from the first to the last simulated step of the monitored
    period, so it was shifted or stretched by up to one sampling period when monitoring
    started after `t = 0` or the duration was not a multiple of `1/sr`; the period was
    rounded down (10 Hz sampled every 99.875 ms at `dt = 0.125ms`).

```julia
v = SNN.getvariable(E, :v)                    # 800 x 401 samples (2 s at 200 Hz, plus t = 0)
v = SNN.record(E, :v)                         # interpolated
v(1, 1.5s)
v(1:10, 1.2s:15ms:1.9s)
v, r = SNN.record(E, :v, range = true)
size(v), r
v = SNN.record(E, :v, interpolate = false)    # same as getvariable
```

### Spike times and firing rates

`spiketimes(pop)` returns a `Spiketimes` object, a vector with the spike times (ms) of each
neuron. The keyword `interval` restricts it to a time window.

```julia
st = SNN.spiketimes(E)
length(st), length(st[1])
st = SNN.spiketimes(E; interval = 0:1ms:1s)
```

`bin_spiketimes(pop; interval)` counts the spikes in the bins defined by the range
`interval` and returns `(counts, r)`, with `counts` a neurons x bins matrix.

```julia
interval = 0:10ms:2s
bins, r = SNN.bin_spiketimes(E; interval)
size(bins)
```

`firing_rate(pop; interval)` returns `(rates, r)`: the firing rate of each neuron (Hz)
sampled on `interval`, obtained by convolving the spike trains with a kernel (alpha
function by default). With `interpolate = true` (default) `rates` is an interpolation
object; `pop_average = true` averages over neurons. On a collection of populations it
returns `(rates, r, names)`, one entry per population.

```julia
fr, r = SNN.firing_rate(E; interval)
fr, r = SNN.firing_rate(E; interval, interpolate = false)
fr, r, names = SNN.firing_rate(model.pop; interval)
names
```

The same quantities are available through `record`:

```julia
fr = SNN.record(E, :fire; interval)
fr, r = SNN.record(E, :fire; interval, range = true)
st = SNN.record(E, :spikes)
```

!!! note
    The component created in `Main` (`E`) and the one in the model (`model.pop.E`) are the
    same object. Monitoring or reading either is equivalent.

## Synaptic variables

Fields of connections are monitored in the same way. Per-synapse fields such as the weights
`:W` and the STP efficacy `:ρ` have one value per synapse. Plasticity variables live in
nested structures and need the name of the structure, given with the keyword `variables` or
as a third positional argument: `:STPVars` for short-term plasticity (`:u`, `:x` of
`MarkramSTPParameter`), `:LTPVars` for long-term plasticity (e.g. `:tpost`, the postsynaptic
trace of `iSTDPRate`). They are stored under the compound key `Symbol(variables, "_", key)`,
e.g. `:STPVars_u`.

```julia
SNN.monitor!(EE, [:ρ], sr = 10Hz)
SNN.monitor!(IE, [:W], sr = 10Hz)
SNN.monitor!(IE, [:tpost]; sr = 10Hz, variables = :LTPVars)
SNN.monitor!(EE, [:x, :u], :STPVars; sr = 10Hz)
SNN.train!(; model, duration = 2s)
```

!!! note "Plasticity needs `train!`"
    The first simulation of this page used `sim!`, which never updates weights or STP
    variables. The call above uses `train!`, so `IE` (inhibitory STDP) and `EE` (STP) are
    plastic. If you ran this example with SNNModels 1.5.0 to 1.8.1 the `IE` weights were
    potentiated at the wrong synapses (see [Release notes](release_notes.md)); results
    differ from SNNModels 1.8.2 on.

!!! warning
    Recording per-synapse variables can use a lot of memory in large networks; use a low
    sampling rate.

### Weight matrices

Connectivity is stored in sparse form; `matrix(c)` returns the current weights as an
`N_post x N_pre` sparse matrix, and `matrix(c, sym)` any other per-synapse field.

```julia
W = SNN.matrix(EE)            # current weights
ρ = SNN.matrix(EE, :ρ)        # current STP efficacy
```

`presynaptic(c, i)` returns the presynaptic neurons of postsynaptic neuron `i`, and
`postsynaptic(c, j)` the postsynaptic neurons of presynaptic neuron `j`; both also accept a
vector of neurons.

```julia
neuron = 1
Is = SNN.postsynaptic(EE, neuron)
mean(W[Is, neuron])           # mean weight of the outgoing connections of neuron 1
Js = SNN.presynaptic(EE, neuron)
mean(W[neuron, Js])           # mean weight of the incoming connections of neuron 1
Js_many = SNN.presynaptic(EE, 1:10)
```

A recorded per-synapse variable is a synapses x samples array; `record` interpolates it in
time, and `matrix_record(c, sym, t)` rebuilds the sparse matrix at time `t` (a vector of
times gives a 3-dimensional array). The times are those of the `train!` call above, from
2 s to 4 s.

```julia
W_IE, r = SNN.record(IE, :W, range = true)
W_IE(axes(W_IE, 1), 3.5s)                      # weights of all IE synapses at t = 3.5 s
W_mat = SNN.matrix_record(IE, :W, 3.5s)         # 800 x 200 sparse matrix
W_mat2 = SNN.matrix(IE, W_IE, 3.5s)             # same, from the interpolated record
all(W_mat .== W_mat2)
W_T = SNN.matrix_record(IE, :W, 3s:100ms:3.5s)  # 800 x 200 x 6
```

### Plasticity variables

Plasticity variables are read with their compound key:

```julia
x = SNN.record(EE, :STPVars_x)
x(1, 3.14s)
x, r = SNN.record(EE, :STPVars_x, range = true)
x_raw = SNN.getvariable(EE, :STPVars_x)
tpost = SNN.record(IE, :LTPVars_tpost)
tpost(1:10, 2.4s:15ms:3.1s)
```

!!! tip
    For plotting, see the functions of SNNPlots on the [Visualization](visualization.md)
    page, or use any plotting package directly on the interpolated records:
    <!-- norun -->
    ```julia
    using Plots
    v, r = SNN.record(E, :v, range = true)
    plot(r, v(1, r), label = "membrane potential of neuron 1")
    ```

## Recording API

```@autodocs
Modules = [SNNModels]
Pages   = ["utils/record.jl"]
```
