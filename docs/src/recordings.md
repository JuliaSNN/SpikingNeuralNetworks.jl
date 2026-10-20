# Recordings

```@meta
CurrentModule = SNNModels
```

After this page you can record any variable of a population, connection or stimulus, and read
it back as raw samples, interpolated traces, spike times, firing rates or weight matrices.

The blocks of this page run in sequence on the network below (STP on `EE`, inhibitory STDP on `IE`).

<!-- run: sequential -->

```@example recordings
using Logging # hide
Logging.disable_logging(Logging.Warn) # hide
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
model = SNN.compose(; E, I, EE, EI, IE, II, input_E, input_I, silent = true)
nothing # hide
```

## Monitoring variables

`monitor!(obj, keys; sr)` registers the fields `keys` of `obj` for recording. `obj` can be
a single component or a collection such as `model.pop`. The sampling rate `sr` defaults to
`1000Hz` for a single component and to `200Hz` for a collection; the sampling period is
`max(1, round(1 / (sr * dt)))` steps. `:fire` records every spike, independently of `sr`.

```@example recordings
SNN.monitor!(E, [:ge, :gi], sr = 200Hz)
SNN.monitor!(model.pop, :v, sr = 200Hz)
SNN.monitor!(model.pop, :fire)
SNN.sim!(; model, duration = 2s)
nothing # hide
```

Buffers are allocated before the run from its duration (or `monitor_time`) and grow when
needed, with a one-time warning; `monitor_rate` (default `20Hz`, written with units) sizes the spike buffer.

A tuple `(key, indices)` restricts the recording of a vector field to some neurons, e.g.
`SNN.monitor!(E, [(:v, 1:10)])`. For `:fire` only the spikes of the listed neurons are
recorded, with their original indices.

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

Samples are taken at the global steps that are multiples of the sampling period, and `r` runs
from the first to the last sample taken.

```@example recordings
v = SNN.getvariable(E, :v)                    # 800 x 401 samples (2 s at 200 Hz, plus t = 0)
v = SNN.record(E, :v)                         # interpolated
v(1, 1.5s)
v(1:10, 1.2s:15ms:1.9s)
v, r = SNN.record(E, :v, range = true)
size(v), r
v = SNN.record(E, :v, interpolate = false)    # same as getvariable
nothing # hide
```

### Spike times and firing rates

`spiketimes(pop)` returns a `Spiketimes` object, a vector with the spike times (ms) of each
neuron. The keyword `interval` restricts it to a time window.

```@example recordings
st = SNN.spiketimes(E)
length(st), length(st[1])
st = SNN.spiketimes(E; interval = 0:1ms:1s)
nothing # hide
```

`bin_spiketimes(pop; interval)` counts the spikes in the bins defined by the range
`interval` and returns `(counts, r)`, with `counts` a neurons x bins matrix.

```@example recordings
interval = 0:10ms:2s
bins, r = SNN.bin_spiketimes(E; interval)
size(bins)
```

`firing_rate(pop; interval)` returns `(rates, r)`: the firing rate of each neuron (Hz)
sampled on `interval`, obtained by convolving the spike trains with a kernel (alpha
function by default). With `interpolate = true` (default) `rates` is an interpolation
object; `pop_average = true` averages over neurons. On a collection of populations it
returns `(rates, r, names)`, one entry per population.

```@example recordings
fr, r = SNN.firing_rate(E; interval)
fr, r = SNN.firing_rate(E; interval, interpolate = false)
fr, r, names = SNN.firing_rate(model.pop; interval)
names
```

The same quantities are available through `record`:

```@example recordings
fr = SNN.record(E, :fire; interval)
fr, r = SNN.record(E, :fire; interval, range = true)
st = SNN.record(E, :spikes)
nothing # hide
```

## Synaptic variables

Fields of connections are monitored in the same way. `:W` and `:ρ` have one value per synapse.
Plasticity variables need the name of their structure, as the keyword `variables` or a third
argument: `:STPVars` (`:u`, `:x` of `MarkramSTPParameter`) or `:LTPVars` (e.g. `:tpost` of
`iSTDPRate`). They are stored under `Symbol(variables, "_", key)`, e.g. `:STPVars_u`, and recorded
at their value at the sampling time: some rules store traces and weights lazily, so read them
outside a recording with `plasticity_value` or `trace_values` (see [Plasticity](plasticity.md)).

```@example recordings
SNN.monitor!(EE, [:ρ], sr = 10Hz)
SNN.monitor!(IE, [:W], sr = 10Hz)
SNN.monitor!(IE, [:tpost]; sr = 10Hz, variables = :LTPVars)
SNN.monitor!(EE, [:x, :u], :STPVars; sr = 10Hz)
SNN.train!(; model, duration = 2s)
nothing # hide
```

The first run of this page used `sim!`, which never changes weights or STP variables; this one
uses `train!`.

!!! warning
    Recording per-synapse variables can use a lot of memory in large networks; use a low
    sampling rate.

### Weight matrices

Connectivity is stored in sparse form; `matrix(c)` returns the current weights as an
`N_post x N_pre` sparse matrix, and `matrix(c, sym)` any other per-synapse field.

```@example recordings
W = SNN.matrix(EE)            # current weights
ρ = SNN.matrix(EE, :ρ)        # current STP efficacy
nothing # hide
```

`presynaptic(c, i)` returns the presynaptic neurons of postsynaptic neuron `i`, and
`postsynaptic(c, j)` the postsynaptic neurons of presynaptic neuron `j`; both also accept a
vector of neurons.

```@example recordings
neuron = 1
Is = SNN.postsynaptic(EE, neuron)
mean(W[Is, neuron])           # mean weight of the outgoing connections of neuron 1
Js = SNN.presynaptic(EE, neuron)
mean(W[neuron, Js])           # mean weight of the incoming connections of neuron 1
Js_many = SNN.presynaptic(EE, 1:10)
nothing # hide
```

A recorded per-synapse variable is a synapses x samples array; `record` interpolates it in
time, and `matrix_record(c, sym, t)` rebuilds the sparse matrix at time `t` (a vector of
times gives a 3-dimensional array). The times are those of the `train!` call above, from
2 s to 4 s.

```@example recordings
W_IE, r = SNN.record(IE, :W, range = true)
W_IE(axes(W_IE, 1), 3.5s)                      # weights of all IE synapses at t = 3.5 s
W_mat = SNN.matrix_record(IE, :W, 3.5s)         # 800 x 200 sparse matrix
W_mat2 = SNN.matrix(IE, W_IE, 3.5s)             # same, from the interpolated record
all(W_mat .== W_mat2)
W_T = SNN.matrix_record(IE, :W, 3s:100ms:3.5s)  # 800 x 200 x 6
nothing # hide
```

### Plasticity variables

Plasticity variables are read with their compound key:

```@example recordings
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

```@example recordings
Logging.disable_logging(Logging.Debug) # hide
nothing # hide
```

## Recording API

```@autodocs
Modules = [SNNModels]
Pages   = ["utils/record.jl"]
```
