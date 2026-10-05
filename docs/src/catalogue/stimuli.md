# Stimuli

```@meta
CurrentModule = SNNModels
```

Stimuli are the external inputs of a network. A stimulus is attached to one target
population and, at every time step, writes into one of its variables: the input buffer of a
receptor or conductance (Poisson and spike-time stimuli), or the external current `I`
(current stimuli). Every stimulus type is built from a parameter object with the generic
constructor

```
Stimulus(param, post, sym, [comp]; kwargs...)
```

whose method is chosen by the type of `param`. In the simulation loop the stimuli are
called first, after the clock has advanced by `dt` and before the populations are
integrated (see [Stimuli](../stimuli.md) for the user guide and `sim!` for the loop).

The variable `sym` is resolved by the `synaptic_target` method of the target population:
for generalized integrate-and-fire neurons `:ge`/`:he` select the glutamatergic receptor
buffer and `:gi`/`:hi` the GABAergic one (or the receptor group name of a
`MultiReceptorSynapse`); for dendritic neurons the compartment is given as `comp`
(e.g. `:d1`). Rates use the library units: `Hz` = 1e-3 per ms, so `10Hz` is the rate
of 10 spikes per second.

| Stimulus | Parameter types | Writes | Presynaptic layer |
|:--|:--|:--|:--|
| [`PoissonStimulus`](@ref) | `PoissonFixed`, `PoissonInterval`, `PoissonVariable` | adds `μ` per input spike to `g` | none (one independent train per target) |
| [`PoissonStimulusLayer`](@ref) | `PoissonLayer`, `PoissonLayerHet` | adds `W[i,j]` per spike of layer neuron `j` | `N` Poisson neurons, sparse weights |
| [`SpikeTimeStimulus`](@ref) | `SpikeTimeStimulusParameter` | adds `W[i,j]` per listed spike | `N` virtual neurons, sparse weights |
| [`CurrentStimulus`](@ref) | `CurrentNoise` | sets the current `I` | none |
| [`BalancedStimulus`](@ref) | `BalancedParameter` | adds Poisson E and I input | none |
| [`StimulusGroup`](@ref) | (shared Poisson parameter) | its elements | |
| `EmptyStimulus` | `EmptyParam` | nothing (placeholder) | |

## Poisson stimulus

`PoissonStimulus` delivers to each targeted neuron ``n`` its own Poisson process of rate
``\nu(t)``. There is no presynaptic population and no weight matrix.

```math
g_n \leftarrow g_n + \mu\, k_n, \qquad k_n \sim \mathrm{Poisson}\big(\nu(t)\,\Delta t\big)
```

The rate ``\nu(t)`` is returned by `get_poisson_rate(param, time)`:

| Parameter | Rate ``\nu(t)`` |
|:--|:--|
| `PoissonFixed(; rate, μ)` | `rate` |
| `PoissonInterval(; rate, intervals, μ)` | `rate` if `int[1] < t < int[end]` for some `int` in `intervals`, else 0 |
| `PoissonVariable(; variables, rate::Function, μ)` | `rate(t, variables)` |

| Field | Default | Units | Meaning |
|:--|:--|:--|:--|
| `rate` | `0` (`PoissonFixed`), required otherwise | rate (`Hz`) | rate per target neuron |
| `intervals` | `[]` | ms | `PoissonInterval` only: list of `[start, end]` |
| `variables` | required | | `PoissonVariable` only: arguments of the rate function |
| `μ` | `1.0` | units of the target variable | increment per input spike |
| `active` | `[true]` | | on/off switch, see `set_active!` |

Constructor keywords: `neurons = :ALL` (all neurons), a vector of indices, or `:p_post`
together with `p_post` (fraction of randomly chosen target neurons); `comp` for
multicompartment targets; `name`.

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100)
fixed = SNN.Stimulus(SNN.PoissonFixed(rate = 2kHz, μ = 1.0), E, :ge)
pulses = SNN.Stimulus(SNN.PoissonInterval(rate = 1kHz, intervals = [[100ms, 200ms]]), E, :ge; neurons = 1:20)
osc(t, v) = v[:r0] * (1 + sin(2π * t / v[:period]))
rhythm = SNN.Stimulus(SNN.PoissonVariable(variables = Dict{Symbol,Any}(:r0 => 1kHz, :period => 100ms), rate = osc), E, :ge)
model = SNN.compose(; E, fixed, pulses, rhythm)
SNN.monitor!(E, :fire)
SNN.sim!(; model, duration = 300ms)
```

```@autodocs
Modules = [SNNModels]
Pages   = ["stimuli/poisson.jl"]
```

## Poisson layer

`PoissonStimulusLayer` is a layer of `N` independent Poisson neurons connected to the target
population through a sparse matrix ``W`` (`post.N x N`) generated from `conn` with
`sparse_matrix` (rules `:Fixed`, `:FixedIn`, `:FixedOut`, `:Bernoulli`, `:PowerLaw`; weight
distribution `μ`, `σ`, `dist`). At each step layer neuron ``j`` fires with probability
``\nu_j \Delta t`` (at most one spike per step), and then

```math
g_i \leftarrow g_i + W_{ij} \quad \text{for every target } i \text{ of } j .
```

| Parameter | Field | Default | Units | Meaning |
|:--|:--|:--|:--|:--|
| `PoissonLayer` | `rate` | required | rate (`Hz`) | rate of every layer neuron |
| | `N` | `1` | | number of layer neurons |
| | `active` | `[true]` | | not read by `stimulate!` in 1.8.4 |
| `PoissonLayerHet` | `rates` | required | rate (`Hz`) | rate of each layer neuron (length `N`) |
| | `N` | `1` | | number of layer neurons |
| | `active` | `[true]` | | not read by `stimulate!` in 1.8.4 |

Unlike `PoissonStimulus`, a layer has spikes of its own (`stim.fire`), which can be
recorded with `monitor!(stim, :fire)` and read with `spiketimes(stim)`.

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100)
layer = SNN.Stimulus(SNN.PoissonLayer(rate = 10Hz, N = 1000), E, :ge; conn = (p = 0.05, μ = 1.0))
het = SNN.Stimulus(SNN.PoissonLayerHet(N = 50, rates = collect(range(1Hz, 50Hz, length = 50))), E, :gi;
                   conn = (p = 0.2, μ = 0.5))
SNN.monitor!(layer, :fire)
model = SNN.compose(; E, layer, het)
SNN.sim!(; model, duration = 200ms)
length(SNN.spiketimes(layer))   # 1000 input neurons
```

!!! warning
    `set_active!(stim, false)` has no effect on a `PoissonStimulusLayer` in SNNModels
    1.8.4: the `active` flag of `PoissonLayer`/`PoissonLayerHet` is not checked by
    `stimulate!`.

```@autodocs
Modules = [SNNModels]
Pages   = ["stimuli/poisson_layer.jl"]
```

## Spike-time stimulus

`SpikeTimeStimulus` replays a list of spikes (`neurons[k]` fires at `spiketimes[k]`) from
`N` virtual input neurons, delivered through a sparse weight matrix built from `conn`
(NamedTuple or matrix). The spikes are consumed in order, so they must be sorted by time:
build the parameter with `SpikeTimeParameter`, which sorts them.
`SpikeTimeStimulusIdentity` connects input neuron `j` to target neuron `j` with weight 1.

A spike with time ``t_k`` is delivered in the first step whose time ``t`` satisfies
``t_k \le t``, adding ``W_{ij}`` to the target ``i`` of input neuron ``j``.

| Field of `SpikeTimeStimulusParameter` | Default | Units | Meaning |
|:--|:--|:--|:--|
| `spiketimes` | `[]` | ms | spike times, sorted |
| `neurons` | `[]` | | input neuron of each spike |

| Constructor keyword | Default | Meaning |
|:--|:--|:--|
| `conn` | required (except for `SpikeTimeStimulusIdentity`) | connectivity NamedTuple or `post.N x N` matrix |
| `N` | `max_neuron(param)` | number of input neurons |
| `name` | `"SpikeTime"` | |

Related functions: `shift_spikes!(stim, delay)` shifts the spike list and rewinds the
stimulus; `update_spikes!(stim, spikes, start_time)` replaces it; `next_neuron(stim)`.

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 10)
param = SNN.SpikeTimeParameter([30ms, 10ms, 20ms], [1, 2, 1])   # sorted by time
stim = SNN.SpikeTimeStimulus(E, :ge; param, conn = (p = 0.5, μ = 2.0))
one_to_one = SNN.SpikeTimeStimulusIdentity(E, :gi; param = SNN.SpikeTimeParameter([5ms], [3]))
SNN.monitor!(stim, :fire)
model = SNN.compose(; E, stim, one_to_one)
SNN.sim!(; model, duration = 50ms)
SNN.update_spikes!(stim, SNN.SpikeTimeParameter([5ms, 15ms], [2, 2]), SNN.get_time(model.time))
SNN.sim!(; model, duration = 50ms)
```

```@autodocs
Modules = [SNNModels]
Pages   = ["stimuli/timed.jl"]
```

## Current stimulus

`CurrentStimulus` writes into the external-current field (`sym = :I` by default) of the
targeted neurons. The loaded parameter type is `CurrentNoise`:

```math
I_i \leftarrow (1-\alpha_i)\,\big(I^{base}_i + \xi_i\big) + \alpha_i\, I_i,
\qquad \xi_i \sim \texttt{I\_dist}
```

With ``\alpha_i = 0`` the current is redrawn every step (white noise around `I_base`, with
per-step standard deviation equal to that of `I_dist`, not scaled by `dt`); with
``0 < \alpha_i < 1`` it is an exponentially filtered (AR(1)) process with correlation time
``\Delta t/(1-\alpha_i)``.

| Field | Default (`CurrentNoise(N; ...)`) | Units | Meaning |
|:--|:--|:--|:--|
| `I_base` | `0`, filled to length `N` | pA | baseline current, indexed by neuron id |
| `I_dist` | `Normal(0.0, 0.0)` | pA | noise distribution |
| `α` | `0.0`, filled to length `N` | | filter coefficient in ``[0, 1]`` |

The stimulus sets (does not add to) `I`, so two current stimuli on the same neurons
overwrite each other; neurons outside `neurons` are not touched. Since SNNModels 1.8.4 a
subset of neurons (`neurons = [...]`) is handled correctly (the random numbers are indexed
by position in `neurons`, `I_base` and `α` by neuron id).

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 10, param = SNN.IFParameter(C = 281pF, gl = 40nS))
noise = SNN.CurrentNoise(E; I_base = 600pA, I_dist = SNN.SNNModels.Normal(0.0, 100.0), α = 0.9)
stim = SNN.CurrentStimulus(E; neurons = [1, 2, 3], param = noise)
model = SNN.compose(; E, stim)
SNN.monitor!(E, [:v, :fire, :I])
SNN.sim!(; model, duration = 200ms)
SNN.getvariable(E, :I)[4, end]   # 0: neuron 4 is not stimulated
```

A time-dependent current parameter (`ramping_current`, `CurrentVariableParameter`) is
present in `src/stimuli/variable_inputs.jl` but that file is not loaded by SNNModels 1.8.4.

```@autodocs
Modules = [SNNModels]
Pages   = ["stimuli/current.jl"]
```

## Balanced stimulus

`BalancedStimulus` drives every neuron of the target with Poisson inhibitory input of rate
``k_{IE} r_0`` and Poisson excitatory input whose rate fluctuates around ``r_0`` with
correlation time ``\tau`` and amplitude ``\beta``. The equations and the parameters are
listed in [`BalancedParameter`](@ref).

| Field | Default | Units | Meaning |
|:--|:--|:--|:--|
| `kIE` | `1.0` | | inhibitory rate / `r0` |
| `β` | `0.0` | | amplitude of the rate fluctuations |
| `τ` | `50ms` | ms | correlation time of the fluctuations |
| `r0` | `1kHz` | rate | baseline rate |
| `w` | `1.0` | units of the target | increment per input spike |
| `wIE` | `1.0` | | extra factor on the inhibitory increment |
| `same_input` | `false` | | one rate process shared by all neurons |

!!! warning "Not usable in SNNModels 1.8.4"
    The default configuration (`same_input = false`) throws `UndefVarError: randcache not
    defined` at the first step, and `same_input = true` delivers all excitatory input to
    neuron 1. The generic `Stimulus(param::BalancedParameter, post, sym)` uses the same
    target for excitation and inhibition. No runnable example is given until these are
    fixed.

```@autodocs
Modules = [SNNModels]
Pages   = ["stimuli/balanced.jl"]
```

## Stimulus groups

A `StimulusGroup` bundles several stimuli. `sim!`/`train!` unpack it, and `set_variable!`,
`set_intervals!`, `set_active!` and `record` are applied to all elements.
`MultiCompartmentStimulusGroup` creates one Poisson stimulus per compartment of a
dendritic neuron, all sharing the same parameter object.

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.Tripod(N = 10)
param = SNN.PoissonInterval(rate = 100Hz, intervals = [[0ms, 50ms]])
group = SNN.MultiCompartmentStimulusGroup(param, E, :glu, [:d1, :d2])
model = SNN.compose(; E, group)
SNN.sim!(; model, duration = 100ms)
SNN.set_intervals!(group, [[100ms, 150ms]])
SNN.sim!(; model, duration = 100ms)
```

```@autodocs
Modules = [SNNModels]
Pages   = ["stimuli/stimulus_group.jl"]
```

## Common interface

`stimulate!`, `neurons`, `set_variable!`, `set_intervals!`, `set_active!`, and the empty
placeholder stimulus.

```@autodocs
Modules = [SNNModels]
Pages   = ["stimuli/stimuli.jl", "stimuli/empty.jl"]
```
