# Stimuli

```@meta
CurrentModule = SNNModels
```

External inputs, built with `Stimulus(param, post, sym, [comp]; kwargs...)`; the type of `param`
selects the model. Usage: [Stimuli](../stimuli.md).

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

One independent Poisson process per targeted neuron:

```math
g_n \leftarrow g_n + \mu\, k_n, \qquad k_n \sim \mathrm{Poisson}\big(\nu(t)\,\Delta t\big)
```


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

`N` Poisson neurons connected through a sparse matrix built from `conn`; layer neuron ``j`` fires
with probability ``\nu_j \Delta t`` and

```math
g_i \leftarrow g_i + W_{ij} \quad \text{for every target } i \text{ of } j .
```

| Parameter | Field | Default | Units | Meaning |
|:--|:--|:--|:--|:--|
| `PoissonLayer` | `rate` | required | rate (`Hz`) | rate of every layer neuron |
| | `N` | `1` | | number of layer neurons |
| | `active` | `[true]` | | `false` silences the layer |
| `PoissonLayerHet` | `rates` | required | rate (`Hz`) | rate of each layer neuron (length `N`) |
| | `N` | `1` | | number of layer neurons |
| | `active` | `[true]` | | `false` silences the layer |

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

```@autodocs
Modules = [SNNModels]
Pages   = ["stimuli/poisson_layer.jl"]
```

## Spike-time stimulus

Replays a sorted spike list (`neurons[k]` fires at `spiketimes[k]`) through a sparse matrix built
from `conn`; `SpikeTimeParameter` sorts the list, `SpikeTimeStimulusIdentity` connects input `j` to
neuron `j`.

| Field of `SpikeTimeStimulusParameter` | Default | Units | Meaning |
|:--|:--|:--|:--|
| `spiketimes` | `[]` | ms | spike times, sorted |
| `neurons` | `[]` | | input neuron of each spike |


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

Sets the external current (`:I` by default) of the targeted neurons:

```math
I_i \leftarrow (1-\alpha_i)\,\big(I^{base}_i + \xi_i\big) + \alpha_i\, I_i,
\qquad \xi_i \sim \texttt{I\_dist}
```

``\alpha = 0`` gives white noise per step; ``0 < \alpha < 1`` a filtered process.

| Field | Default (`CurrentNoise(N; ...)`) | Units | Meaning |
|:--|:--|:--|:--|
| `I_base` | `0`, filled to length `N` | pA | baseline current, indexed by neuron id |
| `I_dist` | `Normal(0.0, 0.0)` | pA | noise distribution |
| `α` | `0.0`, filled to length `N` | | filter coefficient in ``[0, 1]`` |

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

```@autodocs
Modules = [SNNModels]
Pages   = ["stimuli/current.jl"]
```

## Balanced stimulus

Poisson inhibition at rate ``k_{IE} r_0`` and Poisson excitation fluctuating around ``r_0``
(see [`BalancedParameter`](@ref)).

| Field | Default | Units | Meaning |
|:--|:--|:--|:--|
| `kIE` | `1.0` | | inhibitory rate / `r0` |
| `β` | `0.0` | | amplitude of the rate fluctuations |
| `τ` | `50ms` | ms | correlation time of the fluctuations |
| `r0` | `1kHz` | rate | baseline rate |
| `w` | `1.0` | units of the target | increment per input spike |
| `wIE` | `1.0` | | extra factor on the inhibitory increment |
| `same_input` | `false` | | one rate process shared by all neurons |

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100)
stim = SNN.Stimulus(SNN.BalancedParameter(r0 = 2kHz, kIE = 0.5), E, :ge)
model = SNN.compose(; E, stim)
SNN.monitor!(E, [:fire])
SNN.sim!(; model, duration = 200ms)
```

```@autodocs
Modules = [SNNModels]
Pages   = ["stimuli/balanced.jl"]
```

## Stimulus groups

A `StimulusGroup` bundles stimuli; `MultiCompartmentStimulusGroup` makes one per compartment.

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

```@autodocs
Modules = [SNNModels]
Pages   = ["stimuli/stimuli.jl", "stimuli/empty.jl"]
```
