# Metaplasticity

```@meta
CurrentModule = SNNModels
```

Metaplasticity objects act on the weights of other connections. They are subtypes of
`AbstractConnection` (via `AbstractMetaPlasticity`) and are added to the model like any
connection, after the synapses they act on:

<!-- norun -->
```julia
model = SNN.compose(; E, EE, norm = SNN.SynapseNormalization([EE]; param = SNN.MultiplicativeNorm(τ = 20ms)))
```

They transmit nothing. Their weight updates are done in `plasticity!`, which only `train!`
calls; `sim!` leaves the weights unchanged. The only exception is the activity trace and target
weight of `AggregateScaling`, which are updated in `forward!` and therefore also evolve under
`sim!` (the weights themselves are only rescaled under `train!`).

All objects are built with `MetaPlasticity(param, synapses)` or with their own constructor:

| Parameter | Object | Acts on |
|:----------|:-------|:--------|
| `MultiplicativeNorm`, `AdditiveNorm` | `SynapseNormalization` | vector of sparse synapses with the same postsynaptic population |
| `AggregateScalingParameter` | `AggregateScaling` | vector of sparse synapses with the same postsynaptic population |
| `ActivityDependentTurnover`, `RandomTurnover` | `Turnover` | one `SpikingSynapse` |

Periodic operations run when the step counter `get_step(T)` is a multiple of
`round(Int, τ / dt)`. The code gives no literature reference for these rules.

## SynapseNormalization

Keeps the summed input weight of each postsynaptic neuron ``i`` close to its value at
construction, ``W^0_i = \sum_{s \to i} W_s`` (sum over the synapses onto ``i`` of all
normalized connections). Every `τ`, with ``W^1_i`` the current sum:

| Parameter | Factor or offset | Weight update |
|:----------|:-----------------|:--------------|
| `MultiplicativeNorm(τ)` | ``\mu_i = W^0_i / W^1_i`` | ``W_s \leftarrow W_s\,\mu_i`` (restores ``W^0_i`` exactly) |
| `AdditiveNorm(τ)` | ``\mu_i = (W^0_i - W^1_i) / W^1_i`` | ``W_s \leftarrow W_s + \mu_i`` |

The additive offset is not divided by the number of inputs ``n_i``: after the update the sum is
``W^1_i + n_i (W^0_i - W^1_i)/W^1_i``, which equals ``W^0_i`` only when ``n_i = W^1_i``.

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `τ` | required | ms | interval between normalizations |
| `operator` | `*` / `+` | | combination operator (do not change) |

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100)
P = SNN.Poisson(N = 200, param = SNN.PoissonParameter(10Hz))
PE = SNN.SpikingSynapse(P, E, :ge; conn = (p = 0.2, μ = 2.0), LTPParam = SNN.STDPGerstner())
norm = SNN.SynapseNormalization([PE]; param = SNN.MultiplicativeNorm(τ = 20ms))
model = SNN.compose(; E, P, PE, norm)
SNN.train!(model; duration = 200ms)
```

## AggregateScaling

Homeostatic scaling: a spike-count trace ``y_i`` of each postsynaptic neuron drives a target
summed weight ``W^T_i``, and the incoming weights are rescaled towards it. At every step
(`forward!`, under both `sim!` and `train!`; the constants are used per step, not per ms):

```math
y_i \leftarrow y_i - \frac{y_i}{\tau_a} + \delta_i, \qquad
W^T_i \leftarrow W^T_i + \frac{1}{\tau_e}\left(1 - \frac{W^T_i}{W_{max}}\right)\left(1 - \frac{y_i}{Y_i}\right)
```

with ``\delta_i = 1`` if ``i`` fired. Every `τ`, under `train!` only, with ``W^t_i`` the current
summed input weight:

```math
\mu_i = \frac{W^T_i - W_{min}}{W^t_i}, \qquad W_s \leftarrow W_s\,\mu_i + W_{min}
```

``W^T_i`` starts from the summed input weight at construction. Because ``\tau_a`` and
``\tau_e`` are divided per step, their effective time constants scale with `dt`; ``y`` is a
spike count while ``Y`` is specified as a rate.

`AggregateScalingParameter` has two constructors:

| Field | Keyword constructor | Positional `AggregateScalingParameter(N, rate = 10Hz; ...)` | Meaning |
|:------|:--------------------|:------------------------------------------------------------|:--------|
| `τ` | `10ms` | `10ms` | interval between rescalings (ms) |
| `τa` | required | `100ms` | decay of the activity trace (per step) |
| `τe` | required | `100ms` | time constant of the target weight (per step) |
| `Y` | required (vector) | `fill(rate, N)` | target activity per neuron |
| `Wmin` | `0.5pF` | `0.05` | offset added after rescaling (units of the weights) |
| `Wmax` | `250pF` | `250pF` | soft upper bound of ``W^T`` |

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100)
P = SNN.Poisson(N = 200, param = SNN.PoissonParameter(10Hz))
PE = SNN.SpikingSynapse(P, E, :ge; conn = (p = 0.2, μ = 2.0))
AS = SNN.AggregateScaling(E, [PE]; param = SNN.AggregateScalingParameter(E.N, 5Hz))
model = SNN.compose(; E, P, PE, AS)
SNN.train!(model; duration = 200ms)
```

## Turnover

Structural plasticity of one `SpikingSynapse`: periodically, a set of synapses is moved to new
postsynaptic targets and given a new weight ``\mathcal{N}(\mu, \sqrt{\mu})``.

With `ActivityDependentTurnover`, at every `train!` step the traces

```math
a^{pre}_j \leftarrow a^{pre}_j + \frac{-a^{pre}_j\,dt + \delta_j}{\tau_{pre}}, \qquad
a^{post}_i \leftarrow a^{post}_i + \frac{-a^{post}_i\,dt + \delta_i}{\tau_{post}}
```

are updated and every `τ` the co-activity ``p_s = a^{pre}_{j(s)} a^{post}_{i(s)}`` of every
synapse is computed. The synapses with ``p_s`` at or below the `fraction`-quantile are rewired
by `synaptic_turnover!`: for each presynaptic neuron, as many new targets as rewired synapses
are drawn without replacement among the neurons it does not contact yet (uniformly), the
postsynaptic index and the weight of the synapse are replaced and the sparse storage is
rebuilt. Per-synapse arrays other than `I`, `J`, `W`, `index` (STP efficacy `ρ`, plasticity
traces) are not reordered.

| Field | `ActivityDependentTurnover` | `RandomTurnover` | Units | Meaning |
|:------|:----------------------------|:-----------------|:------|:--------|
| `rate` | `-1` | `-1` | 1/ms | turnover rate, used only for the default of `τ` |
| `τ` | `1 / rate` | `1 / rate` | ms | interval between turnover events (pass `rate` or `τ`) |
| `fraction` | `0.1` | | | quantile of co-activity below which synapses are rewired |
| `threshold` | | `0.1` | | unused |
| `τpre`, `τpost` | `250ms` | | ms | time constants of the activity traces |
| `μ` | `3.0` | `3.0` | weight units | mean of new weights |

`RandomTurnover` has no `plasticity!(c, param, dt, T)` method in SNNModels 1.8.4, so `train!`
raises a `MethodError` when a `Turnover` with this parameter is in the model.

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 100)
P = SNN.Poisson(N = 200, param = SNN.PoissonParameter(20Hz))
PE = SNN.SpikingSynapse(P, E, :ge; conn = (p = 0.2, μ = 2.0))
TO = SNN.MetaPlasticity(SNN.ActivityDependentTurnover(τ = 50.0f0ms), PE)
model = SNN.compose(; E, P, PE, TO)
SNN.train!(model; duration = 200ms)
```

## API

```@autodocs
Modules = [SNNModels]
Pages   = ["connections/metaplasticity/normalization.jl", "connections/metaplasticity/aggregate_scaling.jl", "connections/metaplasticity/turnover.jl"]
```
