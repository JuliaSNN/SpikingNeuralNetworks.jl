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
calls; `sim!` leaves the weights unchanged. The only state updated under `sim!` is the rate
estimate of `AggregateScaling` (its target weight and the weights change only under `train!`).

All objects are built with `MetaPlasticity(param, synapses)` or with their own constructor:

| Parameter | Object | Acts on |
|:----------|:-------|:--------|
| `MultiplicativeNorm`, `AdditiveNorm` | `SynapseNormalization` | vector of sparse synapses with the same postsynaptic population |
| `AggregateScalingParameter` | `AggregateScaling` | vector of sparse synapses with the same postsynaptic population |
| `ActivityDependentTurnover`, `RandomTurnover` | `Turnover` | one `SpikingSynapse` |

Periodic operations run when the step counter `get_step(T)` is a multiple of
`max(1, round(Int, τ / dt))` (in SNNModels 1.8.4 `τ < dt/2` divided by zero). The code gives no literature reference for these rules.

## SynapseNormalization

Keeps the summed input weight of each postsynaptic neuron ``i`` close to its value at
construction, ``W^0_i = \sum_{s \to i} W_s`` (sum over the synapses onto ``i`` of all
normalized connections). Every `τ`, with ``W^1_i`` the current sum:

| Parameter | Factor or offset | Weight update |
|:----------|:-----------------|:--------------|
| `MultiplicativeNorm(τ)` | ``\mu_i = W^0_i / W^1_i`` | ``W_s \leftarrow W_s\,\mu_i`` (restores ``W^0_i`` exactly) |
| `AdditiveNorm(τ)` | ``\mu_i = (W^0_i - W^1_i) / n_i`` (``n_i`` inputs) | ``W_s \leftarrow W_s + \mu_i`` (restores ``W^0_i`` exactly) |

(In SNNModels 1.8.4 the additive offset was ``(W^0_i - W^1_i)/W^1_i``, so the sum became
``W^1_i + n_i (W^0_i - W^1_i)/W^1_i`` instead of ``W^0_i``.) `param` is a required keyword of the
keyword constructors.

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `τ` | required | ms | interval between normalizations |
| `operator` | `*` / `+` | | kept for compatibility, not used |

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

Homeostatic scaling: a rate estimate ``y_i`` of each postsynaptic neuron drives a target
summed weight ``W^T_i``, and the incoming weights are rescaled towards it:

```math
\tau_a \frac{dy_i}{dt} = -y_i + \sum_f \delta(t - t_i^f), \qquad
\tau_e \frac{dW^T_i}{dt} = \left(1 - \frac{W^T_i}{W_{max}}\right)\left(1 - \frac{y_i}{Y_i}\right)
```

``y_i`` (1/ms) jumps by ``1/\tau_a`` at each spike of ``i`` and is updated in `forward!` (under
`sim!` and `train!`); ``W^T_i`` is updated with Euler steps of `dt` under `train!` only. Every `τ`,
under `train!`, with ``W^t_i`` the current summed input weight and ``n_i`` the number of inputs:

```math
\mu_i = \frac{\max(W^T_i - n_i W_{min}, 0)}{W^t_i}, \qquad W_s \leftarrow W_s\,\mu_i + W_{min}
```

so that the summed weight equals ``W^T_i``. ``W^T_i`` starts from the summed input weight at
construction.

!!! note "Changed after SNNModels 1.8.4"
    In SNNModels 1.8.4 ``y`` and ``W^T`` were updated per step without `dt` (time constants
    ``\tau_a dt``, ``\tau_e dt``), ``y`` counted spikes while ``Y`` is a rate (the fixed point was
    at the rate ``Y/(\tau_a\,dt)``, i.e. ``Y`` / 12.5 ms at the defaults), ``W^T`` also evolved
    under `sim!`, the rescaling gave a sum of ``W^T_i + (n_i - 1) W_{min}``, the positional
    constructor defaulted to `Wmin = 0.05`, and `N` was always 0.

`AggregateScalingParameter` has two constructors:

| Field | Keyword constructor | Positional `AggregateScalingParameter(N, rate = 10Hz; ...)` | Meaning |
|:------|:--------------------|:------------------------------------------------------------|:--------|
| `τ` | `10ms` | `10ms` | interval between rescalings (ms) |
| `τa` | required | `100ms` | time constant of the rate estimate (ms) |
| `τe` | required | `100ms` | time constant of the target weight (ms) |
| `Y` | required (vector) | `fill(rate, N)` | target rate per neuron (1/ms, write it with `Hz`) |
| `Wmin` | `0.5pF` | `0.5pF` | offset added after rescaling (units of the weights) |
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
postsynaptic targets and given a new weight from ``\mathcal{N}(\mu, \sqrt{\mu})`` truncated to
positive values.

With `ActivityDependentTurnover`, at every `train!` step the traces

```math
a^{pre}_j \leftarrow a^{pre}_j + \frac{-a^{pre}_j\,dt + \delta_j}{\tau_{pre}}, \qquad
a^{post}_i \leftarrow a^{post}_i + \frac{-a^{post}_i\,dt + \delta_i}{\tau_{post}}
```

are updated and every `τ` the co-activity ``p_s = a^{pre}_{j(s)} a^{post}_{i(s)}`` of every
synapse is computed. The synapses with ``p_s`` at or below the `fraction`-quantile are rewired
by `synaptic_turnover!`: for each presynaptic neuron, as many new targets as rewired synapses
are drawn without replacement among the neurons it does not contact yet (uniformly), the
postsynaptic index and the weight of the synapse are replaced, its efficacy `ρ` is reset to 1
and the sparse storage is rebuilt with the per-synapse arrays reordered accordingly. If a
presynaptic neuron has fewer free targets than selected synapses, only as many are rewired.
With `RandomTurnover`, every `τ` each synapse is rewired with probability `threshold`.

| Field | `ActivityDependentTurnover` | `RandomTurnover` | Units | Meaning |
|:------|:----------------------------|:-----------------|:------|:--------|
| `rate` | `-1` | `-1` | 1/ms | turnover rate, used only for the default of `τ` |
| `τ` | `1 / rate` | `1 / rate` | ms | interval between turnover events (pass `rate` or `τ`) |
| `fraction` | `0.1` | | | quantile of co-activity below which synapses are rewired |
| `threshold` | | `0.1` | | probability that a synapse is rewired at each event |
| `τpre`, `τpost` | `250ms` | | ms | time constants of the activity traces |
| `μ` | `3.0` | `3.0` | weight units | mean of new weights |

(In SNNModels 1.8.4 `RandomTurnover` had no `plasticity!(c, param, dt, T)` method and `train!`
raised a `MethodError`; the default `p_new` of `synaptic_turnover!` had the wrong arity; new
weights could be negative; `ρ` was not reordered after a rebuild.)

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
