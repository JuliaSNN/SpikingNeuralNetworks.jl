# Metaplasticity

```@meta
CurrentModule = SNNModels
```

Components that act on the weights of other connections, added to the model like connections;
their weight updates run only under `train!`.

All objects are built with `MetaPlasticity(param, synapses)` or with their own constructor:

| Parameter | Object | Acts on |
|:----------|:-------|:--------|
| `MultiplicativeNorm`, `AdditiveNorm` | `SynapseNormalization` | vector of sparse synapses with the same postsynaptic population |
| `AggregateScalingParameter` | `AggregateScaling` | vector of sparse synapses with the same postsynaptic population |
| `ActivityDependentTurnover`, `RandomTurnover` | `Turnover` | one `SpikingSynapse` |

| Model | Reference | Integration |
|:------|:----------|:------------|
| `SynapseNormalization` (`MultiplicativeNorm`, `AdditiveNorm`) | Turrigiano et al. (1998); Turrigiano (1999); Turrigiano (2008) | rescaling every `max(1, round(τ / dt))` steps |
| `AggregateScaling` | Turrigiano et al. (1998); Turrigiano (2008) | forward Euler for ``y`` and ``W^T``, rescaling every `max(1, round(τ / dt))` steps |
| `Turnover` | none | forward Euler traces (`ActivityDependentTurnover`), rewiring every `max(1, round(τ / dt))` steps |

References for the scaling rules (homeostatic synaptic scaling: all excitatory synapses of a neuron are scaled by a common factor so that the total input follows
the activity of the neuron):

- Turrigiano, Leslie, Desai, Rutherford and Nelson (1998), Activity-dependent scaling of quantal amplitude in neocortical neurons, Nature 391:892-896,
  [10.1038/36103](https://doi.org/10.1038/36103).
- Turrigiano (1999), Homeostatic plasticity in neuronal networks: the more things change, the more they stay the same, Trends Neurosci. 22:221-227,
  [10.1016/S0166-2236(98)01341-1](https://doi.org/10.1016/S0166-2236(98)01341-1).
- Turrigiano (2008), The self-tuning neuron: synaptic scaling of excitatory synapses, Cell 135:422-435,
  [10.1016/j.cell.2008.10.008](https://doi.org/10.1016/j.cell.2008.10.008).

## SynapseNormalization

Every `τ`, restores the summed input weight ``W^0_i`` of each neuron at construction (``W^1_i``: current sum):

| Parameter | Factor or offset | Weight update |
|:----------|:-----------------|:--------------|
| `MultiplicativeNorm(τ)` | ``\mu_i = W^0_i / W^1_i`` | ``W_s \leftarrow W_s\,\mu_i`` (restores ``W^0_i`` exactly) |
| `AdditiveNorm(τ)` | ``\mu_i = (W^0_i - W^1_i) / n_i`` (``n_i`` inputs) | ``W_s \leftarrow W_s + \mu_i`` (restores ``W^0_i`` exactly) |

| Field | Default | Units | Meaning |
|:------|:--------|:------|:--------|
| `τ` | required | ms | interval between normalizations |

**Integration:** every `max(1, round(τ / dt))` steps under `train!`, after the `plasticity!` of the connections listed before it, on the current weights. **Reference:** Turrigiano et al. (1998), Turrigiano (2008).

!!! note "Implementation note"
    The normalisation does not clamp to the `Wmin`, `Wmax` of the plasticity rules: a weight can leave the bounds until its rule touches it again.

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

A rate estimate ``y_i`` drives a target summed weight ``W^T_i``; every `τ` the inputs are rescaled to it:

```math
\tau_a \frac{dy_i}{dt} = -y_i + \sum_f \delta(t - t_i^f), \qquad
\tau_e \frac{dW^T_i}{dt} = \left(1 - \frac{W^T_i}{W_{max}}\right)\left(1 - \frac{y_i}{Y_i}\right)
```

with ``W^t_i`` the current summed weight and ``n_i`` the number of inputs:

```math
\mu_i = \frac{\max(W^T_i - n_i W_{min}, 0)}{W^t_i}, \qquad W_s \leftarrow W_s\,\mu_i + W_{min}
```


| Field | Keyword constructor | Positional `AggregateScalingParameter(N, rate = 10Hz; ...)` | Meaning |
|:------|:--------------------|:------------------------------------------------------------|:--------|
| `τ` | `10ms` | `10ms` | interval between rescalings (ms) |
| `τa` | required | `100ms` | time constant of the rate estimate (ms) |
| `τe` | required | `100ms` | time constant of the target weight (ms) |
| `Y` | required (vector) | `fill(rate, N)` | target rate per neuron (1/ms, write it with `Hz`) |
| `Wmin` | `0.5pF` | `0.5pF` | offset added after rescaling (units of the weights) |
| `Wmax` | `250pF` | `250pF` | soft upper bound of ``W^T`` |

**Integration:** forward Euler: ``y`` in `forward!` (every step of `sim!` and `train!`), ``W^T`` in `plasticity!` (`train!` only); every `max(1, round(τ / dt))` steps the weights are rescaled.
**Reference:** Turrigiano et al. (1998), Turrigiano (2008).

!!! note "Implementation note"
    `Wmin` is an offset added to every weight after rescaling, not a clamp; `Wmax` is a soft bound of the target sum ``W^T``, not of single weights.

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

Every `τ`, some synapses of one `SpikingSynapse` move to new targets with a new weight from
``\mathcal{N}(\mu, \sqrt{\mu})``. `ActivityDependentTurnover` rewires the synapses with the lowest
co-activity ``p_s = a^{pre}_{j(s)} a^{post}_{i(s)}`` (below the `fraction`-quantile), with traces

```math
a^{pre}_j \leftarrow a^{pre}_j + \frac{-a^{pre}_j\,dt + \delta_j}{\tau_{pre}}, \qquad
a^{post}_i \leftarrow a^{post}_i + \frac{-a^{post}_i\,dt + \delta_i}{\tau_{post}}
```

`RandomTurnover` rewires each synapse with probability `threshold`.

| Field | `ActivityDependentTurnover` | `RandomTurnover` | Units | Meaning |
|:------|:----------------------------|:-----------------|:------|:--------|
| `rate` | `-1` | `-1` | 1/ms | turnover rate, used only for the default of `τ` |
| `τ` | `1 / rate` | `1 / rate` | ms | interval between turnover events (pass `rate` or `τ`) |
| `fraction` | `0.1` | | | quantile of co-activity below which synapses are rewired |
| `threshold` | | `0.1` | | probability that a synapse is rewired at each event |
| `τpre`, `τpost` | `250ms` | | ms | time constants of the activity traces |
| `μ` | `3.0` | `3.0` | weight units | mean of new weights |

**Integration:** forward Euler for the activity traces at every step; rewiring every `max(1, round(τ / dt))` steps. **Reference:** none.

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
