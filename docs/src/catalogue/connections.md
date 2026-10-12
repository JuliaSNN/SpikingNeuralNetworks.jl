# Connections

```@meta
CurrentModule = SNNModels
```

A connection links a presynaptic population to a target variable of a postsynaptic one. Weights
have the units of the target (nS for conductances, pA for currents) and the precision of the target
population. Plasticity: [Plasticity rules](plasticity_rules.md), [Metaplasticity](metaplasticity.md).

## SpikingSynapse

Sparse `N_post x N_pre` connection between spiking populations:

```math
g_i \leftarrow g_i + \sum_{j \,:\, \text{fire}_j} W_{ij}\,\rho_{ij}
```

``g`` is the target selected by `sym` (and `comp`), ``W`` the weight, ``\rho`` the short-term
efficacy (1 without a short-term rule).

### Connectivity: `conn`

A `NamedTuple` of `sparse_matrix` options, or an `N_post x N_pre` matrix:

| Key | Default | Meaning |
|:----|:--------|:--------|
| `p` or `ρ` | required (exactly one) | connection probability / density in [0, 1] |
| `μ` | `1` | location parameter of the weight distribution (sign flips all weights) |
| `σ` | `0` | scale parameter of the weight distribution |
| `dist` | `:Normal` | name of a two-parameter `Distributions` type, called as `dist(abs(μ), σ)` |
| `rule` | `:Fixed` | connectivity rule, see below |
| `γ`, `kmin` | `-1` | shape and scale of the Pareto out-degree distribution (`:PowerLaw` only) |

| `rule` | Structure |
|:-------|:----------|
| `:Fixed`, `:FixedIn` | each postsynaptic neuron receives exactly ``K = N_{pre} - \mathrm{round}((1-p) N_{pre})`` inputs, drawn without replacement |
| `:FixedOut` | each presynaptic neuron projects to exactly ``K = N_{post} - \mathrm{round}((1-p) N_{post})`` targets |
| `:Bernoulli` | each pair is connected independently with probability ``p`` |
| `:PowerLaw` | out-degree ``\min(\mathrm{round}(k), N_{post}-1)``, ``k \sim \mathrm{Pareto}(\gamma, k_{min})`` |

Draws ``\le 0`` are removed, so with `σ > 0` degrees can be lower than ``K``.

### Delays

With `delay_dist`, one delay ``d_{ij}`` (ms) is drawn per synapse; the increment arrives
``\lceil d_{ij}/dt \rceil`` steps after the spike. A delay common to all synapses without plasticity
reads `W` at arrival.

### Example

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 400)
I = SNN.IF(N = 100)
EE = SNN.SpikingSynapse(E, E, :ge; conn = (p = 0.1, μ = 2.0))
EI = SNN.SpikingSynapse(E, I, :ge; conn = (p = 0.2, μ = 3.0, σ = 0.5, rule = :Bernoulli))
IE = SNN.SpikingSynapse(I, E, :gi; conn = (p = 0.2, μ = 5.0),
                        LTPParam = SNN.iSTDPRate(r = 5Hz))
II = SNN.SpikingSynapse(I, I, :gi; conn = (p = 0.2, μ = 5.0),
                        delay_dist = SNN.SNNModels.Uniform(1ms, 2ms))
model = SNN.compose(; E, I, EE, EI, IE, II)
SNN.sim!(model; duration = 100ms)     # weights frozen
SNN.train!(model; duration = 100ms)   # iSTDP updates IE
size(SNN.matrix(IE))                  # (400, 100): N_post x N_pre
```

## Sparse storage and connectivity helpers

Storage of sparse connections (`dsparse`):

| Field | Content |
|:------|:--------|
| `colptr` | CSC column pointers: synapses of presynaptic neuron `j` are `colptr[j]:colptr[j+1]-1` |
| `I`, `J` | postsynaptic and presynaptic neuron of every synapse (CSC order) |
| `W` | weights in CSC order (all per-synapse arrays use this order) |
| `rowptr` | row pointers of the transposed matrix: positions `rowptr[i]:rowptr[i+1]-1` |
| `index` | map from row-major position to CSC position |

| Function | Returns |
|:---------|:--------|
| `matrix(c)`, `matrix(c, sym)` | `N_post x N_pre` sparse matrix of `W` or of the per-synapse field `sym` |
| `matrix_record(c, sym, t)` | matrix of the recorded field `sym` at time `t` (or a 3-d array for a vector of times) |
| `presynaptic(c[, i])`, `postsynaptic(c[, j])` | input neurons of `i` / target neurons of `j` |
| `presynaptic_idxs(c, i)` | row-major positions of the inputs of `i` (index `c.index` with them) |
| `postsynaptic_idxs(c, j)` | CSC positions of the synapses of `j` (index `c.W` with them) |
| `indices(c, js, is)` | CSC positions of the synapses from `js` to `is` |
| `update_weights!(c, j, i, w)` | set the weight of existing synapses |
| `connect!(c, j, i, μ)` | set or create the synapse `j -> i` and rebuild the storage |
| `update_sparse_matrix!(c[, W])` | rebuild the storage from `W` or from `c.I`, `c.J`, `c.W` |

`J`, `index` and `ρ` of a `SpikingSynapse` may be empty: call `SNN.SNNModels.ensure_transpose!(c)`
before reading `J`/`index`, read `ρ` with `SNN.SNNModels.efficacy(c)`.

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 50)
I = SNN.IF(N = 20)
EI = SNN.SpikingSynapse(E, I, :ge; conn = (p = 0.2, μ = 2.0))
inputs_of_1 = SNN.SNNModels.presynaptic(EI, 1)
w_in_1 = EI.W[EI.index[SNN.SNNModels.presynaptic_idxs(EI, 1)]]
targets_of_1 = SNN.SNNModels.postsynaptic(EI, 1)
SNN.SNNModels.update_weights!(EI, 1, targets_of_1[1], 4.0f0)
w = SNN.SNNModels.sparse_matrix(100, 40; p = 0.1, rule = :PowerLaw, γ = 2.0, kmin = 2)
```

## RateSynapse

Sparse connection between rate populations, ``W = \mu X / \sqrt{p N_{pre}}`` (``X`` sparse standard normal).

```math
g_i \leftarrow g_i + \sum_j W_{ij}\, r_j
```

Under `train!`:

```math
\Delta_j = \eta\left(r_j - \sum_i r_i W_{ij}\right), \qquad W_{ij} \leftarrow W_{ij} + r_i\,\Delta_j
```

with ``\eta`` = `lr` (once per step).

| Keyword | Default | Meaning |
|:--------|:--------|:--------|
| `μ` | `0.0` | weight scale |
| `p` | `0.0` | density in `(0, 1]` |
| `param` | `RateSynapseParameter(lr = 1e-3)` | learning rate |

```julia
using SpikingNeuralNetworks
SNN.@load_units
R = SNN.Rate(N = 100)
RR = SNN.RateSynapse(R, R; μ = 1.0, p = 0.2)
SNN.train!([R], [RR]; duration = 10ms)
```

## FLSynapse (FORCE learning)

Dense rate connection with a readout ``z = w^\top r`` fed back through ``u``, trained by
recursive least squares towards the target `c.f` (Sussillo and Abbott 2009):

```math
q = P\, r, \qquad g = W r + z\, u, \qquad
C = \frac{1}{1 + q^\top r}, \qquad w \leftarrow w + C (f - z)\, q, \qquad
P \leftarrow P - C\, q\, q^\top
```

| Keyword | Default | Meaning |
|:--------|:--------|:--------|
| `μ` | `1.5` | gain: ``W_{ij} = \mu\,\xi_{ij}/\sqrt{N_{pre}}``, ``\xi \sim \mathcal{N}(0,1)`` |
| `α` | `1` | ``P = \alpha \mathbb{1}`` at start |
| `p` | `0.0` | unused (dense matrix) |

```julia
using SpikingNeuralNetworks
SNN.@load_units
R = SNN.Rate(N = 100)
F = SNN.FLSynapse(R, R; μ = 1.5)
T = SNN.SNNModels.Time()
for k in 1:800                          # 100 ms at dt = 0.125 ms
    F.f = sin(2π * k * 0.125 / 20)      # target: 50 Hz sine
    SNN.train!([R], [F], SNN.SNNModels.AbstractStimulus[], 0.125f0, T)
end
```

## PINningSynapse

Dense rate connection whose recurrent weights learn the targets ``f_i`` (Rajan, Harvey and Tank 2016):

```math
q = P\, r, \qquad g = W r, \qquad C = \frac{1}{1 + q^\top r}, \qquad
W \leftarrow W + C (f - g)\, q^\top, \qquad P \leftarrow P - C\, q\, q^\top
```

```julia
using SpikingNeuralNetworks
SNN.@load_units
R = SNN.Rate(N = 100)
P = SNN.PINningSynapse(R, R; μ = 1.5)
P.f .= 0.1f0
SNN.train!([R], [P]; duration = 10ms)
```

## Non-exported connection types

| Type | Description |
|:-----|:------------|
| `FLSparseSynapse` | sparse FORCE learning (`p` in `(0, 1]`); runs under `sim!`/`train!` |
| `PINningSparseSynapse` | sparse PINning (`p` in `(0, 1]`); runs under `sim!`/`train!` |
| `SpikeRateSynapse` | spikes of a spiking population as delta inputs to `post.g` of a rate population (`x` jumps by `W`); no learning rule |

## EmptySynapse

A connection that does nothing; the default connection list of `sim!`/`train!`.

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.IF(N = 10)
SNN.sim!([E], [SNN.EmptySynapse()]; duration = 10ms)
```

## API

```@autodocs
Modules = [SNNModels]
Pages   = ["connections/connections.jl", "connections/empty.jl", "connections/spiking_synapse.jl"]
```

### Connectivity and sparse storage

```@autodocs
Modules = [SNNModels]
Pages   = ["utils/sparse_matrix.jl"]
```

### Rate connections

```@autodocs
Modules = [SNNModels]
Pages   = ["connections/rate_synapse.jl", "connections/fl_synapse.jl", "connections/fl_sparse_synapse.jl", "connections/pinning_synapse.jl", "connections/pinning_sparse_synapse.jl", "connections/spike_rate_synapse.jl"]
```
