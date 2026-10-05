# Connections

```@meta
CurrentModule = SNNModels
```

A connection (`AbstractConnection`) links a presynaptic population to a target variable of a
postsynaptic population. At every time step, after the stimuli and the populations have been
updated, `sim!` and `train!` call for each connection, in the order of the model:

| Call | `sim!` | `train!` | Role |
|:-----|:------:|:--------:|:-----|
| `update_traces!(c, c.param, dt, T)` | no | yes | update the traces of the plasticity rules |
| `forward!(c, c.param, dt, T)` | yes | yes | add the presynaptic activity to the postsynaptic target |
| `plasticity!(c, c.param, dt, T)` | no | yes | apply long-term, short-term and metaplasticity rules |

Plasticity therefore runs only under `train!`. The plasticity rules themselves are described
in [Plasticity rules](plasticity_rules.md), the objects that act on the weights of other
connections in [Metaplasticity](metaplasticity.md).

All weights and per-synapse data are stored as `Float32`. Weights carry the units of the
target variable: with conductance-based synapse models they are increments of conductance
(nS), with current-based models increments of current (pA), and for rate models they are
dimensionless.

## SpikingSynapse

`SpikingSynapse` is the connection used between spiking populations. It stores a sparse
`N_post x N_pre` weight matrix and, at every step, adds the weight of every synapse whose
presynaptic neuron fired to the postsynaptic target.

```math
g_i \leftarrow g_i + \sum_{j \,:\, \text{fire}_j} W_{ij}\,\rho_{ij}
```

``g`` is the target variable selected by `sym` (and `comp` for multicompartment models),
``W_{ij}`` the weight and ``\rho_{ij}`` the short-term efficacy (1 unless a short-term
plasticity rule is set). The increment is applied once per spike; the postsynaptic synapse
model then integrates it (for example, the rise and decay of a double-exponential
conductance). For the generalized integrate-and-fire models, the symbols `:ge` and `:he` are
mapped to the excitatory receptor array `:glu`, `:gi` and `:hi` to the inhibitory array
`:gaba`; receptor-based synapse models and multicompartment neurons accept their own receptor
and compartment names (see [Synapse and receptor models](synapses.md) and
[Multicompartment neurons](multicompartment.md)).

### Constructor

<!-- norun -->
```julia
SpikingSynapse(pre, post, sym, comp = nothing; conn, delay_dist = nothing,
               LTPParam = NoLTP(), STPParam = NoSTP(), name = "SpikingSynapse")
```

| Argument | Default | Meaning |
|:---------|:--------|:--------|
| `pre`, `post` | required | presynaptic and postsynaptic populations |
| `sym` | required | target variable of `post` (`:ge`, `:gi`, `:he`, `:hi`, a receptor name, ...) |
| `comp` | `nothing` | target compartment of multicompartment models |
| `conn` | required | `NamedTuple` of `sparse_matrix` options, or an `N_post x N_pre` matrix |
| `delay_dist` | `nothing` | `Distribution` of per-synapse delays (ms); `nothing` means no delay |
| `LTPParam` | `NoLTP()` | long-term plasticity rule (an `LTPParameter`) |
| `STPParam` | `NoSTP()` | short-term plasticity rule (an `STPParameter`) |
| `name` | `"SpikingSynapse"` | name used by `print_model` and in records |
| `dt` | `0.125f0` | unused |

`LTPParam` and `STPParam` are keyword and field names; the abstract types of the rules are
`LTPParameter` and `STPParameter`. The names `LTPParam`, `STPParam` and `SpikingSynapseDelay`
appear in export lists of SpikingNeuralNetworks and SNNModels but are not defined.

When `pre === post`, autapses are removed structurally after the matrix has been drawn, so no
zero-weight self-synapse can be grown by plasticity.

### Connectivity: `conn`

A `NamedTuple` is passed to `sparse_matrix(pre.N, post.N; conn...)`, which builds the CSC
matrix directly (no dense `N_post x N_pre` array; since SNNModels 1.8.2):

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

One weight is drawn per connection; draws ``\le 0`` are not synapses and are removed (so with
`σ > 0` realised degrees can be lower than ``K``), and a negative `μ` makes all weights
negative (with a warning). Unknown keys are ignored silently.

A matrix `conn` (dense or sparse, any element type) of size `N_post x N_pre` is used as given,
converted to `SparseMatrixCSC{Float32,Int}`.

### Delays

With `delay_dist`, one delay ``d_{ij}`` (ms, `Float32`) is drawn per synapse. A spike of ``j``
at time ``t`` puts ``W_{ij}\rho_{ij}`` (evaluated at emission) in the queue of neuron ``i``
with arrival time ``t + d_{ij}``; at each step every queued increment with arrival time
``\le t`` is added to ``g_i``. The parameter of a delayed synapse is a
`SpikingSynapseDelayParameter` instead of `SpikingSynapseParameter`.

### Plasticity

The rule given with `LTPParam` modifies `W`, the rule given with `STPParam` modifies `ρ`; both
only under `train!`. `set_LTP!(c, state)`, `set_STP!(c, state)` and
`set_plasticity!(c, c.LTPParam, state)` switch a rule on or off, and
`update_plasticity!(c; LTP, STP)` replaces it. The two-argument `set_plasticity!(c, state)`
and `has_plasticity(c)` read `c.param.active`, which `SpikingSynapseParameter` does not have:
they raise an error for a `SpikingSynapse`.

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

Sparse connections (`AbstractSparseSynapse`) store the matrix twice, as returned by `dsparse`:

| Field | Content |
|:------|:--------|
| `colptr` | CSC column pointers: synapses of presynaptic neuron `j` are `colptr[j]:colptr[j+1]-1` |
| `I`, `J` | postsynaptic and presynaptic neuron of every synapse (CSC order) |
| `W` | weights in CSC order (all per-synapse arrays use this order) |
| `rowptr` | row pointers of the transposed matrix: positions `rowptr[i]:rowptr[i+1]-1` |
| `index` | map from row-major position to CSC position |

The weights onto neuron `i` are therefore `W[index[rowptr[i]:rowptr[i+1]-1]]`.

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

`connect!` and `update_sparse_matrix!` do not resize or reorder `ρ`, delays or plasticity
variables; use them only on connections without short-term plasticity and before creating
plasticity variables, or rebuild the connection.

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

Sparse connection between rate populations (fields `r` and `g`, e.g. `Rate`). Weights
``W = \mu X / \sqrt{p N_{pre}}`` with ``X`` a sparse standard-normal matrix of density ``p``.

```math
g_i \leftarrow g_i + \sum_j W_{ij}\, r_j
```

Under `train!` the weights follow, for every presynaptic neuron ``j``,

```math
\Delta_j = \eta\left(r_j - \sum_i r_i W_{ij}\right), \qquad W_{ij} \leftarrow W_{ij} + r_i\,\Delta_j
```

with ``\eta`` = `lr` of `RateSynapseParameter` (applied once per step, independent of `dt`).
No reference is given in the code for this rule.

| Keyword | Default | Meaning |
|:--------|:--------|:--------|
| `μ` | `0.0` | weight scale |
| `p` | `0.0` | density; must be positive (with `p = 0` every weight is `NaN`) |
| `param` | `RateSynapseParameter(lr = 1e-3)` | learning rate |

```julia
using SpikingNeuralNetworks
SNN.@load_units
R = SNN.Rate(N = 100)
RR = SNN.RateSynapse(R, R; μ = 1.0, p = 0.2)
SNN.train!([R], [RR]; duration = 10ms)
```

## FLSynapse (FORCE learning)

Dense recurrent connection between rate populations with a linear readout
``z = w^\top r`` fed back through the weights ``u``, trained by recursive least squares
(Sussillo and Abbott 2009, Neuron 63:544-557):

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

The target `f` is the scalar field `c.f` (default 0), to be set by the user. The readout
weights start as ``\mathcal{U}(-1,1)/\sqrt{N}``, the feedback weights as ``\mathcal{U}(-1,1)``.

In SNNModels 1.8.4 `FLSynapseParameter` is not a subtype of `AbstractConnectionParameter`, so
`sim!` and `train!` fail with a `MethodError` on this connection; `forward!(c, c.param)` and
`plasticity!(c, c.param, dt, T)` can be called directly:

```julia
using SpikingNeuralNetworks
SNN.@load_units
R = SNN.Rate(N = 100)
F = SNN.FLSynapse(R, R; μ = 1.5)
F.f = 0.5f0
for _ in 1:10
    SNN.SNNModels.forward!(F, F.param)
    SNN.SNNModels.plasticity!(F, F.param, 0.125f0, SNN.SNNModels.Time())
end
```

## PINningSynapse

Dense recurrent connection between rate populations trained with partial in-network training
(Rajan, Harvey and Tank 2016, Neuron 90:128-142): the recurrent weights are updated by
recursive least squares so that the input of each unit follows its target ``f_i``:

```math
q = P\, r, \qquad g = W r, \qquad C = \frac{1}{1 + q^\top r}, \qquad
W \leftarrow W + C (f - g)\, q^\top, \qquad P \leftarrow P - C\, q\, q^\top
```

Keywords and initialisation as `FLSynapse` (`μ = 1.5`, `α = 1`, `p` unused); `c.f` is a
vector of length `N_post`. The same restriction applies: `sim!`/`train!` do not dispatch on
`PINningSynapseParameter` in SNNModels 1.8.4.

```julia
using SpikingNeuralNetworks
SNN.@load_units
R = SNN.Rate(N = 100)
P = SNN.PINningSynapse(R, R; μ = 1.5)
P.f .= 0.1f0
SNN.SNNModels.forward!(P, P.param)
SNN.SNNModels.plasticity!(P, P.param, 0.125f0, SNN.SNNModels.Time())
```

## Non-exported connection types

| Type | Status in SNNModels 1.8.4 |
|:-----|:--------------------------|
| `FLSparseSynapse` | sparse FORCE learning; the constructor fails (vector minus scalar) and `forward!` uses an undefined `colptr` |
| `PINningSparseSynapse` | sparse PINning; constructor, `forward!` and `plasticity!` work when called directly, but `sim!`/`train!` do not dispatch on its parameter |
| `SpikeRateSynapse` | spikes of a spiking population added to `post.g` of a rate population; `sim!` works, `plasticity!` reads a missing field `rJ` |

## EmptySynapse

A connection whose `forward!` does nothing. `sim!` and `train!` use `[EmptySynapse()]` as the
default connection list.

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
