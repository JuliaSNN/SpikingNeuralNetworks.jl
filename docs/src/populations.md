# Populations

```@meta
CurrentModule = SNNModels
```

After this page you can build a population, choose its neuron, synapse and spike models, and
know where connections deliver their input. The equations of every model are in the
[Model catalogue](catalogue/index.md).

A population is `N` units of one model; its state variables are vectors of length `N`.

```@example populations
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.AdEx(N = 100, param = SNN.AdExParameter(b = 0pA),
             synapse = SNN.SingleExpSynapse(), spike = SNN.PostSpike(τabs = 2ms))
I = SNN.IF(N = 25, param = SNN.IFParameter(τm = 10ms, R = 0.06), name = "I")
E.v[1:5], I.N
```

Every population has `id`, `name`, `param`, `N` and `records`; spiking populations have
`fire::Vector{Bool}`, the spikes of the last step. Most neuron models have `v` (mV) and an
external current `I` (pA). Any vector field can be recorded with `monitor!`.

## Integrate-and-fire populations

`IF`, `AdEx`, `Tripod` and `BallAndStick` combine three independent objects:

- `param`: the neuron model (`IFParameter`, `AdExParameter`, ...);
- `synapse`: how input becomes a current, any [synapse model](catalogue/synapses.md)
  (`DoubleExpSynapse()` by default);
- `spike`: reset, refractory period and adaptive threshold (`PostSpike`).

A step integrates the synaptic variables, computes the synaptic current, then integrates the
membrane, detects spikes and resets. For `IF` (forward Euler):

```math
\tau_m \frac{dv}{dt} = -(v - E_l) + R\,(I - w) - R\, I_{syn}.
```

`Population(param; kwargs...)` builds the population that matches a parameter object, e.g.
`Population(AdExParameter(); synapse = DoubleExpSynapse(), N = 100)` returns an `AdEx`.

## Synaptic targets

Connections and spiking stimuli write into the variable named by their target symbol:

| Population | Target symbols |
|:-----------|:---------------|
| `IF`, `AdEx`, `Tripod`, `BallAndStick` | `:ge`, `:glu`, `:he` (excitatory); `:gi`, `:gaba`, `:hi` (inhibitory); receptor names of `ReceptorSynapse` |
| `ExtendedIF` | `:g_Exc`, `:g_PV`, `:g_SST` |
| `IZ`, `HH`, `MorrisLecar` | `:ge`, `:gi` |
| `HetRec` | `:glu`, `:gaba` (dendrites) |
| `Rate`, `WilsonCowan`, `Identity` | any symbol, mapped to `g` |

Multicompartment models take the compartment as an extra argument (`:s`, `:d`, `:d1`, `:d2`; see
[Multicompartment neurons](catalogue/multicompartment.md)).

## Choosing a model

| Need | Model | Page |
|:-----|:------|:-----|
| fast point neuron | `IF` | [Integrate-and-fire neurons](catalogue/neurons_if.md) |
| spike initiation and adaptation, heterogeneous parameters | `AdEx` | [Integrate-and-fire neurons](catalogue/neurons_if.md) |
| three conductances (E, PV, SST) | `ExtendedIF` | [Integrate-and-fire neurons](catalogue/neurons_if.md) |
| Izhikevich firing patterns | `IZ` | [Izhikevich, Hodgkin-Huxley, Morris-Lecar](catalogue/neurons_other.md) |
| conductance-based spike generation | `HH`, `MorrisLecar` | [Izhikevich, Hodgkin-Huxley, Morris-Lecar](catalogue/neurons_other.md) |
| dendritic integration, NMDA spikes | `Tripod`, `BallAndStick` | [Multicompartment neurons](catalogue/multicompartment.md) |
| firing-rate units | `Rate`, `WilsonCowan` | [Rate models](catalogue/rate_models.md) |
| stochastic units with heterogeneous dendritic timescales | `HetRec` | [Rate models](catalogue/rate_models.md) |
| background or input spike trains | `Poisson`, `VariablePoisson`, `InhomogeneousPoisson` | [Spike sources](catalogue/sources.md) |
| relaying spikes | `Identity` | [Spike sources](catalogue/sources.md) |

Populations are `Float32` unless built with `VFT = Vector{Float64}` (see
[Running simulations](simulation.md#Precision)). New models: [Model extensions](models_ext.md).
