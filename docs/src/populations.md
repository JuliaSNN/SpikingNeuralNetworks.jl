# Populations

```@meta
CurrentModule = SNNModels
```

A population is a group of `N` model units of the same type, stored as one Julia struct whose
state variables are vectors of length `N`. Every population is a subtype of
`AbstractPopulation` and is advanced by one time step with
`integrate!(population, population.param, dt)`, called by `sim!` and `train!` after the stimuli
and before the connections. The full list of models, with equations and parameters, is in the
[Model catalogue](catalogue/index.md); this page explains what the models have in common.

## Fields shared by all populations

| Field | Meaning |
|:------|:--------|
| `id::String` | unique identifier (random 12-character string), used by connections and stimuli to refer to the population |
| `name::String` | human-readable name, used in model printouts and recordings |
| `param` | parameter object; its type selects the `integrate!` method |
| `N` | number of units |
| `records::Dict` | recordings of the population (see [Recordings](recordings.md)) |

Spiking populations also have `fire::Vector{Bool}`, the spike flags of the last step; most
neuron models have `v` (membrane potential, mV) and `I` (external current, pA, or a
dimensionless input for `IZ` and the rate units). Any vector field can be recorded with
`monitor!(population, [:field, ...])`.

## Generalized integrate-and-fire populations

`IF`, `AdEx`, `Tripod` and `BallAndStick` are subtypes of `AbstractGeneralizedIF`. They combine
three independent objects:

- `param`: the neuron model, e.g. `IFParameter` or `AdExParameter`;
- `synapse`: the synapse model, any `AbstractSynapseParameter`
  (`DoubleExpSynapse()` by default), see [Synapse and receptor models](catalogue/synapses.md);
- `spike`: the spike parameters `PostSpike` (refractory period, adaptive threshold).

```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.AdEx(N = 100, param = SNN.AdExParameter(b = 0pA),
             synapse = SNN.SingleExpSynapse(), spike = SNN.PostSpike(τabs = 2ms))
I = SNN.IF(N = 25, param = SNN.IFParameter(τm = 10ms), synapse = SNN.DoubleExpSynapse())
```

One step of a generalized IF population is: (1) `update_synapses!` adds the spikes delivered
by the connections in the previous step to the synaptic variables and integrates them,
(2) `synaptic_current!` computes the synaptic current `syn_curr` from the synaptic variables
and the membrane potential, (3) `update_neuron!` integrates the membrane and adaptation
equations, detects spikes and applies reset and refractoriness. All schemes are forward Euler.
For `IF` the membrane update is

```math
v \leftarrow v + \frac{dt}{\tau_m}\left(-(v - E_l) + R\,(I - w) - R\, I_{syn}\right),
```

i.e. ``\tau_m\, dv/dt = -(v - E_l) + R\,(I - w) - R\, I_{syn}`` with
``I_{syn}`` = `syn_curr`; see [Integrate-and-fire neurons](catalogue/neurons_if.md) for `AdEx`
and the details.

`Population(param; synapse, N, ...)` builds the population that matches a parameter object
(e.g. `Population(AdExParameter(); synapse = DoubleExpSynapse(), N = 100)` returns an `AdEx`).

## Synaptic targets

A connection (`SpikingSynapse(pre, post, target; ...)`) writes presynaptic spikes into a
field of the postsynaptic population selected by the target symbol. For generalized IF
populations the spikes go to the `receptors` buffers of the synapse model:

| Target symbol | Two-receptor synapse models (`DoubleExpSynapse`, `SingleExpSynapse`, `CurrentSynapse`, `DeltaSynapse`, ...) |
|:--------------|:------|
| `:ge`, `:glu`, `:he` | excitatory buffer `receptors.glu` |
| `:gi`, `:gaba`, `:hi` | inhibitory buffer `receptors.gaba` |
| any other symbol | used as is (e.g. a receptor name of `ReceptorSynapse`) |

Multicompartment models additionally take the compartment index or name as fourth argument
(see [Multicompartment neurons](catalogue/multicompartment.md)). Other models expose their
conductances directly: `IZ` and `HH` use `:ge` and `:gi`; `Rate` maps every target to `g`;
`Identity` maps every target to `g`; `HetRec` uses `:glu` and `:gaba` on its dendrites;
`MorrisLecar` uses `:ge`, `:gi`; `ExtendedIF` uses `:g_Exc`, `:g_PV`, `:g_SST`; `WilsonCowan`
maps every target to `g`. (In SNNModels 1.8.4 `MorrisLecar`, `ExtendedIF` and `WilsonCowan`
could not receive connections.)

## Choosing a model

| Need | Model | Page |
|:-----|:------|:-----|
| fast point neuron, current or conductance synapses | `IF` | [Integrate-and-fire neurons](catalogue/neurons_if.md) |
| spike initiation and adaptation, heterogeneous parameters | `AdEx` | [Integrate-and-fire neurons](catalogue/neurons_if.md) |
| three conductances (E, PV, SST) with interaction | `ExtendedIF` | [Integrate-and-fire neurons](catalogue/neurons_if.md) |
| firing patterns of the Izhikevich model | `IZ` | [Izhikevich, Hodgkin-Huxley, Morris-Lecar](catalogue/neurons_other.md) |
| conductance-based spike generation | `HH`, `MorrisLecar` | [Izhikevich, Hodgkin-Huxley, Morris-Lecar](catalogue/neurons_other.md) |
| dendritic integration, NMDA spikes | `Tripod`, `BallAndStick` | [Multicompartment neurons](catalogue/multicompartment.md) |
| firing-rate units | `Rate`, `WilsonCowan` | [Rate models](catalogue/rate_models.md) |
| stochastic units with heterogeneous dendritic timescales | `HetRec` | [Rate models](catalogue/rate_models.md) |
| background or input spike trains | `Poisson`, `VariablePoisson`, `InhomogeneousPoisson` | [Spike sources](catalogue/sources.md) |
| relaying spikes | `Identity` | [Spike sources](catalogue/sources.md) |

Plasticity of the connections is applied only by `train!`. Every population type runs under both
`sim!` and `train!` (in SNNModels 1.8.4 `IZ`, `HH` and `MorrisLecar` failed under `train!`).

New population models can be added by defining a parameter type, a population struct and an
`integrate!` method, see [Model Extensions](models_ext.md).
