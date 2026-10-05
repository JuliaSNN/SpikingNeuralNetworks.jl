# Model catalogue

```@meta
CurrentModule = SNNModels
```

This section lists every model that the JuliaSNN stack provides, one page per family. Each entry
gives the equations as implemented in the code, the parameter table with the default values of
the constructor, the integration scheme, a short example and the API reference of the source file
that defines it.

All quantities use the unit system of `SNNModels` (load it with `SNN.@load_units`): time in ms,
voltage in mV, current in pA, capacitance in pF, conductance in nS, length in cm, concentration in
μM, rates in events per ms (`Hz = 1e-3`). Writing parameters with units (`20ms`, `-70mV`, `281pF`)
keeps them correct whatever the internal base unit is.

A network is composed of three kinds of objects: populations (neuron models and spike sources),
connections (synapses between populations, with optional plasticity) and stimuli (external inputs
to a population). Populations of the generalized integrate-and-fire family also carry a synapse
model (how synaptic input is turned into a current) and a post-spike model (refractoriness).

| Family | Models | Page |
|:-------|:-------|:-----|
| Integrate-and-fire neurons | `IF`, `ExtendedIF`, `AdEx`, post-spike model `PostSpike` | [Integrate-and-fire neurons](neurons_if.md) |
| Other single-compartment neurons | `IZ` (Izhikevich), `HH` (Hodgkin-Huxley), `MorrisLecar` | [Izhikevich, Hodgkin-Huxley, Morris-Lecar](neurons_other.md) |
| Rate models | `Rate`, `WilsonCowan`, `HetRec` | [Rate models](rate_models.md) |
| Spike sources | `Poisson` (homogeneous, heterogeneous, variable rate), `InhomogeneousPoisson`, `Identity` | [Spike sources](sources.md) |
| Multicompartment neurons | `Tripod`, `BallAndStick`, dendrite geometry (`Dendrite`, `Physiology`) | [Multicompartment neurons](multicompartment.md) |
| Synapse and receptor models | `DeltaSynapse`, `CurrentSynapse`, `SingleExpSynapse`, `DoubleExpSynapse`, `DoubleExpCurrentSynapse`, `ReceptorSynapse`, `MultiReceptorSynapse`, `Confavreux2025Synapse`, `Receptor`, `NMDAVoltageDependency` | [Synapse and receptor models](synapses.md) |
| Connections | `SpikingSynapse`, `RateSynapse`, `FLSynapse`, `PINningSynapse`, `EmptySynapse`, connectivity rules of `sparse_matrix` | [Connections](connections.md) |
| Long- and short-term plasticity | `STDPGerstner`, `STDPTriplet`, `STDPWeightDependent`, `STDPMexicanHat`, `STDPConfavreux2025`, `STDPSymmetric`, `STDPAntiSymmetric`, `vSTDPParameter`, `iSTDPRate`, `iSTDPPotential`, `iSTDPTime`, `MarkramSTPParameter` | [Plasticity rules](plasticity_rules.md) |
| Metaplasticity | `SynapseNormalization` (`MultiplicativeNorm`, `AdditiveNorm`), `AggregateScaling`, `Turnover` (`RandomTurnover`, `ActivityDependentTurnover`) | [Metaplasticity](metaplasticity.md) |
| Stimuli | `PoissonStimulus`, `PoissonLayer`, `CurrentStimulus`, `SpikeTimeStimulus`, `BalancedStimulus`, `StimulusGroup`, `EmptyStimulus` | [Stimuli](stimuli.md) |
| Network models in SNNUtils | parameter sets and network builders in `SNNUtils/src/models` | [SNNUtils models](snnutils_models.md) |
| SNNUtils tools | sequence stimuli, E/I balance, classifiers, weight analysis | [SNNUtils tools](snnutils.md) |

Plasticity (long-term, short-term and metaplasticity) is applied only when the network is run with
`train!`; `sim!` integrates the same dynamics with frozen synapses (see [Plasticity](../plasticity.md)).

Some source files are present in the SNNModels source tree but are not loaded by the package
(their `include` is commented out); they are listed on the page of their family for completeness
and cannot be used without editing the package.
