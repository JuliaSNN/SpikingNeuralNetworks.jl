# Model catalogue

```@meta
CurrentModule = SNNModels
```

Every model, one page per family: equations, parameters with defaults, an example and the API.
Units (`SNN.@load_units`): ms, mV, pA, pF, nS, cm, μM; rates per ms (`Hz = 1e-3`).

| Family | Models | Page |
|:-------|:-------|:-----|
| Integrate-and-fire neurons | `IF`, `ExtendedIF`, `AdEx`, post-spike model `PostSpike` | [Integrate-and-fire neurons](neurons_if.md) |
| Other single-compartment neurons | `IZ` (Izhikevich), `HH` (Hodgkin-Huxley), `MorrisLecar` | [Izhikevich, Hodgkin-Huxley, Morris-Lecar](neurons_other.md) |
| Rate models | `Rate`, `WilsonCowan`, `HetRec` | [Rate models](rate_models.md) |
| Spike sources | `Poisson` (homogeneous, heterogeneous, variable rate), `InhomogeneousPoisson`, `Identity` | [Spike sources](sources.md) |
| Multicompartment neurons | `Tripod`, `BallAndStick`, dendrite geometry (`Dendrite`, `Physiology`) | [Multicompartment neurons](multicompartment.md) |
| Synapse and receptor models | `DeltaSynapse`, `CurrentSynapse`, `SingleExpSynapse`, `DoubleExpSynapse`, `DoubleExpCurrentSynapse`, `ReceptorSynapse`, `MultiReceptorSynapse`, `Confavreux2025Synapse`, `Receptor`, `NMDAVoltageDependency` | [Synapse and receptor models](synapses.md) |
| Connections | `SpikingSynapse`, `RateSynapse`, `FLSynapse`, `PINningSynapse`, `EmptySynapse`, connectivity rules of `sparse_matrix` | [Connections](connections.md) |
| Long- and short-term plasticity | `STDPGerstner`, `STDPTriplet`, `STDPWeightDependent`, `STDPMexicanHat`, `STDPConfavreux2025`, `STDPSymmetric`, `STDPAntiSymmetric`, `vSTDPParameter`, `vSTDPDenseParameter`, `iSTDPRate`, `iSTDPPotential`, `MarkramSTPParameter` | [Plasticity rules](plasticity_rules.md) |
| Metaplasticity | `SynapseNormalization` (`MultiplicativeNorm`, `AdditiveNorm`), `AggregateScaling`, `Turnover` (`RandomTurnover`, `ActivityDependentTurnover`) | [Metaplasticity](metaplasticity.md) |
| Stimuli | `PoissonStimulus`, `PoissonLayer`, `CurrentStimulus`, `SpikeTimeStimulus`, `BalancedStimulus`, `StimulusGroup`, `EmptyStimulus` | [Stimuli](stimuli.md) |
| Short-term-plasticity parameters in SNNUtils | heterogeneous STP parameters sampled from experimental fits (`SNNUtils/src/models`) | [SNNUtils models](snnutils_models.md) |
| SNNUtils tools | sequence stimuli, E/I balance, classifiers, weight analysis | [SNNUtils tools](snnutils.md) |
