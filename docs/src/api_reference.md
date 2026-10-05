# API References

```@meta
CurrentModule = SpikingNeuralNetworks
```

```@contents
Pages = ["api_reference.md"]
```


## Functions

```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:function]
```

## Plots

```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNPlots]
Order   = [:function]
```

## Plasticity rules and variables

Long-term plasticity rules are passed to `SpikingSynapse` with `LTPParam` and run under `train!`
(see [Plasticity](plasticity.md)). This section lists, among others, `STDPGerstner`, `STDPTriplet`,
`STDPWeightDependent`, `STDPConfavreux2025`, `STDPMexicanHat`, `iSTDPRate`, `iSTDPPotential`
and their variable types (`STDPVariables`, `STDPTripletVariables`, `iSTDPVariables`, ...).

```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> t <: SNNModels.PlasticityParameter || t <: SNNModels.PlasticityVariables
```

## Other types
```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> !(t <: SNNModels.AbstractComponent || t <: SNNModels.AbstractParameter ||
                t <: SNNModels.PlasticityParameter || t <: SNNModels.PlasticityVariables)
```


## Helper macros

```@autodocs
Modules = [SpikingNeuralNetworks, SNNModels]
Order   = [:macro]
```