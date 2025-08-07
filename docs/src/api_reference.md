# API References

```@meta
CurrentModule = SpikingNeuralNetworks
```

```@contents
Pages = ["api_reference.md"]
```


## Populations
```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> typeof(t) !== SNNModels.AbstractPopulation && t <: SNNModels.AbstractPopulation
```

## Synapses
```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> t <: SNNModels.AbstractConnection
```


## Stimuli
```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> typeof(t) !== SNNModels.AbstractStimulus && t <: SNNModels.AbstractStimulus
```

## Plasticity
```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> t <: SNNModels.PlasticityParameter
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

## Other types
```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> !(t <: SNNModels.AbstractConnection || t <: SNNModels.AbstractPopulation || t <: SNNModels.AbstractStimulus || t <: SNNModels.PlasticityParameter)
```


## Helper macros

```@autodocs
Modules = [SpikingNeuralNetworks, SNNModels]
Order   = [:macro]
```