# API References

```@meta
CurrentModule = SpikingNeuralNetworks
```

```@contents
Pages = ["api_reference.md"]
```


## Populations
```@autodocs
Modules = [SpikingNeuralNetworks]
Order   = [:type]
Filter = t -> typeof(t) !== SNN.AbstractPopulation && t <: SNN.AbstractPopulation
```

## Synapses
```@autodocs
Modules = [SpikingNeuralNetworks]
Order   = [:type]
Filter = t -> typeof(t) !== SNN.AbstractConnection && t <: SNN.AbstractConnection
```


## Stimuli
```@autodocs
Modules = [SpikingNeuralNetworks]
Order   = [:type]
Filter = t -> typeof(t) !== SNN.AbstractStimulus && t <: SNN.AbstractStimulus
```

## Plasticity
```@autodocs
Modules = [SpikingNeuralNetworks]
Order   = [:type]
Filter = t -> t <: SNN.PlasticityParameter
```

## Functions

```@autodocs
Modules = [SpikingNeuralNetworks]
Order   = [:function]
```

## Other types
```@autodocs
Modules = [SpikingNeuralNetworks]
Order   = [:type]
Filter = t -> !(t <: SNN.AbstractConnection || t <: SNN.AbstractPopulation || t <: SNN.AbstractStimulus || t <: SNN.PlasticityParameter)
```


## Helper macros

```@autodocs
Modules = [SpikingNeuralNetworks]
Order   = [:macro]
```