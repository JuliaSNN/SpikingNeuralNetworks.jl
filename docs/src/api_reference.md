# API References

```@meta
CurrentModule = SpikingNeuralNetworks
```

## Types

### Populations
```@autodocs
Modules = [SpikingNeuralNetworks]
Order   = [:type]
Filter = t -> typeof(t) !== SNN.AbstractPopulation && t <: SNN.AbstractPopulation
```

### Synapses
```@autodocs
Modules = [SpikingNeuralNetworks]
Order   = [:type]
Filter = t -> typeof(t) !== SNN.AbstractConnection && t <: SNN.AbstractConnection
```

### Stimuli
```@autodocs
Modules = [SpikingNeuralNetworks]
Order   = [:type]
Filter = t -> typeof(t) !== SNN.AbstractStimulus && t <: SNN.AbstractStimulus
```

## Other types
```@autodocs
Modules = [SpikingNeuralNetworks]
Order   = [:type]
Filter = t -> !(t <: SNN.AbstractConnection || t <: SNN.AbstractPopulation || t <: SNN.AbstractStimulus)
```

## Functions

```@autodocs
Modules = [SpikingNeuralNetworks]
Order   = [:function]
```

## Helper macros

```@autodocs
Modules = [SpikingNeuralNetworks]
Order   = [:macro]
```