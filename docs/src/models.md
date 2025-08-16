# Models


```@meta
CurrentModule = SpikingNeuralNetworks
```

```@contents
Pages = ["models.md"]
```


## Populations

### Types
```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> typeof(t) !== SNNModels.AbstractPopulation && t <: SNNModels.AbstractPopulation
```

### Parameters
```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> typeof(t) !== SNNModels.AbstractPopulationParameter && t <: SNNModels.AbstractPopulationParameter
```

## Synapses
```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> t <: SNNModels.AbstractConnection
```

## Plasticity
```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> t <: SNNModels.PlasticityParameter
```


## Stimuli

### Types

```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> typeof(t) !== SNNModels.AbstractStimulus && t <: SNNModels.AbstractStimulus
```

```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> typeof(t) !== SNNModels.AbstractStimulusParameter && t <: SNNModels.AbstractStimulusParameter
```

