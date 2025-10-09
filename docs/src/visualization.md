# Plots




#### Other models

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



