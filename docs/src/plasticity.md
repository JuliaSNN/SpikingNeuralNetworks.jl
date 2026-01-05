# Plasticity


## Hebbian Synaptic Plasticity

## Heterosynaptic Plasticity


```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> t <: SNN.SNNModels.AbstractMetaPlasticity 
```

```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> t <: SNN.SNNModels.MetaPlasticityParameter
```


```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> t <: SNN.SNNModels.AbstractSpikingSynapseParameter
```