# Stimuli


```@meta
CurrentModule = SpikingNeuralNetworks
```

```@contents
Pages = ["models.md"]
```

## Basic types

```@autodocs
Modules = [SNNModels]
Order   = [:type]
Filter = t -> t == SNNModels.AbstractParameter || t == SNNModels.AbstractConnectionParameter
```


## SpikeTime Stimulus
```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> t <: SNN.SpikeTimeStimulusParameter
```

```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> t <: SNN.SpikeTimeStimulus
```


## Poisson Layer


```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> t <: SNN.PoissonLayer
```

```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> t <: SNNModels.PoissonStimulusLayer
```


## Poisson Stimulus


```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> t <: SNN.PoissonStimulusParameter
```

```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> t <: SNN.PoissonStimulus
```

## Balanced Stimulus

```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> t <: SNNModels.BalancedParameter
```

## Stimulus Parameter

```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> t <: SNN.AbstractStimulus
```

```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> t <: SNN.AbstractStimulusParameter
```