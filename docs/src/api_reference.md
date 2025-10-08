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

## Other types
```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> !(t <: SNNModels.AbstractComponent || t <: SNNModels.AbstractParameter)
```


## Helper macros

```@autodocs
Modules = [SpikingNeuralNetworks, SNNModels]
Order   = [:macro]
```