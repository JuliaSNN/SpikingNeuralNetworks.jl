# Populations


```@meta
CurrentModule = SpikingNeuralNetworks
```

```@contents
Pages = ["models.md"]
```

## Generalized Integrate and Fire models

This set of models are implementations of the abstract type:
`AbstractGeneralizedIFParameter`. 

### Leaky Integrate and Fire model

The leaky integrate-and-fire is one of the simplest model for neuronal integration. It implements this equation:

```math
\begin{align}
    \frac{dV}{dt} &= - \frac{(V - E_l)}{\tau_m} + R (-w + I - I_{syn}) \\ \\
    \frac{dw}{dt} &= \frac{(a (V - E_l) - w)}{\tau_w}
\end{align}
```

where the adaptations parameters are optional and set to zero by default.


```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> t <: SNN.SNNModels.IFParameter 
```

```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> t <: SNN.IF 
```

### Adaptive Exponential model

The AdEx model is an expansion of the IF, it has a non-linear function of the membrane potential that is activated when the potential is above a threshold $\theta$

```math
\begin{align}
    \frac{dV}{dt} &= - \frac{(V - E_l)}{\tau_m} + \Delta T \exp{\frac{V - \theta}{\Delta T}} + R (-w + I - I_{syn}) \\ \\
    \frac{dw}{dt} &= \frac{(a (V - E_l) - w)}{\tau_w}
\end{align}
```

where the adaptations parameters are optional and set to zero by default.


```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> t <: SNN.SNNModels.AdExParameter 
```

```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> t <: SNN.AdEx
```

### Generalized IF synapses

Models of the type Generalized IF implements can implement these type of synapses:


```@autodocs
Modules = [SpikingNeuralNetworks, SNN.SNNModels]
Order   = [:type]
Filter = t -> t <: SNN.AbstractSynapseParameter
```