# SpikingNeuralNetworks.jl Documentation


## Simple and powerful simulation framework

Julia Spiking Neural Networks (JuliaSNN) is a library for simulation of biophysical and abstract neuronal network models. 

The library strength points are:
 - Modular, intuitive, and quick instantiation of complex biophysical models;
 - Large pool of standard models already available and easy implementation of custom new models;
 - High performance and native multi-threading support, laptop and cluster-friendly;
 - Access to all model's variables at runtime and save-load-rerun of arbitrarily complex networks;
 - Growing ecosystem for stimulation protocols, network analysis, and visualization ([SNNUtils](https://github.com/JuliaSNN/SNNUtils), [SNNPlots](https://github.com/JuliaSNN/SNNPlots), [SNNGeometry](https://github.com/JuliaSNN/SNNGeometry)).

`SpikingNeuralNetworks.jl` is defined within the `JuliaSNN` ecosystem, which offers `SNNPlots` to plot models' recordings and `SNNUtils` for further stimulation protocols and analysis.

![A raster plot of a spiking neural network](assets/spiking.png)

## Models: populations, connections, and stimuli

SpikingNeuralNetworks.jl builds on the idea that a neural network is composed of three classes of objects, the network _populations_, their recurrent _connections_, and the external _stimuli_ they receive. Thus, a SNN model is simply a`NamedTuple` with keys: `pop`, `syn`, `stim`. The element associated to the keys must be concrete subtypes of `AbstractPopulation`, `AbstractConnection`, or `AbstractStimulus`. 

The user can define the model by associating the correct subtypes to the named tuple, however, the simplest usage is calling the function `merge_models` with any population, connection, or stimulus type as keyworded arguments. For example:

```julia
using SpikingNeuralNetworks
E = SNN.IF() # create an Integrate-and-Fire model
EE = SNN.SpikingSynapse(E, E, :ge, w = zeros(E.N, E.N))
my_model = SNN.merge_models(E=E, EE=EE)
```

The function assigns the correct subtypes to the three keys and makes further checks that the connections bind populations included in the model. 
The population and synapse elements will be assigned to `my_model.pop.E` and `my_model.syn.EE`, respectively.

`merge_models` can also be used recursively. This is useful when we want to instantiante multiple subnetworks with same population names.

```julia
using SpikingNeuralNetworks
subnet1 = let
    E = SNN.IF() # create an Integrate-and-Fire model
    EE = SNN.SpikingSynapse(E, E, :ge, w = zeros(E.N, E.N))
    SNN.merge_models(E=E, EE=EE)
end
subnet2 = let
    E = SNN.IF() # create an Integrate-and-Fire model
    EE = SNN.SpikingSynapse(E, E, :ge, w = zeros(E.N, E.N))
    SNN.merge_models(E=E, EE=EE)
end
model = SNN.merge_models(;subnet1, subnet2)
```
In this case the field `model.pop.subnet1_E` and `model.pop.subnet2_E` are instantiated.


NB.
User are not expected to use the abstract types, which, but only their concrete subtypes.  . Populations are the fundamental block, a model must at least include one population.
Because in biophysical models connections are normally synapses, the two terms are used interchangably. In future version we may change the AbstractConnection type for AbstractSynapse, thus the explicit use of this abstract type is strongly discouraged.




The classical network model is the __Balanced Network__ by [Brunel, 2000](https://link.springer.com/article/10.1023/A:1008925309027)



```md
================
[Info:  Model: Balanced network
[Info:  ----------------
[Info:  Populations (2):
[Info:  E         : IF        :  4000       IFParamete
[Info:  I         : IF        :  1000       IFParamete
[Info:  ----------------
[Info:  Synapses (4): 
[Info:  E_to_E             : E -> E.ge                     :          : NoLTP      : NoSTP     
[Info:  E_to_I             : E -> I.ge                     :          : NoLTP      : NoSTP     
[Info:  I_to_E             : I -> E.gi                     :          : NoLTP      : NoSTP     
[Info:  I_to_I             : I -> I.gi                     :          : NoLTP      : NoSTP     
[Info:  ----------------
[Info:  Stimuli (2):
[Info:  noiseE     : noiseE -> E.ge                 PoissonStimulus
[Info:  noiseI     : noiseI -> I.ge                 PoissonStimulus
[Info:  ================
```

For each subtype, JuliaSNN offers a library of pre-existing models. In the aforementioned case, an integrate-and-fire population (`IF<:AbstractPopulation`), a spiking synapse (`SpikingSynapse<:AbstractSynapse`) and poisson-distributed spike train (`PoissonStimulus<:AbstractStimulus`).

Leveraging the Julia's [multiple dispatch](https://docs.julialang.org/en/v1/manual/methods/#Methods), the simulation loop calls the methods defined for each type of model and parameter:

```julia

function sim!(...)
    update_time!(T, dt)
    for s in stimuli
        s_type = getfield(s, :param)
        stimulate!(s, s_type, T, dt)
        record!(s, T)
    end
    for p in populations
        p_type = getfield(t, :param)
        integrate!(p, p_type, dt)
        record!(p, T)
    end
    for c in connections
        c_type = getfield(c, :param)
        forward!(c, c_type)
        ## if train!(...) 
            plasticity!(c, c.param, dt, T)
        record!(c, T)
    end
end
```

This allows for great flexibility, 

Populations are stimulated, integrated, and their information is propagated through connections. 
Connections and stimuli objects maintain internal pointers to the populations' fields they are attached to. This allow to seamlessy read and updates the populations fields within the `stimulate!` and `forward!` functions.



## Installation

JuliaSNN is now available on the public Julia repository!
You can easily install the last stable release via:

```
]add SpikingNeuralNetworks
```

otherwise, you can install the most recent updates from the git repository:

```
]add https://github.com/JuliaSNN/SpikingNeuralNetworks.jl
```

The collection of available models can be found under [Models](@ref) (WIP)
Models can be easily extended by importing the `AbstractPopulation`, `AbstractConnection`, or `AbstractStimulus` types. Guidelines on how to create a model are presented in [Models Extensions ](@ref) (WIP)

To learn how to use the library you can follow the [Tutorial](@ref).





