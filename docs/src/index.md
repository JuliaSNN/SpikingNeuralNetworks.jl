# SpikingNeuralNetworks.jl Documentation



Julia Spiking Neural Networks (JuliaSNN) is a library for simulation of biophysical and abstract neuronal network models. 

This documentation is _work in progress_ please consider contacting me via the github repository if you have any specific questions. 

## Simple and powerful simulation framework

The library strength points are:
 - Modular, intuitive, and quick instantiation of complex biophysical models;
 - Large pool of standard models already available and easy implementation of custom new models;
 - High performance and native multi-threading support, laptop and cluster-friendly;
 - Access to all model's variables at runtime and save-load-rerun of arbitrarily complex networks;
 - Growing ecosystem for stimulation protocols, network analysis, and visualization ([SNNUtils](https://github.com/JuliaSNN/SNNUtils), [SNNPlots](https://github.com/JuliaSNN/SNNPlots), [SNNGeometry](https://github.com/JuliaSNN/SNNGeometry)).

`SpikingNeuralNetworks.jl` is defined within the `JuliaSNN` ecosystem, which offers `SNNPlots` to plot models' recordings and `SNNUtils` for further stimulation protocols and analysis.


## Models: populations, connections, and stimuli

SpikingNeuralNetworks.jl builds on the idea that a neural network is composed of three classes of objects, the network _populations_, their recurrent _connections_, and the external _stimuli_ they receive. Thus, a SNN model is simply a`NamedTuple` with keys: `pop`, `syn`, `stim`. The element associated to the keys must be concrete subtypes of `AbstractPopulation`, `AbstractConnection`, or `AbstractStimulus`. 

Models can be generated using `compose` with any population, connection, or stimulus type as keyworded arguments -The user can define the model by associating the correct subtypes to the named tuple, but we advise against it. For example:

```julia
using SpikingNeuralNetworks

E = SNN.IF(N = 100) # create an Integrate-and-Fire model population with 100 neurons. Use default parameters
EE = SNN.SpikingSynapse(E, E, :ge, w = rand(E.N, E.N)) # connect the populations with recurrent, spiking synapses, the synapse target the :ge field.
my_model = SNN.compose(E=E, EE=EE) # create a model with the E population and the EE connection.
# my_model = SNN.compose(;E, EE) # equivalent
```

`compose` assigns the correct typpes to the `pop` and `syn` and carries further integrity checks. 
The population and synapse elements will be assigned to `my_model.pop.E` and `my_model.syn.EE`, respectively.

`compose` can also be used recursively. This is useful when we want to instantiante multiple subnetworks with same population names.

```julia
using SpikingNeuralNetworks
subnet1 = let
    E = SNN.IF() # create an Integrate-and-Fire model
    EE = SNN.SpikingSynapse(E, E, :ge, w = zeros(E.N, E.N))
    SNN.compose(E=E, EE=EE)
end
subnet2 = let
    E = SNN.IF() # create an Integrate-and-Fire model
    EE = SNN.SpikingSynapse(E, E, :ge, w = zeros(E.N, E.N))
    SNN.compose(E=E, EE=EE)
end
model = SNN.compose(;subnet1, subnet2)
```
In this case the field `model.pop.subnet1_E` and `model.pop.subnet2_E` are instantiated.


!!! note
    - User are not expected to use the abstract types, which, but only their concrete subtypes.
    - Models must at least include one population. Connections and Stimuli always target one population.
    - Because in biophysical models connections are normally synapses, the two terms are used interchangably. 


### Pre-existing models

For each subtype, JuliaSNN offers a library of pre-existing models. In the aforementioned case, an integrate-and-fire population (`IF<:AbstractPopulation`), a spiking synapse (`SpikingSynapse<:AbstractSynapse`) and poisson-distributed spike train (`PoissonStimulus<:AbstractStimulus`). The collection of available models can be found under [Models](@ref).

Models can also be extended by importing the `AbstractPopulation`, `AbstractConnection`, or `AbstractStimulus` types. Guidelines on how to create new models are presented in [Models Extensions ](@ref) (WIP)


## Simulation

Leveraging Julia's [multiple dispatch](https://docs.julialang.org/en/v1/manual/methods/#Methods), the simulation loop calls the methods defined for each type of model and parameter:

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

In a loop step, the first to be activated are the stimuli which provide inputs to the populations. Thus, the differential equations associated to the populations are integrated. Finally, the population activity is propagated through the synapses (connections!). 

Using Julia's [passing-by-sharing](https://docs.julialang.org/en/v1/manual/functions/#man-argument-passing), connections and stimuli maintain internal pointers to the populations' fields they are attached to. This allow to seamlessy read and updates the populations variables within the `stimulate!` and `forward!` functions.


## Installation

JuliaSNN/SpikingNeuralNetworks.jl is not yet available on the public Julia repository! For the moment `]add SpikingNeuralNetworks` will still direct you to the old version of the package. 

You can install the latest version directly from the git repository:

```
]add https://github.com/JuliaSNN/SpikingNeuralNetworks.jl
```


To learn how to use the library you can follow the [Tutorial](@ref).





