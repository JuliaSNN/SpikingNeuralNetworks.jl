abstract type AbstractParameter end
abstract type AbstractSynapseParameter <: AbstractParameter end
abstract type AbstractNeuronParameter <: AbstractParameter end
abstract type AbstractSynapse end
abstract type AbstractNeuron end

abstract type AbstractSparseSynapse <: AbstractSynapse end
abstract type AbstractNormalization <: AbstractSynapse end

Spiketimes = Vector{Vector{Float32}}
Model = @NamedTuple{syn::Any, pop::Any}

export Spiketimes
