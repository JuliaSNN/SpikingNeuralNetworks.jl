abstract type AbstractParameter end
abstract type AbstractConnectionParameter <: AbstractParameter end
abstract type AbstractPopulationParameter <: AbstractParameter end
abstract type AbstractStimulusParameter <: AbstractParameter end
abstract type AbstractNeuronParameter <: AbstractParameter end
abstract type AbstractSynapseParameter <: AbstractParameter end
abstract type AbstractConnection end
abstract type AbstractPopulation end
abstract type AbstractStimulus end

abstract type AbstractSparseSynapse <: AbstractConnection end
abstract type AbstractNormalization <: AbstractConnection end

Spiketimes = Vector{Vector{Float32}}

@snn_kw struct EmptyParam
    type::Symbol = :empty
end



export Spiketimes