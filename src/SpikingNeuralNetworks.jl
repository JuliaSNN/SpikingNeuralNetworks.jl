module SpikingNeuralNetworks

export SNN

const SNN = SpikingNeuralNetworks

using LinearAlgebra
using SparseArrays
using Reexport
using Requires
using UnPack
using OrdinaryDiffEq
using DifferentialEquations
using Parameters
using RecursiveArrayTools
#using Unitful
#using Unitful.DefaultSymbols
#@reexport using Utils

include("units.jl")
include("main.jl")
include("utils.jl")

include("neuron/if.jl")
include("neuron/if2.jl")
include("neuron/noisy_if.jl")
include("neuron/poisson.jl")
include("neuron/iz.jl")
include("neuron/hh.jl")
include("neuron/rate.jl")

include("synapse/spiking_synapse.jl")
include("synapse/rate_synapse.jl")
include("synapse/pinning_full_synapse.jl")
# include("synapse/pinning_synapse.jl")
include("synapse/fl_full_synapse.jl")
# include("synapse/fl_synapse.jl")

function __init__()
    @require Plots="91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("plot.jl")
end

end
