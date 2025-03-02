module SpikingNeuralNetworks


SNN = SpikingNeuralNetworks
export SNN

using DrWatson
import Dates: now
using JLD2
using LinearAlgebra
using SparseArrays
using Requires
using Documenter
using UnPack
using Random
using Logging
using StaticArrays
using ProgressBars
using Parameters
using LoopVectorization
using ThreadTools
using Distributions
using Graphs, MetaGraphs



include("macros.jl")
include("structs.jl")
include("unit.jl")
include("util.jl")
include("io.jl")
include("graph.jl")
include("record.jl")
include("main.jl")
include("spikes.jl")
include("populations.jl")
include("synapse.jl")
include("neuron/neurons.jl")
include("synapse/synapses.jl")
include("stimulus/stimuli.jl")


function __init__()
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("plot.jl")
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("extra_plots.jl")
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("stdp_plots.jl")
end

end
