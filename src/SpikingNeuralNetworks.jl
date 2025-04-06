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



include("utils/macros.jl")
include("utils/structs.jl")
include("utils/unit.jl")
include("utils/util.jl")
include("utils/io.jl")
include("utils/graph.jl")
include("utils/record.jl")
include("utils/main.jl")
include("utils/spatial.jl")
include("analysis/spikes.jl")
include("analysis/populations.jl")
include("analysis/targets.jl")
include("neuron/neurons.jl")
include("synapse/synapses.jl")
include("stimulus/stimuli.jl")


function __init__()
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("plots/plot.jl")
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("plots/extra_plots.jl")
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("plots/stdp_plots.jl")
end

end
