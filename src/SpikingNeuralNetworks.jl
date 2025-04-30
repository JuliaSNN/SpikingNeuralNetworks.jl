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
using IterTools
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
end
