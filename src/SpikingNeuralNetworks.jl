module SpikingNeuralNetworks

export SNN
const SNN = SpikingNeuralNetworks

using LinearAlgebra
using SparseArrays
using Requires
using UnPack

include("unit.jl")
include("main.jl")
include("util.jl")

include("neuron/if.jl")
include("neuron/if2.jl")
include("neuron/noisy_if.jl")
include("neuron/poisson.jl")
include("neuron/iz.jl")
include("neuron/adexp.jl")
include("neuron/hh.jl")
include("neuron/rate.jl")

include("synapse/spiking_synapse.jl")
include("synapse/rate_synapse.jl")
include("synapse/fl_synapse.jl")
include("synapse/fl_sparse_synapse.jl")
include("synapse/pinning_synapse.jl")
include("synapse/pinning_sparse_synapse.jl")

function __init__()
    @require Plots="91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("plot.jl")
end

end
