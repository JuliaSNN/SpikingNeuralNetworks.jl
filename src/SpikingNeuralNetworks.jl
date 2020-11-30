module SpikingNeuralNetworks

const SNN = SpikingNeuralNetworks

using SparseArrays
try
    using Reexport
    using Parameters
    using Requires

catch
    import Pkg; Pkg.add("Reexport")
    import Pkg; Pkg.add("Parameters")
    import Pkg; Pkg.add("Requires")
    using Reexport
    using Parameters
    using Requires
end
#using Unitful
#using Unitful.DefaultSymbols
#@reexport using Utils

const SNNInt = Int32
const SNNFloat = Float32
# srand(1000)

include("units.jl")
include("main.jl")
include("utils.jl")

include("neuron/if.jl")
include("neuron/if2.jl")
include("neuron/noisy_if.jl")
include("neuron/poisson.jl")
include("neuron/iz.jl")
include("neuron/iz_more_param.jl")
include("neuron/adexp.jl")

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
