using SpikingNeuralNetworks
using Test
SNN.@load_units

if VERSION > v"1.1"
include("ctors.jl")
end
include("chain.jl")
include("hh_net.jl")
include("hh_neuron.jl")
include("if_net.jl")
include("if_neuron.jl")
include("iz_net.jl")
include("iz_neuron.jl")
include("oja.jl")
include("rate_net.jl")
include("stdp_demo.jl")
include("adexp_neuron.jl")
include("adexp_net.jl")
