using SpikingNeuralNetworks
using Test
SNN.@load_units

if VERSION > v"1.1"
    include("ctors.jl")
end
##


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
include("poisson_stim.jl")
include("spiketime.jl")

# include("dendrite.jl")
# include("ballandstick.jl")
# include("tripod_network.jl")
#include("tripod_soma.jl")
#include("tripod.jl")
#include("tripod_network.jl")
#include("spiketime.jl")
#include("ballandstick.jl")
