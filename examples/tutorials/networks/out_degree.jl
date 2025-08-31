##
using ThreadTools
using Plots
using UnPack
using Statistics
using SpikingNeuralNetworks
SNN.@load_units

import SpikingNeuralNetworks: IFSinExpParameter, IF, SpikingSynapse, compose, @update

config = (
    Npop = (E=4000, I=1000),
    neuron_param = IFSinExpParameter(),
    connections = (
        out_degree = (;rule=:Fixed,),
        E_to_E = (p = 0.05, μ = 2nS),
        ),
)

function create_network(config)
    @unpack connections, Npop = config
    E = IF(N=Npop.E, param=config.neuron_param, name="E")
    EE = SpikingSynapse(E, E, :ge, p=connections.E_to_E.p, μ=connections.E_to_E.μ, name="E_to_E"; connections.out_degree...)
    return compose(;E, EE, silent=true) 
end


##
p = plot()
for rule in [:Fixed, :Bernoulli, :PowerLaw]
    my_config = @update config begin
        connections.out_degree = rule == :PowerLaw ? (;rule=rule, γ=3, kmin=100) : (;rule=rule,)
    end
    model = create_network(my_config)
    length.(SNN.presynaptic(model.syn.EE)) |> x-> histogram!(x, bins=0:10:500, label=String(rule), alpha=0.8, lc=:auto, normed=true)
end

plot!(xlabel="Out degree", ylabel="Density", legend=:topright, size=(400,300), legend_title="Connection rule", legendtitlefontsize=8)
