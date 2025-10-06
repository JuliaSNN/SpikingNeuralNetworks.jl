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
        E_to_E = (p = 0.05, μ = 2nS, ),
        ),
)
# #
p = plot()
for rule in [:Fixed, :Bernoulli, :PowerLaw]
    E = IF(N=4000, param=SNN.AdExParameter(), name="E")
    conn = (p = 0.05, μ = 2nS, rule=rule, γ=1, kmin=100)
    E_to_E = SpikingSynapse(E, E, :ge; conn, name="E_to_E")
    length.(SNN.postsynaptic(E_to_E)) |> x-> histogram!(x, bins=0:10:500, label=String(rule), alpha=0.8, lc=:auto, normed=true)
end

plot!(xlabel="Out degree", ylabel="Density", legend=:topright, size=(400,300), legend_title="Connection rule", legendtitlefontsize=8)
