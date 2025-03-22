using Plots
using SpikingNeuralNetworks
SNN.@load_units
import SpikingNeuralNetworks: AdExParameter
using Statistics, Random


E = SNN.AdExNeuron(; N = 800, param = AdExSynapseParameter(; El = -50mV))
I = SNN.IF(; N = 200, param = SNN.IFParameter())
# G = SNN.Rate(; N = 100)
EE = SNN.SpikingSynapse(E, E, :he; μ = 10, p = 0.02)
EI = SNN.SpikingSynapse(E, I, :ge; μ = 40, p = 0.02)
IE = SNN.SpikingSynapse(I, E, :hi; μ = -50, p = 0.02)
II = SNN.SpikingSynapse(I, I, :gi; μ = -10, p = 0.02)
# EG = SNN.SpikeRateSynapse(E, G; μ = 1.0, p = 0.02)
# GG = SNN.RateSynapse(G, G; μ = 1.2, p = 1.0)
# P = [E, I]
# C = [EE, EI, IE, II, EG]
# C = [EE, EG, GG]
model = merge_models(; E = E, I = I, EE = EE, EI = EI, IE = IE, II = II)

SNN.monitor(E, [:he, :h, :g, :v])
# SNN.monitor(G, [(:r)])
SNN.monitor(model.pop, [:fire])
SNN.sim!(model = model; duration = 4second)
SNN.raster(model.pop, [3.4s, 4s])
SNN.vecplot(E, :g, sym_id = 1, neurons = 1, r = 3:4s)
SNN.vecplot(E, :v, neurons = 1, r = 3:4s)

# Random.seed!(101)
# E = SNN.AdEx(;N = 100, param = AdExParameter(;El=-40mV))
# EE = SNN.SpikingSynapse(E, E, :ge; μ=10, p = 0.02)
# EG = SNN.SpikeRateSynapse(E, G; μ = 1., p = 1.0)
# SNN.monitor(E, [:fire])
# SNN.sim!(P, C; duration = 4second)
# SNN.raster([E], [900, 1000])
# plot!(xlims=(100,1000))

path = datadir("test") |> mkpath
info = (u = 4, v = 3, w = 2, x = 1)
save_model(; path, model, name = "AdEx_network", info = info, config = info)
