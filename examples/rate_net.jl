using Plots
using SpikingNeuralNetworks
SNN.@load_units

G = SNN.Rate(; N = 100)
GG = SNN.RateSynapse(G, G; μ = 1.2, p = 1.0)

SNN.monitor!(G, [(:r, [1, 50, 10, 20])])
model = merge_models(; G = G, GG = GG)

SNN.sim!(;model, duration = 100ms)
SNN.vecplot(G, :r)
