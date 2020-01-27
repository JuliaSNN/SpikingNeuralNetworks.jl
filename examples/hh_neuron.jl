using SpikingNeuralNetworks
SNN = SpikingNeuralNetworks
# varinfo()
include("../src/units.jl")
include("../src/plot.jl")
# ms = SNN.units.ms
# using Plots, SNN

E = SNN.HH(;N = 1)
E.I = [0.01825nA]


SNN.monitor(E, [:v])
SNN.sim!([E], []; dt = 0.015ms, duration = 2000ms)
SNN.vecplot(E, :v) |> display
