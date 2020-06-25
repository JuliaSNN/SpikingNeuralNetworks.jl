using Plots, SpikingNeuralNetworks

const SNN = SpikingNeuralNetworks

E = SNN.IF(;N = 1)
E.I = [11]
SNN.monitor(E, [:v, :fire])

SNN.sim!([E], []; duration = 400SNN.ms)
SNN.vecplot(E, :v) |> display
