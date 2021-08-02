using Plots
using SpikingNeuralNetworks
SNN.@load_units

E = SNN.IF(;N = 1)
E.I = [11]
SNN.monitor(E, [:v, :fire])

SNN.sim!([E], []; duration = 300ms)
SNN.vecplot(E, :v)
