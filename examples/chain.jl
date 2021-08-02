using Plots
using SpikingNeuralNetworks
SNN.@load_units

N = 3
E = SNN.IF(;N = N)
EE = SNN.SpikingSynapse(E, E, :ge; σ=0.5, p=0.8)
for n in 1:(N - 1)
    SNN.connect!(EE, n, n + 1, 50)
end
E.I[1] = 30

SNN.monitor(E, [(:v, [1, N])])
SNN.sim!([E], [EE]; duration = 100ms)
SNN.vecplot(E, :v)
