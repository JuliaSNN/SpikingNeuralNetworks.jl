using SpikingNeuralNetworks
using Plots
SNN.@load_units

E = SNN.MorrisLecar(N=1)
SNN.monitor!(E, [:v, :w])
model = SNN.merge_models(;E)
SNN.sim!(model, 1.5s)
E.I .=20pA
SNN.sim!(model, 1.5s)


plot(
SNN.vecplot(E, :v),
SNN.vecplot(E, :w)
)