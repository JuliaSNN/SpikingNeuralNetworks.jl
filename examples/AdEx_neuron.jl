using Plots
using SpikingNeuralNetworks
SNN.@load_units

E = SNN.AdExNeuron(; N = 1, param = SNN.AdExSynapseParameter(; El = -49mV))
SNN.monitor!(E, [:v, :fire, :w], sr = 8kHz)
model = merge_models(; E = E)
SNN.sim!(;model, duration = 700ms)
p1 = plot(SNN.vecplot(E, :w), SNN.vecplot(E, :v))

SNN.monitor!(E, [:v, :fire, :w], sr = 8kHz)
SNN.sim!([E]; duration = 700ms)
p2 = plot(SNN.vecplot(E, :w), SNN.vecplot(E, :v))
plot(p1, p2, layout = (2, 1), size = (800, 800))
