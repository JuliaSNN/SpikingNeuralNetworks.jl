using Plots
using SpikingNeuralNetworks
SNN.@load_units

E = SNN.IF(; N = 3200, param = SNN.IFParameter(; El = -49mV))
I = SNN.IF(; N = 800, param = SNN.IFParameter(; El = -49mV))
EE = SNN.SpikingSynapse(E, E, :ge; μ = 0.5 * 0.27 / 10, p = 0.02)
EI = SNN.SpikingSynapse(E, I, :ge; μ = 0.5 * 0.27 / 10, p = 0.02)
IE = SNN.SpikingSynapse(I, E, :gi; μ = 20 * 4.5 / 10, p = 0.02)
II = SNN.SpikingSynapse(I, I, :gi; μ = 20 * 4.5 / 10, p = 0.02)
P = (; E, I)
C = (; EE, EI, IE, II)
model = merge_models(;P..., C..., name = "IF_network")

SNN.monitor!([E, I], [:fire])
SNN.sim!(;model, duration = 1second)
SNN.train!(;model, duration = 1second)
raster(model.pop, [0.5s, 1s])