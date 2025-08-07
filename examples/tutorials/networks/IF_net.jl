using SpikingNeuralNetworks
SNN.@load_units

E = SNN.IF(; N = 3200, param = SNN.IFParameter(; El = -49mV), name = "Excitatory")
I = SNN.IF(; N = 800, param = SNN.IFParameter(; El = -60mV), name = "Inhibitory")
EE = SNN.SpikingSynapse(E, E, :ge; μ = 0.2, p = 0.2)
EI = SNN.SpikingSynapse(E, I, :ge; μ = 0.5, p = 0.2)
IE = SNN.SpikingSynapse(I, E, :gi; μ = 2, p = 0.2)
II = SNN.SpikingSynapse(I, I, :gi; μ = 2, p = 0.2)
P = (; E, I)
C = (; EE, EI, IE, II)
model = SNN.merge_models(; P..., C..., name = "IF_network")

SNN.monitor!([E, I], [:fire])
SNN.sim!(; model, duration = 1second)
SNN.train!(; model, duration = 1second)
SNN.raster(model.pop, [0.5s, 1s])
