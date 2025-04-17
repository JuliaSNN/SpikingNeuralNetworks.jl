using SNNPlots
using SpikingNeuralNetworks
SNN.@load_units

Ne = 800;
Ni = 200;
E = SNN.IZ(;
    N = Ne,
    param = SNN.IZParameter(; a = 0.02, b = 0.2, c = -65, d = 8),
    name = "E",
)
I = SNN.IZ(;
    N = Ni,
    param = SNN.IZParameter(; a = 0.1, b = 0.2, c = -65, d = 2),
    name = "I",
)

EE = SNN.SpikingSynapse(E, E, :v; μ = 0.05, p = 0.8)
EI = SNN.SpikingSynapse(E, I, :v; μ = 0.5, p = 0.8)
IE = SNN.SpikingSynapse(I, E, :v; μ = -1.0, p = 0.8)
II = SNN.SpikingSynapse(I, I, :v; μ = -1.0, p = 0.8)
P = (; E, I)
C = (; EE, EI, IE, II)
model = merge_models(; P..., C..., name = "IZ_network")

SNN.monitor!([E, I], [:fire])
for t = 1:1000
    E.I .= 5randn(Ne)
    I.I .= 2randn(Ni)
    SNN.sim!(; model, duration = 1.0f0)
end

SNN.raster(model.pop, [0.5s, 1s])
