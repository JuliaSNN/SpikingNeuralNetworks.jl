# Firing patterns of the Izhikevich model.
using SpikingNeuralNetworks
using CairoMakie
SNN.@load_units

RS = SNN.IZ(; N = 1, param = SNN.IZParameter(; a = 0.02, b = 0.2, c = -65, d = 8))
IB = SNN.IZ(; N = 1, param = SNN.IZParameter(; a = 0.02, b = 0.2, c = -55, d = 4))
CH = SNN.IZ(; N = 1, param = SNN.IZParameter(; a = 0.02, b = 0.2, c = -50, d = 2))
FS = SNN.IZ(; N = 1, param = SNN.IZParameter(; a = 0.1, b = 0.2, c = -65, d = 2))
TC1 = SNN.IZ(; N = 1, param = SNN.IZParameter(; a = 0.02, b = 0.25, c = -65, d = 0.05))
TC2 = SNN.IZ(; N = 1, param = SNN.IZParameter(; a = 0.02, b = 0.25, c = -65, d = 0.05))
RZ = SNN.IZ(; N = 1, param = SNN.IZParameter(; a = 0.1, b = 0.26, c = -65, d = 2))
LTS = SNN.IZ(; N = 1, param = SNN.IZParameter(; a = 0.1, b = 0.25, c = -65, d = 2))
P = (; RS, IB, CH, FS, TC1, TC2, RZ, LTS)
model = SNN.compose(; P..., name = "IZ_neurons", silent = true)

SNN.monitor!(model.pop, [:v])
T = 2second
dt = 0.125f0
for t = 0:dt:T     # inputs change during the run: one step at a time
    for p in [RS, IB, CH, FS, LTS]
        p.I .= 10
    end
    TC1.I .= (t < 0.2T) ? 0 : 2
    TC2.I .= (t < 0.2T) ? -30 : 0
    RZ.I .= (0.5T < t < 0.6T) ? 10 : 0
    SNN.sim!(; model, duration = dt)
end

fig = Figure(size = (800, 600))
for (n, (name, p)) in enumerate(pairs(P))
    ax = Axis(fig[(n - 1) ÷ 2 + 1, (n - 1) % 2 + 1], title = string(name))
    SNN.vecplot!(ax, p, :v, interval = 0:1:2s)
end
save(joinpath(mktempdir(), "IZ_neurons.png"), fig)
