# Morris-Lecar neuron driven by a noisy current, and its phase plane (vector field and nullclines).
using SpikingNeuralNetworks
using CairoMakie, Distributions
SNN.@load_units
const SNNModels = SNN.SNNModels

E = SNN.MorrisLecar(N = 1)
I = SNN.CurrentStimulus(E, :I, param = SNN.CurrentNoise(E.N; I_dist = Normal(100pA, 200pA)))
SNN.monitor!(E, [:v, :w])
model = SNN.compose(; E, I, silent = true)
SNN.sim!(model, 1s)

fig = Figure(size = (900, 700))
SNN.vecplot!(Axis(fig[1, 1], ylabel = "v (mV)"), E, :v)
SNN.vecplot!(Axis(fig[2, 1], ylabel = "w", xlabel = "Time (s)"), E, :w)

# phase plane at I = 100 pA
ws = Float32.(0:0.15:1)
vs = Float32.(-25:8:45)
Iext = Float32(100pA)
dv = [SNNModels.MorrisLecar_dv(v, w, Iext, E.param) / 20 for v in vs, w in ws]
dw = [SNNModels.MorrisLecar_dw(v, w, E.param) for v in vs, w in ws]
vfine = Float32.(-25:0.5:45)
ax = Axis(fig[1:2, 2], xlabel = "v (mV)", ylabel = "w", title = "Phase plane, I = 100 pA")
arrows2d!(ax, repeat(vs, outer = length(ws)), repeat(ws, inner = length(vs)), vec(dv), vec(dw), color = :black)
lines!(ax, vfine, [SNNModels.MorrisLecar_w_nullcline(v, E.param) for v in vfine], label = "w nullcline", color = :blue)
lines!(ax, vfine, [SNNModels.MorrisLecar_v_nullcline(v, Iext, E.param) for v in vfine], label = "v nullcline", color = :red)
ylims!(ax, -0.1, 1.1)
axislegend(ax, position = :lt)
save(joinpath(mktempdir(), "morris_lecar.png"), fig)
