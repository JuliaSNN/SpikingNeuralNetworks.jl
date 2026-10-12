using SpikingNeuralNetworks
using CairoMakie
SNN.@load_units

# Two PINning networks, an input layer ("ear") and a recurrent layer ("ppc"); unit 1 of ppc is
# trained to follow a target signal
ear = SNN.Rate(; N = 2)
ear.r .= 0
ppc = SNN.Rate(; N = 200)
ear_ppc = SNN.PINningSynapse(ear, ear; μ = 1.5, p = 1.0)
ppc_ppc = SNN.PINningSynapse(ppc, ppc; μ = 1.5, p = 1.0)
model = SNN.compose(; ear, ppc, ear_ppc, ppc_ppc, silent = true)
SNN.monitor!(ppc_ppc, [(:g, [1])], sr = 1kHz)

A = 1.3 / 1.5
fr = 1 / 60ms
f(t) = (A / 1.0) * sin(1π * fr * t) + (A / 2.0) * sin(2π * fr * t) +
       (A / 6.0) * sin(3π * fr * t) + (A / 3.0) * sin(4π * fr * t)

dt = 0.125ms
ts = 0:dt:720ms
for t in ts
    ppc_ppc.f .= [f(t); ppc_ppc.g[2:end]]
    SNN.train!(; model, duration = dt, dt)
end

g, r = SNN.record(ppc_ppc, :g, range = true)
fig = Figure(size = (600, 300))
ax = Axis(fig[1, 1], xlabel = "Time (ms)", ylabel = "Input of unit 1")
lines!(ax, r, g(1, r), label = "trained")
lines!(ax, ts, f.(ts), label = "target")
axislegend(ax)
save(joinpath(mktempdir(), "ppc.png"), fig)
