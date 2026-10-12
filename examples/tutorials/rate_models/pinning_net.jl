using SpikingNeuralNetworks
using CairoMakie
SNN.@load_units

# PINning (Rajan et al. 2016): the input of unit 1 is trained to follow a target signal
S = SNN.Rate(; N = 200)
SS = SNN.PINningSynapse(S, S; μ = 1.5, p = 1.0)
model = SNN.compose(; S, SS, silent = true)
SNN.monitor!(SS, [(:g, [1])], sr = 1kHz)

A = 1.3 / 1.5
fr = 1 / 60ms
f(t) = (A / 1.0) * sin(1π * fr * t) + (A / 2.0) * sin(2π * fr * t) +
       (A / 6.0) * sin(3π * fr * t) + (A / 3.0) * sin(4π * fr * t)

dt = 0.125ms
ts = 0:dt:720ms
for t in ts
    SS.f .= [f(t); SS.g[2:end]]       # unit 1 follows f, the others their own input
    SNN.train!(; model, duration = dt, dt)
end

g, r = SNN.record(SS, :g, range = true)
fig = Figure(size = (600, 300))
ax = Axis(fig[1, 1], xlabel = "Time (ms)", ylabel = "Input of unit 1")
lines!(ax, r, g(1, r), label = "trained")
lines!(ax, ts, f.(ts), label = "target")
axislegend(ax)
save(joinpath(mktempdir(), "pinning_net.png"), fig)
