# f-I curve of the Hodgkin-Huxley neuron.
using SpikingNeuralNetworks
using CairoMakie, Statistics
SNN.@load_units

gn = 120msiemens * cm^(-2) * 20_000um^2
gk = 36msiemens * cm^(-2) * 20_000um^2
gl = 0.03msiemens * cm^(-2) * 20_000um^2

HHP = SNN.HHParameter(En = 45mV, Ek = -82mV, El = -59.38mV, gn = gn, gk = gk, gl = gl)

xs = range(0, 2, length = 15)
ys = zeros(length(xs))
for n in eachindex(xs)
    E = SNN.HH(; N = 10, param = HHP)
    E.I .= xs[n]
    SNN.monitor!(E, [:fire])
    SNN.sim!([E]; dt = 0.01ms, duration = 1000ms)
    ys[n] = mean(SNN.firing_rate(E; interval = 0:10ms:1s, time_average = true)[1])
end
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "Input current", ylabel = "Firing rate (Hz)", title = "HH f-I curve")
lines!(ax, xs, ys)
save(joinpath(mktempdir(), "HH_fI.png"), fig)
