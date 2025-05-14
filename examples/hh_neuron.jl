using SNNPlots
using SpikingNeuralNetworks
SNN.@load_units
using Statistics

gn = 120msiemens * cm^*(-2) * 20_000um^2
gk = 36msiemens * cm^*(-2) * 20_000um^2
gl = 0.03msiemens * cm^*(-2) * 20_000um^2

HHP = SNN.HHParameter(En = 45mV, Ek = -82mV, El = -59.38mV, gn = gn, gk = gk, gl = gl)

xs = range(0, 2, length = 30)
ys = zeros(length(xs))
for n in eachindex(xs)
    E = SNN.HH(; N = 10, param = HHP)
    E.I .= xs[n]
    SNN.monitor!(E, [:v, :fire])
    SNN.sim!([E]; dt = 0.01ms, duration = 5000ms, pbar = true)
    r = mean(SNN.firing_rate(E)[1])
    ys[n] = r
end
plot(xs, ys)
