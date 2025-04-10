using Plots
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
    r = mean(SNN.average_firing_rate(E))
    ys[n] = r
end
plot(xs, ys)

# E = SNN.HH(;N = 10)
# x=20e-6
# E.I .= x
# SNN.monitor!(E, [:v,:fire])
# SNN.sim!([E], []; dt = 0.01ms, duration = 10_000ms)
# bar(SNN.HH_spike_count(E))
# SNN.vecplot(E, :v)|> x->plot!(x,xlims=(9e3,10e4),lw=3)

# # plot(
# # SNN.vecplot(E, :v)|> x->plot!(x,xlims=(9e3,10e4),lw=3),
# # SNN.vecplot(E, :m)|> x->plot!(x,xlims=(9e3,10e4),lw=3),
# # SNN.vecplot(E, :n)|> x->plot!(x,xlims=(9e3,10e4),lw=3),
# # SNN.vecplot(E, :h)|> x->plot!(x,xlims=(9e3,10e4),lw=3),
# # )
# # fieldnames(SNN.HH)
