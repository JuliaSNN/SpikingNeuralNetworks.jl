using Plots
using SpikingNeuralNetworks
SNN.@load_units

S = SNN.Rate(;N = 200)
SS = SNN.PINningSynapse(S, S; σ = 1.5, p = 1.0)
P, C = [S], [SS]

SNN.monitor(SS, [(:g, [1])])

A = 1.3 / 1.5; fr = 1 / 60ms
f(t) = (A /1.0) * sin(1π * fr * t) +  (A / 2.0) * sin(2π * fr * t) +
(A / 6.0) * sin(3π * fr * t) +   (A / 3.0) * sin(4π * fr * t)

ts = 0:0.1ms:1440ms
for (i, t) in enumerate(ts)
    SS.f .= [f(t); SS.g[2:end]]
    SNN.train!(P, C, 0.1ms, t)
end

SNN.vecplot(SS, :g); plot!(f.(ts))
