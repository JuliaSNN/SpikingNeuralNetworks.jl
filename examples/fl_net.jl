using Plots
using SpikingNeuralNetworks
SNN.@load_units

S = SNN.Rate(;N = 200)
SS = SNN.FLSynapse(S, S; σ = 1.5, p = 1.0)
P, C = [S], [SS]

SNN.monitor(SS, [:f, :z])

A = 1.3 / 1.5; fr = 1 / 60ms
f(t) = (A /1.0) * sin(1π * fr * t) +  (A / 2.0) * sin(2π * fr * t) +
(A / 6.0) * sin(3π * fr * t) +   (A / 3.0) * sin(4π * fr * t)

for t in 0:0.1ms:1440ms
    SS.f = f(t)
    SNN.train!(P, C, 0.1ms)
end

for t in 1440ms:0.1ms:1700ms
    SS.f = f(t)
    SNN.sim!(P, C, 0.1ms)
end

plot([SNN.getrecord(SS, :f) SNN.getrecord(SS, :z)], label = ["f" "z"]);
vline!([1440ms/0.1ms], color = :cyan, label = "")
