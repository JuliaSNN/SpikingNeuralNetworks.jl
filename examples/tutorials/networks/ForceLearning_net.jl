using SpikingNeuralNetworks
SNN.@load_units

S = SNN.Rate(; N = 200)
SS = SNN.FLSynapse(S, S; μ = 1.5, p = 1.0)
model = SNN.merge_models(; S, SS)

SNN.monitor!(SS, [:f, :z], sr = 1000Hz)

A = 1.3 / 1.5;
fr = 1 / 60ms;
f(t) =
    (A / 1.0) * sin(1π * fr * t) +
    (A / 2.0) * sin(2π * fr * t) +
    (A / 6.0) * sin(3π * fr * t) +
    (A / 3.0) * sin(4π * fr * t)


for t = 0:0.1ms:2440ms
    SS.f = f(t)
    SNN.train!(; model, duration=0.125f0)
end

for t = 2440ms:0.1ms:3000ms
    SS.f = f(t)
    SNN.sim!(; model, duration = 0.125f0)
end

plot([SNN.getrecord(SS, :f) SNN.getrecord(SS, :z)], label = ["f" "z"], lw=3);

SS.records

SS.records[:f]

vline!([2440ms], color = :black, label = "")
