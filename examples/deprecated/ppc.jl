using Plots
using SpikingNeuralNetworks
SNN.@load_units

ear = SNN.Rate(;N = 2)
ear.r = zeros(ear.N) # exp(- (0:ear.N-1).^2 / 10^2)
ppc = SNN.Rate(;N = 200)
ear_ppc = SNN.PINningSynapse(ear, ear; σ = 1.5, p = 1.0)
ppc_ppc = SNN.PINningSynapse(ppc, ppc; σ = 1.5, p = 1.0)
P, C = [ppc], [ear_ppc, ppc_ppc]

SNN.monitor(ppc_ppc, [(:g, [1])])

A = 1.3 / 1.5; fr = 1 / 60ms
f(t) = (A /1.0) * sin(1π * fr * t) +  (A / 2.0) * sin(2π * fr * t) +
(A / 6.0) * sin(3π * fr * t) +   (A / 3.0) * sin(4π * fr * t)

ts = 0:0.1ms:1440ms
for (i, t) in enumerate(ts)
    ppc_ppc.f .= [f(t); ppc_ppc.g[2:end]]
    SNN.train!(P, C, 0.1ms, t)
end

SNN.vecplot(ppc_ppc, :g); plot!(f.(ts))
