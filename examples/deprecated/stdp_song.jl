using Plots
using SpikingNeuralNetworks
SNN.@load_units

inputs = SNN.Poisson(;N = 1000)
inputs.param = SNN.PoissonParameter(;rate = 15Hz)

neurons = SNN.IF2(;N = 1)
neurons.param = SNN.IF2Parameter(;τm = 10ms, τe = 5ms, El = -74mV, Ee = 0mV, Vt = -54mV, Vr = -60mV)

S = SNN.SpikingSynapse(inputs, neurons, :ge; σ = 0.01, p = 1.0)
S.param = SNN.SpikingSynapseParameter(;Wmax = 0.01)

P = [inputs, neurons]; C = [S]

# histogram(S.W / S.param.Wmax; nbins = 20)
# SNN.monitor(S, [(:W, [1, 2])])
@time SNN.train!(P, C; duration = 100second)

scatter(S.W / S.param.Wmax)
histogram(S.W / S.param.Wmax; nbins = 20)
# plot(hcat(SNN.getrecord(S, :W)...)' / S.param.Wmax)
# heatmap(full(sparse(S.I, S.J, S.W / S.param.Wmax)))
