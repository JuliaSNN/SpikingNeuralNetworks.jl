using SNNPlots
using SpikingNeuralNetworks
SNN.@load_units

inputs = SNN.Poisson(; N = 1000)
inputs.param = SNN.PoissonParameter(; rate = 1Hz)

neurons = SNN.IF(; N = 1)
neurons.param = SNN.IFParameterSingleExponential(;
    τm = 10ms,
    τe = 5ms,
    El = -74mV,
    E_e = 0mV,
    Vt = -54mV,
    Vr = -60mV,
)

S = SNN.SpikingSynapse(
    inputs,
    neurons,
    :ge;
    μ = 0.01,
    p = 1.0,
    param = SNN.vSTDPParameter(; Wmax = 1.0),
)

model = merge_models(; inputs, neurons, S)
# histogram(S.W / S.param.Wmax; nbins = 20)
# SNN.monitor!(S, [(:W, [1, 2])])
@time SNN.train!(; model, duration = 100second)

scatter(S.W / S.param.Wmax)
histogram(S.W / S.param.Wmax; nbins = 20)
# plot(hcat(SNN.getrecord(S, :W)...)' / S.param.Wmax)
# heatmap(full(sparse(S.I, S.J, S.W / S.param.Wmax)))
