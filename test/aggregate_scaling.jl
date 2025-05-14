using SpikingNeuralNetworks
SNN.@load_units;

E = AdExNeuron(N = 100, param = AdExSynapseParameter(), name = "E")
P = Poisson(N = 100, param = PoissonParameter(3Hz))
S = SpikingSynapse(P, E, :he, μ = 10, σ = 3, p = 0.2)
param = AggregateScalingParameter(
    τe = 100ms,
    τa = 300ms,
    τ = 10ms,
    Wmax = 400pF,
    Y = rand(5:0.5:10, E.N),
)
A = AggregateScaling(E, [S], param = param)

model = merge_models(E = E, P = P, S = S, A = A)
SNN.monitor!(E, [:fire])
SNN.monitor!(A, [:Y], sr = 100Hz)
SNN.monitor!(S, [:W], sr = 100Hz)

for n = 1:10_000
    model.syn.S.W .+= randn(size(model.syn.S.W))
    model.syn.S.W = clamp.(model.syn.S.W, 0.5, Inf)
    model.syn.A.param.Y .=
        clamp.(model.syn.A.param.Y .+ randn(size(model.syn.A.param.Y)), 3, 10)
    train!(; model, duration = 10ms)
end
