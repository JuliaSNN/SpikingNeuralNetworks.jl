
using Plots
using SpikingNeuralNetworks

E = IF(N=100,  param=IFParameterSingleExponential())
P = Poisson(N=100, param=PoissonParameter(3Hz) )
S = SpikingSynapse(P,E, :ge, μ=10, σ=3, p=0.2)
param = AggregateScalingParameter(
        τe =100ms,
        τa = 50ms,
        τ = 10ms,
        Y = rand(5:0.5:10, E.N )
)
A = AggregateScaling(E, [S], param=param)

A.μ
A.fire
model = merge_models(E=E, P=P,S=S)
SNN.monitor(E, [:fire])
SNN.monitor(S, [:W], sr=100Hz)
model.time
for n in 1:100
    model.syn.S.W .+= randn(size(model.syn.S.W)) 
    model.syn.S.W = clamp.(model.syn.S.W, 0.5, Inf)
    train!(;model, duration=10ms)
end
raster(model.pop.E, 0:1s)
# vecplot(model.syn.S, :W, interval=0:1000ms)
