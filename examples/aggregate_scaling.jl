using SpikingNeuralNetworks
using DrWatson, Test
using SpikingNeuralNetworks
SNN.@load_units;
using ProgressBars
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
##
model = compose(E = E, P = P, S = S, A = A)
SNN.monitor!(E, [:fire])
SNN.monitor!(A, [:Y], sr = 100Hz)
SNN.monitor!(S, [:W], sr = 100Hz)

for n in ProgressBar(1:20_000)
    model.syn.S.W .+= randn(size(model.syn.S.W))
    model.syn.S.W = clamp.(model.syn.S.W, 0.5, Inf)
    model.syn.A.param.Y .=
        clamp.(model.syn.A.param.Y .+ randn(size(model.syn.A.param.Y)), 3, 10)
    train!(; model, duration = 10ms)
end
# end
# # vecplot(model.syn.S, :W, interval=0:1000ms)
# p1 = vecplot(model.syn.A, :WT, interval=0:1000ms)
# plot!( ylims=(0,300))
# p2 = vecplot(model.syn.A, :y, interval=0:1000ms)
# plot!( ylims=(0,2))

# A.WT
# param.Y

fr, r = firing_rate(model.pop.E, interval = 0:20ms:15s, τ = 200ms)
y = mean((fr .- param.Y), dims = 1)[1, :]
plot(r, y, xlabel = "Time (s)", ylabel = "Firing Rate (Hz)", label = "E", legend = false)
##
vecplot(model.syn.S, :W, interval = 0:1000ms:200_000ms, neurons = 10:20)

raster(model.pop.E, 80s:140s, every = 20)
