using DrWatson
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Plots
using Statistics
using StatsBase
using Distributions

#
pre = Identity(N = 1)
post = IFCurrent(N = 1, param = MongilloParam.Exc)
w = ones(1, 1)
EE = SpikingSynapse(pre, post, :ge, w = w, param = SNN.STPParameter())
SNN.monitor(post, [:v, :ge])
SNN.monitor(EE, [:u, :x], sr = 250Hz)
test = compose(pre = pre, post = post, EE)
SNN.train!(model = test, duration = 5s, dt = 0.125, pbar = true)
for x = 1:6
    pre.g[1] = 1
    SNN.train!(model = test, duration = 30ms, dt = 0.125)
end
SNN.train!(model = test, duration = 200ms, dt = 0.125)
pre.g[1] = 1
SNN.train!(model = test, duration = 10s, dt = 0.125, pbar = true)
p = plot()
SNN.vecplot!(
    p,
    EE,
    :x,
    r = 4s:10s,
    dt = 0.125,
    pop_average = true,
    ribbon = false,
    c = :red,
    label = "",
    ylims = (0, 1),
)
SNN.vecplot!(
    p,
    EE,
    :u,
    r = 4s:10s,
    dt = 0.125,
    pop_average = true,
    ribbon = false,
    c = :blue,
    label = "",
)
plot!(p, ylims = (0, 1))
q = SNN.vecplot(
    post,
    :v,
    r = 4s:10s,
    dt = 0.125,
    pop_average = true,
    ribbon = false,
    c = :blue,
    label = "",
)
plot_single_cell = plot!(p, q, legend = :topleft, layout = (2, 1), xlims = (4.5, 6))
