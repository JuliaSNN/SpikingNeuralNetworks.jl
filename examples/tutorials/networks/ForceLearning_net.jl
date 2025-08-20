

using SpikingNeuralNetworks
SNN.@load_units
using Plots

S = SNN.Rate(; N = 200)
SS = SNN.FLSynapse(S, S; μ = 1.5, p = 1.0)
model = SNN.compose(; S, SS)

SNN.monitor!(SS, [:f, :z], sr = 1000Hz)

A = 1.3 / 1.5;
fr = 1 / 60ms;
f(t) =
    (A / 1.0) * sin(1π * fr * t) +
    (A / 2.0) * sin(2π * fr * t) +
    (A / 6.0) * sin(3π * fr * t) +
(A / 3.0) * sin(4π * fr * t)


for t = 0:0.125ms:2440ms
    SS.f = f(t)
    SNN.train!(; model, duration = 0.125f0)
end

for t = 2440ms:0.125ms:3500ms
    SS.f = f(t)
    SNN.sim!(; model, duration = 0.125f0)
end

#
p = plot([SNN.getrecord(SS, :f) SNN.getrecord(SS, :z)], label = ["Signal" "Prediction"], lw = 3);
plot!(p, xlabel = "Time (ms)", ylabel = "Signal", title = "Force Learning Network",
      legend = :outerright, size = (800, 400), grid = false, ylims = (-1.8, 1.5), xlims =(2000, 3000), 
      fg_legend=:transparent, legendfontsize=14)
annotate!(p, [(2240ms, -1.5, "Training phase")], textsize = 10, color = :black)
annotate!(p, [(2650ms, -1.5, "Testing phase")], textsize = 10, color = :black)

SS.records

SS.records[:f]

vline!([2440ms], color = :black, label = "", lw=3)

savefig(p, joinpath(SNN.DOCS_ASSETS_PATH, "examples", "force_learning.png"))

p