# Out-degree distributions produced by the connectivity rules of `sparse_matrix`.
using SpikingNeuralNetworks
using CairoMakie
SNN.@load_units

fig = Figure(size = (450, 350))
ax = Axis(fig[1, 1], xlabel = "Out degree", ylabel = "Density")
E = SNN.IF(N = 4000, param = SNN.IFParameter(), name = "E")
for rule in [:Fixed, :Bernoulli, :PowerLaw]
    conn = (p = 0.05, μ = 2nS, rule = rule, γ = 1, kmin = 100)
    E_to_E = SNN.SpikingSynapse(E, E, :ge; conn, name = "E_to_E")
    outdeg = [length(SNN.postsynaptic(E_to_E, j)) for j in 1:E.N]
    hist!(ax, outdeg, bins = 0:10:500, label = String(rule), normalization = :pdf)
end
axislegend(ax, "Connection rule")
save(joinpath(mktempdir(), "out_degree.png"), fig)
fig
