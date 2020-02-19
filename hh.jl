using PyCall
@pyimport numpy
#] add https://github.com/AStupidBear/SpikingNeuralNetworks.jl
using Pkg

# Pkg.add("https://github.com/AStupidBear/SpikingNeuralNetworks.jl")


using GR


import SpikingNeuralNetworks



using Unitful

using Plots, SpikingNeuralNetworks
SNN = SpikingNeuralNetworks
E = SNN.HH(;N = 1)
E.I = [0.003]

SNN.monitor(E, [:v])
SNN.sim!([E], []; dt = 0.01, duration = 102)


SNN.vecplot(E, :v) |> display



using NSGAII, vOptSpecific, vOptGeneric, GLPK, GLPKMathProgInterface, PyPlot
m = vModel(solver = GLPKSolverMIP())
id = load2UKP("2KP500-1A.DAT")

p1, p2, w, c = id.P1, id.P2, id.W, id.C

@variable(m, x[1:length(p1)], Bin)
@addobjective(m, Max, dot(x, p1))
@addobjective(m, Max, dot(x, p2))
@constraint(m, dot(x, w) <= c)

function plot_pop(P, titre)
  clf()
  ax = gca()
  ax[:set_xlim]([15800, 20570])
  ax[:set_ylim]([15630, 20877])
  p = plot(map(x -> x.y[1], P), map(x -> x.y[2], P), "bo", markersize=1, label="nsga")
  title(titre)
  !isinteractive() && show()
  sleep(0.1)
end

nsga(100, 5000, m, fplot = p->plot_pop(p, "without seed"), plotevery=500)

solve(m, method=:dichotomy)

Y_N = getY_N(m)
seed = [getvalue(x, 1), getvalue(x, length(Y_N)), getvalue(x, length(Y_N)÷2)]
nsga(100, 5000, m, fplot = p->plot_pop(p, "with seed"), seed=seed, plotevery=500)

f1 = map(y -> y[1], Y_N)
f2 = map(y -> y[2], Y_N)
xlabel("z1") ; ylabel("z2")
p = plot(f1,f2,"kx", markersize = "2", label="exact")
legend() ; display(p)
