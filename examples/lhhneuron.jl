using Distributed
addprocs(4)
@everywhere include("local_hh_neuron.jl")
using NSGAIII
@everywhere include("../src/units.jl")
@everywhere using .HHNSGA
pop = pmap(HHNSGA.init_function,1:3)
println(pop)

scores = pmap(HHNSGA.z,pop)
#=
NSGAIII.indiv(copy(pop[1]),HHNSGA.z)

println(scores)
non_dom = NSGAIII.fast_non_dominated_sort!(scores)[1]
pop = setdiff(pop, non_dom)
HHNSGA.plot_pop(scores)
=#
@show(pop)
@show(scores)
