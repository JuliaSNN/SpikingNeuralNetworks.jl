using Distributed
addprocs(1)
@everywhere include("local_hh_neuron.jl")
using NSGAII
@everywhere include("../src/units.jl")
@everywhere using .HHNSGA
pop = pmap(HHNSGA.init_function,1:3)
#=
println(pop)

export SNN

include("../src/SpikingNeuralNetworks.jl")
include("../src/units.jl")
include("../src/plot.jl")
#SNN = SpikingNeuralNetworks.SNN

=#
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
try
   pop2 = Any[]
   for x in pop
      pop2.append(NSGAII.create_indiv(x, HHNSGA.init_function2, HHNSGA.z))
   end
catch
   @show(pop)
   @show(scores)
end
