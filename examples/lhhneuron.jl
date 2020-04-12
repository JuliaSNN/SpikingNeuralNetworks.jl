#using Distributed
#addprocs(1)
#@everywhere
include("local_hh_neuron.jl")
using NSGAII
#@everywhere
include("../src/units.jl")
#@everywhere
using .HHNSGA
#=
seed = pop[1]
bincoded = NSGAII.encode(seed, HHNSGA.bc)
decoded = NSGAII.decode(bincoded, HHNSGA.bc)
scores = map(HHNSGA.z,pop)
@show(scores)

pop = []
for i in 1:3
   append!(pop,[HHNSGA.init_function()])
end
=#


function fdecode(x)
   decoded = NSGAII.decode(x, HHNSGA.bc)

   #y = [ convert(Int64,i) for i in Int(x) ]
   return decoded
end
#@test
#@show decoded
#@show seed
#assert decoded .≈ seed

#lower_list = [v[0] for k,v in ranges.items()]
#upper_list = [v[1] for k,v in ranges.items()]

#NSGAII.create_indiv(pop[1], fdecode, HHNSGA.z, HHNSGA.init_function2)


#@show(pop)
#pop = map(HHNSGA.init_function,1:3)
#NSGAII.create_indiv(pop[1],HHNSGA.z)
#=
println(pop)

export SNN

include("../src/SpikingNeuralNetworks.jl")
include("../src/units.jl")
include("../src/plot.jl")
#SNN = SpikingNeuralNetworks.SNN

=#

#repop = NSGAII.nsga(2,2,HHNSGA.z,HHNSGA.init_function())
#append!(pop,
using Plots
unicodeplots()

function plot_pop(P)
    P = filter(x -> x.rank == 1, P)
    plot(map(x -> x.y[1], P), map(x -> x.y[2], P))#, "bo", markersize = 1)
    #show()
    #sleep(0.1)
end

repop = NSGAII.nsga(10,10,HHNSGA.z,HHNSGA.init_function, fplot = plot_pop)
x1 = sort(repop, by = ind -> ind.y[1])[end];
decoded = fdecode(x1.pheno)
println("z = $(x1.y)")
#=
NSGAII.indiv(copy(pop[1]),HHNSGA.z)

println(scores)
non_dom = NSGAIII.fast_non_dominated_sort!(scores)[1]
pop = setdiff(pop, non_dom)
HHNSGA.plot_pop(scores)
@show(pop)
@show(scores)
try
   pop2 = Any[]
   for x in pop
      append!(pop,NSGAII.create_indiv(x, HHNSGA.init_function2, HHNSGA.z))
   end
catch
   @show(pop)
   @show(scores)
end
=#
