import julia
j = julia.Julia()
j.eval('using Pkg; Pkg.add("SNN")')
x1 = j.include("hh_neuron.jl")
print(x1)
x2 = j.include("hh_net.jl")
