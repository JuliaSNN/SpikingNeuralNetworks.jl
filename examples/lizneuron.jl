using PackageCompiler

try
    using NSGAII
catch
    include("install.jl")
end
#using Pkg
#Pkg.clone("https://github.com/JuliaCN/Py2Jl.jl")

#using Py2Jl

include("local_iz_neuron.jl")
using NSGAII
include("../src/units.jl")

using .HHNSGA


function fdecode(x)
   decoded = NSGAII.decode(x, HHNSGA.bc)
   return decoded
end

using Plots
unicodeplots()

function plot_pop(P)
    P = filter(x -> x.rank == 1, P)
    plot(map(x -> x.y[1], P), map(x -> x.y[2], P))#, "bo", markersize = 1)
end
#@bp

#HHNSGA.init_function()
#@bp
@time repop = NSGAII.nsga(4,4,HHNSGA.z,HHNSGA.init_function)#, fplot = plot_pop)
@time best = sort(repop, by = ind -> ind.y[1])[end];
@time worst = sort(repop, by = ind -> ind.y[1])[1];

@time decoded = fdecode(best.pheno)
println("z = $(x1.y)")
py"""
from izhi import IZModel
model = IZModel()
"""
py"model.attrs" = decoded

py"""
rt = test['Rheobase test']
rheo = rt.generate_prediction(model)
"""

vm = py"model.get_membrane_potential()"
Pkg.add("Plotly")
using Plotly


trace1 = [
  "x" => vm,
  "y" => vm.times,
  "mode" => "lines+markers",
  "type" => "scatter"
]
data = [trace1]
response = Plotly.plot(data, ["filename" => "line-scatter", "fileopt" => "overwrite"])
plot_url = response["url"]
