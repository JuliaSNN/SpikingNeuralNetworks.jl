

try
    using UnicodePlots
    using PyCall
    using NSGAII
    using OrderedCollections
    using Plotly
catch
    import Pkg
    Pkg.add("Plotly")
    import Pkg; Pkg.add("Plots")
    import Pkg; Pkg.add("UnicodePlots")
    Pkg.add("OrderedCollections")

    Pkg.clone("https://github.com/gsoleilhac/NSGAII.jl")
    Pkg.add("ProgressMeter")
    Pkg.add("UnicodePlots")
    Pkg.add("Plots")
    Pkg.add("OrderedCollections")
    Pkg.add("PyCall")
    Pkg.build("PyCall")
    #Pkg.clone("https://github.com/gsoleilhac/NSGAIII.jl")
    #Pkg.clone("https://github.com/JuliaCN/Py2Jl.jl")
    using PyCall
    using UnicodePlots
end
using UnicodePlots
using PyCall
using NSGAII

include("../src/SpikingNeuralNetworks.jl")
include("../src/units.jl")
include("../src/plot.jl")

SNN = SpikingNeuralNetworks.SNN

py"""
try:
   from julia import Main
except:
   import julia
   julia.install()
   from julia import Main
"""
