using Pkg
Pkg.update()

try
    using UnicodePlots
    using PyCall
    using NSGAII
    using OrderedCollections
    #using Plotly
catch
    #import Pkg
    #Pkg.add("Plotly")
    import Pkg; Pkg.add("Plots")
    import Pkg; Pkg.add("UnicodePlots")
    Pkg.add("OrderedCollections")

    #Pkg.add("PackageCompiler.jl")
    #Pkg.clone("https://github.com/JuliaCN/Py2Jl.jl")
    Pkg.clone("https://github.com/gsoleilhac/NSGAII.jl")
    Pkg.add("ProgressMeter")
    Pkg.add("UnicodePlots")
    Pkg.add("Plots")
    Pkg.add("OrderedCollections")
    Pkg.add("PyCall")
    Pkg.build("PyCall")
    #Pkg.clone("https://github.com/gsoleilhac/NSGAIII.jl")
    #Pkg.clone("https://github.com/JuliaCN/Py2Jl.jl")

end
Pkg.update()
using PyCall
using UnicodePlots

include("src/SpikingNeuralNetworks.jl")
include("src/units.jl")
include("src/plot.jl")
using UnicodePlots
using PyCall
using NSGAII
SNN = SpikingNeuralNetworks.SNN
#=
py"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.get_backend()
import sys
import os
sys.path.append(os.getcwd())
"""
py"""
from neuronunit import tests
from neuronunit.tests import fi
from examples.simple_with_injection import SimpleModel
from neo import AnalogSignal
import quantities as pq
try:
   from julia import Main
except:
   import julia
   julia.install()
   from julia import Main


"""
#=
simple = py"SimpleModel()"

py"""
import pickle
from neuronunit.tests.fi import RheobaseTestP

cell_tests = pickle.load(open('multicellular_constraints.p','rb'))
for test in cell_tests.values():
    if "Rheobase test" in test.keys():
        temp_test = {k:v for k,v in test.items()}
        break
rt = temp_test["Rheobase test"]

rtest = RheobaseTestP(observation=rt.observation)
#                        name='RheobaseTest')
JHH = {
'Vr': -68.9346,
'Cm': 0.0002,
'gl': 1.0 * 1e-5,
'El': -65.0,
'EK': -90.0,
'ENa': 50.0,
'gNa': 0.02,
'gK': 0.006,
'Vt': -63.0
}
JHH1 = {k:(v-0.01*v) for k,v in JHH.items()}
JHH2 = {k:(v+0.01*v) for k,v in JHH.items()}
ranges = {k:[v-0.01*v,v+0.01*v] for k,v in JHH.items()}


"""
#
using OrderedCollections
ranges = OrderedDict(py"ranges")
H1=[values(ranges)]

current_params = py"rt.params"a
print(current_params)
simple.attrs = py"JHH"
=#
=#
