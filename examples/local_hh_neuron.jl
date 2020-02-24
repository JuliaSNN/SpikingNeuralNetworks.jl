try
    using Plots
    using UnicodePlots
    using OrderedCollections
catch
    import Pkg; Pkg.add("Plots")
    Pkg.add("OrderedCollections") 
    import Pkg; Pkg.add("UnicodePlots")
    using OrderedCollections
end
include("../src/SpikingNeuralNetworks.jl")
include("../src/units.jl")
include("../src/plot.jl")
try
    using UnicodePlots
    using PyCall
    using NSGAIII
catch
    using Pkg
    Pkg.add("PyCall")
    Pkg.add("UnicodePlots")
    #Pkg.add("Conda")
    #using Conda
    #Conda.add("matplotlib")
    Pkg.build("PyCall")
    Pkg.clone("https://github.com/gsoleilhac/NSGAIII.jl")
    using PyCall
    using UnicodePlots
end
SNN = SpikingNeuralNetworks.SNN

py"""
import copy

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
from simple_with_injection import SimpleModel
from neo import AnalogSignal
import quantities as pq
try:
   from julia import Main
except:
   import julia
   julia.install()
   from julia import Main


"""
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
ranges = {k:[v-0.01*v,v+0.01*v] for k,v in copy.copy(JHH).items()}

import pdb
pdb.set_trace()
"""
#
ranges = py"ranges"
H1=[values(ranges)]

current_params = py"rt.params"
print(current_params)
simple.attrs = py"JHH"

#vm = simple.inject_square_current(current_params)
print("gets here")
print(typeof(simple))
py"""
model = SimpleModel()
model2 = SimpleModel()
model3 = SimpleModel()
model.attrs = JHH
model2.attrs = JHH1
model3.attrs = JHH2

import pickle
rheo = rt.generate_prediction(model)
print(rheo)
rheo1 = rt.generate_prediction(model2)
print(rheo1)
rheo2 = rt.generate_prediction(model3)
print(rheo2)

"""
py"""
print("parallel Rheobase is currently broken for python as well need to fix in unit tests")

#rheo_broken = rtest.generate_prediction(model)
#print(rheo_broken)
"""
py"""


def evaluate(tests,attrs):
    model = SimpleModel()
    model.attrs = attrs

    from sciunit.scores.collections import ScoreArray
    from sciunit import TestSuite
    import sciunit
    #from neuronunit.optimisation.data_transport_container import DataTC
    scores_ = []
    from collections import OrderedDict
    tests = OrderedDict({k:v for k,v in tests.items()})
    for key, temp_test in tests.items():
        if key == 'Rheobase test':
            temp_test.score_type = sciunit.scores.ZScore
        if key in ['Injected current AP width test', 'Injected current AP threshold test', 'Injected current AP amplitude test', 'Rheobase test']:
            try:
                temp_test.params['injected_square_current']['amplitude'] = rheo['value']
            except:
                pass
        try:
            score = temp_test.judge(model)
        except:
            score = -np.inf
        if isinstance(score, sciunit.scores.incomplete.InsufficientDataScore):
            score = -np.inf
        else:
            score = score
        scores_.append(score)
    tt = TestSuite(list(tests.values()))
    SA = ScoreArray(tt, scores_)
    return SA

cell_tests = pickle.load(open('multicellular_constraints.p','rb'))
scores = []
for tests in cell_tests.values():
    SA = evaluate(tests,JHH)
    SA = evaluate(tests,JHH1)
    SA = evaluate(tests,JHH2)

    #scores = SA
    print(SA)
    scores.append(SA)

"""
sc = py"scores"
@show(sc)
#println()
popsize = 200
nbGenerations = 100

#Define the number of division along each objective to generate the reference directions.
#Alternatively, you can directly pass the reference directions as a Vector{Vector{Float64}} :
#With two objectives, H = 2 is equivalent to
#H = [[1., 0.], [0.5, 0.5], [0., 1.]]

#define how to generate a random genotype :
#init_function = () -> randperm(N) #e.g : a permutation coding

#define how to evaluate a genotype :
#z(x) = z1(x), z2(x) ... # Must return a Tuple

#nsga(popsize, nbGenerations, init_function, z, H)

#A constraint violation function can be passed with the keyword fCV
#It should return 0. if the solution is feasible and a value > 0 otherwise.

#Mutation probability can be changed with the keyword pMut (default is 5%)

ranges = py"ranges"
H = values(ranges)
#nsga(popsize, nbGen, init_fun, z, H, fCV = CV, pMut = 0.1)


popsize = 200
nbGenerations = 100

##Define the number of division along each objective to generate the reference directions.
#H = 5
#Alternatively, you can directly pass the reference directions as a Vector{Vector{Float64}} :
#With two objectives, H = 2 is equivalent to
#H = [[1., 0.], [0.5, 0.5], [0., 1.]]

#define how to generate a random genotype :
function init(H)
    return genes
end

MU = 2
NGEN = 2

function zzz(in_genes)

    #Convert(fitnesses)
    jfit = tuple(fitnesses)
    return jfit#pop
end
fitnesses = zzz(genes)
X = typeof(init())
fCV=(x)->0
seed = 1.0
P = [indiv(genes[1], zzz, fCV) for _=1:MU-length(seed)]
append!(P, indiv.(convert.(X, seed),z, fCV))
fast_non_dominated_sort!(P)
associate_references!(P, references)
Q = similar(P)
=#
try
   nsga(MU, NGEN, init, zzz, fplot = plot_pop)
catch
   P = [indiv(genes[1], zzz, fCV) for _=1:MU-length(seed)]
   append!(P, indiv.(convert.(X, seed),z, fCV))
   fast_non_dominated_sort!(P)
   associate_references!(P, references)
   Q = similar(P)
end
