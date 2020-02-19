include("../src/SpikingNeuralNetworks.jl")
include("../src/units.jl")
include("../src/plot.jl")
try
    using UnicodePlots
    using PyCall
catch
    using Pkg
    Pkg.add("PyCall")
    Pkg.add("UnicodePlots")
    Pkg.add("Conda")
    using Conda
    Conda.add("matplotlib")
    Pkg.build("PyCall")
    
    using PyCall
    using UnicodePlots
end
SNN = SpikingNeuralNetworks.SNN

py"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.get_backend()
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


"""

current_params = py"rt.params"
print(current_params)
simple.attrs = py"JHH"

vm = simple.inject_square_current(current_params)
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


def evaluate(tests):
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
    SA = evaluate(tests)
    #scores = SA
    print(SA)
    scores.append(SA)

"""
sc = py"scores"
@show(sc)
#println()
