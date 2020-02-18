using SpikingNeuralNetworks
SNN = SpikingNeuralNetworks
include("../src/units.jl")
include("../src/plot.jl")
using Plots
using UnicodePlots

E = SNN.HH(;N = 1)
E.I = [-0.282nA]
SNN.monitor(E, [:v])
SNN.sim!([E], []; dt = 0.015ms, delay=100ms, stimulus_duration=1000ms, duration = 1300ms)
SNN.vecplot(E, :v) |> display
E = SNN.HH(;N = 1)
E.I = [0.0752nA]
SNN.monitor(E, [:v])
SNN.sim!([E], []; dt = 0.015ms, delay=100ms, stimulus_duration=1000ms, duration = 1300ms)
SNN.vecplot(E, :v) |> display
using PyCall

py"""
from neuronunit import tests
from neuronunit.tests import fi
from very_reduced_sans_lems import VeryReducedModel
from static import StaticModel
from neo import AnalogSignal
import quantities as pq
import julia
julia.setup()
"""
py"""
vm2=[]
vm2.append(0)
vm=AnalogSignal(vm2,units=pq.mV,sampling_rate=1.0*pq.Hz)
"""
sm = py"StaticModel(vm=vm)"
print(sm)
tests = py"tests"
#vrm = py"VeryReducedModel(backend="raw")"
