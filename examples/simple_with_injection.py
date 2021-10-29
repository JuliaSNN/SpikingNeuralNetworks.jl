import pickle
from neo.core import AnalogSignal
import sciunit
import sciunit.capabilities as scap
import neuronunit.capabilities as cap
import neuronunit.capabilities.spike_functions as sf
import quantities as pq
from elephant.spike_train_generation import threshold_detection
def Id(t,delay,duration,tmax,amplitude):
    if 0.0 < t < delay:
        return 0.0
    elif delay < t < delay+duration:

        return amplitude#(100.0)

    elif delay+duration < t < tmax:
        return 0.0
    else:
        return 0.0

import julia
jl = julia.Julia()
from julia import Main

#Main.eval('include("../src/SpikingNeuralNetworks.jl")')
#Main.eval('include("../src/units.jl")')
#Main.eval('include("../src/plot.jl")')
#Main.eval('SNN = SpikingNeuralNetworks.SNN')


# from sciunit.models.runnable import RunnableModel
# probably runnable is the julia crasher
class SimpleModel(sciunit.Model,
                  cap.ReceivesSquareCurrent,
                  cap.ProducesActionPotentials,
                  cap.ProducesMembranePotential,
                  scap.Runnable):
    """A model which produces a frozen membrane potential waveform."""

    def __init__(self,attrs={},backend="JULIA_HH"):
        """Create an instace of a model that produces a static waveform.
        """
        self.attrs = attrs
        self.vm = None
        #self.Main = Main


        self._backend = self
        self.backend = backend

    def set_run_params(self,t_stop=None):
        #if 'tmax' in self.params.keys():
        #    self.t_stop=self.params['tmax']
        #else:
        self.t_stop = t_stop
    def get_membrane_potential(self, **kwargs):
        """Return the Vm passed into the class constructor."""
        return self.vm
    def set_stop_time(self,tmax=1300*pq.ms):
        self.tmax = tmax
    def set_attrs(self, attrs):
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
        if not len(attrs):
            attrs = JHH
        #Main = self.Main

        Main.eval("SNN.HH(;N = 1)")
        Main.attrs = attrs
        Main.eval('param = SNN.HHParameter(;El =  attrs["El"], Ek = attrs["EK"], En = attrs["ENa"], gl = attrs["gl"], gk = attrs["gK"], gn = attrs["gNa"])')#', N = attrs["N"])')
        self.attrs.update(attrs)


        Main.N_ = attrs["N"]

        Main.eval('E2 = SNN.HH(;N = 1)')
        Main.eval('E2.param = param')
        Main.eval("N = Int32(1)")
        #Main.eval("@show(N)")
        Main.eval('E2.v = ones(N).*attrs["Vr"]')
        Main.eval("E2.m = zeros(N)")
        Main.eval("E2.n = zeros(N)")
        Main.eval("E2.h = ones(N)")
        #Main.eval('include("../src/units.jl")')
        Main.eval("E2.ge = (ones(N) .+ 4)")# .* 10nS")
        Main.eval("E2.gi = (ones(N) .+ 20)")# .* 10nS")
        Main.eval("E2.ge = E2.ge.*10nS")
        Main.eval("E2.gi = E2.gi.*10nS")

        Main.eval("E2.fire = zeros(Bool, N)")
        Main.eval('E2.records = Dict()')




    def inject_square_current(self, current):#, section = None, debug=False):
        """Inputs: current : a dictionary with exactly three items, whose keys are: 'amplitude', 'delay', 'duration'
        Example: current = {'amplitude':float*pq.pA, 'delay':float*pq.ms, 'duration':float*pq.ms}}
        where \'pq\' is a physical unit representation, implemented by casting float values to the quanitities \'type\'.
        Description: A parameterized means of applying current injection into defined
        Currently only single section neuronal models are supported, the neurite section is understood to be simply the soma.

        """
        import numpy as np
        if 'injected_square_current' in current.keys():
            c = current['injected_square_current']
        else:
            c = current
        duration = float(c['duration'])#.rescale('ms'))
        delay = float(c['delay'])#.rescale('ms'))
        amp = float(c['amplitude'])#.rescale('uA')
        tmax =1.3#00
        tmin = 0.0
        DT = 0.025
        T = np.linspace(tmin, tmax, int(tmax/DT))
        Iext_ = []
        for t in T:
            Iext_.append(Id(t,delay,duration,tmax,amp))
        self.attrs['N'] = len(Iext_)
        self.set_attrs(self.attrs)
        Main.eval('pA = 0.001nA')
        Main.eval("SNN.monitor(E2, [:v])")
        Main.dur = current["duration"]
        Main.current = current
        Main.delay = float(current["delay"])
        Main.temp_current = float(amp)
        Main.eval("E2.I = [deepcopy(temp_current)*pA]")
        #Main.eval("@show(E2.I)")
        Main.eval('SNN.sim!([E2], []; dt ='+str(DT)+'*ms, delay=delay,stimulus_duration=1000,simulation_duration = 1300)')
        #Main.eval('SNN.sim!([E2], []; dt = 0.015*ms, delay=current["delay"]*ms,stimulus_duration=1000*ms,simulation_duration = 1300*ms)')

        Main.eval("v = SNN.getrecord(E2, :v)")
        v = Main.v
        #Main.eval('SNN.vecplot(E2, :v) |> display')
        self.vM = AnalogSignal(v,units = pq.mV,sampling_period = DT * pq.ms)
        return self.vM
    def get_membrane_potential(self):
        return self.vM

    def get_spike_count(self):
        thresh = threshold_detection(self.vM,threshold=0*pq.mV)
        return len(thresh)
    def get_backend(self):
        return self
    def get_spike_train(self):
        thresh = threshold_detection(self.vM)
        return thresh
    def _backend_run(self):
        results['vm'] = self.vM.magnitude
        results['t'] = self.vM.times

        return results


    def get_APs(self, **run_params):
        """Return the APs, if any, contained in the static waveform."""
        vm = self.get_membrane_potential(**run_params)
        waveforms = sf.get_spike_waveforms(vm)
        return waveforms
