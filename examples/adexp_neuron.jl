# -*- coding: utf-8 -*-
using Plots
using SpikingNeuralNetworks
SNN.@load_units

adparam = SNN.ADEXParameter(;a = 8.2,
    b = 7.3,
    cm = 484.2,
    v_rest = -80.0,
    tau_m = 32.5,
    tau_w = 296.8,
    v_thresh = -40.3,
    delta_T = 7.5,
    v_spike = -36.6,
    v_reset = -80.5,
    spike_delta = 44.3)

E = SNN.AD(;N = 1, param=adparam)
E.I = [3.9]
SNN.monitor(E, [:v,:I])
SNN.sim!([E],[], dt=0.1ms, duration=2000ms)
SNN.vecplot(E, :v)
