# -*- coding: utf-8 -*-
using Plots
using SpikingNeuralNetworks
SNN.@load_units

adparam = SNN.ADEXParameter(;a =7.78,
    b = 5.85,
    cm = 577,
    v_rest = -97,
    tau_m = 31.5,
    tau_w = 333.8,
    v_thresh = -59.7,
    delta_T = 8.0,
    v_spike = -59.7,
    v_reset = -75.9,
    spike_delta = 22.0)

E = SNN.AD(;N = 1, param=adparam)
E.I = [700.86]
SNN.monitor(E, [:v,:I])
SNN.sim!([E],[], dt=0.1ms, duration=3000ms)
SNN.vecplot(E, :v)


