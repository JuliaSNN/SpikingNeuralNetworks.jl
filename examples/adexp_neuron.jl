# -*- coding: utf-8 -*-
using Plots
using SpikingNeuralNetworks
SNN.@load_units

# +
adparam = SNN.ADEXParameter(;a =7.78,
    b = 5.85,
    cm = 577,
    v_rest = -96,
    tau_m = 31.5,
    tau_w = 333.8,
    v_thresh = -25.47,
    delta_T = 8.0,
    v_spike = -59.7,
    v_reset = -75.9,
    spike_delta = 22.0)


#=
{'cm': 577.424113896445,
 'v_spike': -59.704166972923005,
 'v_reset': -75.30349181562713,
 'v_rest': -96.9407099405355,
 'tau_m': 31.12267480749466,
 'a': 7.7881055214232635,
 'b': 5.859705728754745,
 'delta_T': 8.074015033913897,
 'tau_w': 333.1571313567891,
 'v_thresh': -25.472280967665,
 'spike_delta': 22.070735858044355}
=#
# -

E = SNN.AD(;N = 1, param=adparam)
E.I = [3.86]
SNN.monitor(E, [:v,:I])
SNN.sim!([E],[], dt=0.1ms, duration=3000ms)
SNN.vecplot(E, :v)
