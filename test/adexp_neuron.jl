# -*- coding: utf-8 -*-
adparam = SNN.ADEXParameter(;a = 6.0,
    b = 7.3,
    cm = 803.1,
    v_rest= -63.2,
    tau_m=19.7,
    tau_w=351.0,
    v_thresh=-39.2,
    delta_T=6.4,
    v_spike = -36.6,
    v_reset = -59.2,
    spike_delta = 16.3)

E = SNN.AD(;N = 1, param=adparam)
E.I = [795.57128906]
SNN.monitor(E, [:v,:I])
SNN.sim!([E],[], dt=0.1ms, duration=2000ms)
# # Julia SNN Implementation of AdExp Neuron.
# [Adaptive_exponential_integrate and fire neuron](http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model)
# Dr. Wulfram Gerstner
# Romain Brette, Ecole Normale Supérieure, Paris, France
