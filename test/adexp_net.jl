Ne = 800;
Ni = 200
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

E = SNN.AD(;N = Ne, param=adparam)
I = SNN.AD(;N = Ni, param=adparam)

EE = SNN.SpikingSynapse(E, E, :v; σ = 0.5,  p = 0.8)
EI = SNN.SpikingSynapse(E, I, :v; σ = 0.5,  p = 0.8)
IE = SNN.SpikingSynapse(I, E, :v; σ = -1.0, p = 0.8)
II = SNN.SpikingSynapse(I, I, :v; σ = -1.0, p = 0.8)
P = [E, I]
C = [EE, EI, IE, II]

SNN.monitor([E, I], [:fire])
for t = 1:1000
    E.I = 795.6randn(Ne)
    I.I .= 200randn(Ni)
    SNN.sim!(P, C, 1ms)
end
