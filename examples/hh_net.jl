using Plots, SNN

E = SNN.HH(;N = 3200)
I = SNN.HH(;N = 800)
EE = SNN.SpikingSynapse(E, E, :ge; σ = 6nS, p = 0.02)
EI = SNN.SpikingSynapse(E, I, :ge; σ = 6nS, p = 0.02)
IE = SNN.SpikingSynapse(I, E, :gi; σ = 67nS, p = 0.02)
II = SNN.SpikingSynapse(I, I, :gi; σ = 67nS, p = 0.02)
P = [E, I]
C = [EE, EI, IE, II]

SNN.monitor(E, [(:v, [1, 10, 100])])
SNN.sim!(P, C; dt = 0.01ms, duration = 100ms)
SNN.vecplot(E, :v) |> display


using Plots, SNN

RS = SNN.IZ(;N = 1, param = SNN.IZParameter(;a = 0.02, b = 0.2, c = -65, d = 8))
IB = SNN.IZ(;N = 1, param = SNN.IZParameter(;a = 0.02, b = 0.2, c = -55, d = 4))
CH = SNN.IZ(;N = 1, param = SNN.IZParameter(;a = 0.02, b = 0.2, c = -50, d = 2))
FS = SNN.IZ(;N = 1, param = SNN.IZParameter(;a = 0.1, b = 0.2, c = -65, d = 2))
TC1 = SNN.IZ(;N = 1, param = SNN.IZParameter(;a = 0.02, b = 0.25, c = -65, d = 0.05))
TC2 = SNN.IZ(;N = 1, param = SNN.IZParameter(;a = 0.02, b = 0.25, c = -65, d = 0.05))
RZ = SNN.IZ(;N = 1, param = SNN.IZParameter(;a = 0.1, b = 0.26, c = -65, d = 2))
LTS = SNN.IZ(;N = 1, param = SNN.IZParameter(;a = 0.1, b = 0.25, c = -65, d = 2))
P = [RS, IB, CH, FS, TC1, TC2, RZ, LTS]

SNN.monitor(P, [:v])
T = 2second
for t = 0:T
    for p in [RS, IB, CH, FS, LTS]
        p.I = [10]
    end
    TC1.I = [(t < 0.2T) ? 0mV : 2mV]
    TC2.I = [(t < 0.2T) ? -30mV : 0mV]
    RZ.I =  [(0.5T < t < 0.6T) ? 10mV : 0mV]
    SNN.sim!(P, [], 0.1ms)
end
SNN.vecplot(P, :v) |> display
