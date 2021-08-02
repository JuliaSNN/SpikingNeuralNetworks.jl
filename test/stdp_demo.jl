N = 100
E1 = SNN.IF(;N = N)
E2 = SNN.IF(;N = N)
EE = SNN.SpikingSynapse(E1, E2, :ge)
for n = 1:E1.N SNN.connect!(EE, n, n) end
SNN.monitor([E1, E2], [:fire])
SNN.monitor(EE, [:W])

for t = 1:N
    E1.v[t] = 100
    E2.v[N - t + 1] = 100
    SNN.train!([E1, E2], [EE], 0.5ms, (t - 1) * 0.5ms)
end
ΔW = EE.records[:W][end]
