N = 3
E = SNN.IF(; N = N)
EE = SNN.SpikingSynapse(E, E, :ge; μ = 0.5, p = 0.8)
for n = 1:(N-1)
    SNN.connect!(EE, n, n + 1, 50)
end
E.I[1] = 30

SNN.monitor!(E, [(:v, [1, N])])
SNN.train!([E], [EE]; duration = 100ms)
