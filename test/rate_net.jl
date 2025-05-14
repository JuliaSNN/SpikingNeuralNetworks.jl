G = SNN.Rate(; N = 100)
GG = SNN.RateSynapse(G, G; μ = 1.2, p = 1.0)
SNN.monitor!(G, [(:r, [1, 50, 100])])

SNN.sim!([G], [GG]; duration = 100ms)
