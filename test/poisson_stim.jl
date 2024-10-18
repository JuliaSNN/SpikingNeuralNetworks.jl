E = SNN.IF(; N = 100)
typeof(E) <: SNN.AbstractPopulation


r(t)= SNN.get_time(t)Hz
S = SNN.PoissonStimulus(E, :ge, r, p_post=0.2f0, N_pre=50)
SNN.monitor(E, [:ge])
SNN.sim!([E],[SNN.EmptySynapse()], [S]; duration = 1000ms)





