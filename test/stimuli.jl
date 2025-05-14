

E_rate = 200Hz
I_rate = 400Hz

E_BallStick = SNN.BallAndStick(
    300um,
    N = 1,
    NMDA = SNN.EyalNMDA,
    param = SNN.AdExSoma(Vr = -55mV, Vt = -50mV),
)

E_Tripod = SNN.Tripod(
    300um,
    300um,
    N = 1,
    NMDA = SNN.EyalNMDA,
    param = SNN.AdExSoma(Vr = -55mV, Vt = -50mV),
)

##        
stim = Dict{Symbol,Any}()
for (E, d) in zip([E_BallStick, E_Tripod], [:d, :d1])
    SE = SNN.PoissonStimulus(E, :he, d, param = E_rate, μ = 30.0f0, neurons = [1])
    SI = SNN.PoissonStimulus(E, :hi, d, param = I_rate, μ = 15.0f0, neurons = [1])
    my_stim = (SE = SE, SI = SI)
    push!(stim, d => my_stim)
end

model = SNN.merge_models(BallStick = E_BallStick, Tripod = E_Tripod, stim)

SNN.monitor!([model.pop...], [:fire, :h_d, :v_d, :v_s, :v_d1, :v_d2])
SNN.sim!(model = model, duration = 10s, pbar = true, dt = 0.125ms)

p = plot()
q = plot()
SNN.vecplot!(
    p,
    model.pop.BallStick,
    :v_d,
    r = 0.5s:4s,
    sym_id = 1,
    dt = 0.125,
    pop_average = true,
)
SNN.vecplot!(
    p,
    model.pop.BallStick,
    :v_s,
    r = 0.5s:4s,
    sym_id = 1,
    dt = 0.125,
    pop_average = true,
)
plot!(title = "Ball and Stick", ylims = :auto)
SNN.vecplot!(
    q,
    model.pop.Tripod,
    :v_d1,
    r = 0.5s:4s,
    sym_id = 1,
    dt = 0.125,
    pop_average = true,
)
SNN.vecplot!(
    q,
    model.pop.Tripod,
    :v_d2,
    r = 0.5s:4s,
    sym_id = 1,
    dt = 0.125,
    pop_average = true,
)
SNN.vecplot!(
    q,
    model.pop.Tripod,
    :v_s,
    r = 0.5s:4s,
    sym_id = 1,
    dt = 0.125,
    pop_average = true,
)
plot!(title = "Tripod")

plot(p, q, layout = (2, 1), ylims = :auto)
##
