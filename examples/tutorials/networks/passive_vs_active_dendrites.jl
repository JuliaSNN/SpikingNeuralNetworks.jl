# Two recurrent populations of ball-and-stick neurons receiving the same Poisson input on the
# dendrite: one with AMPA-only (passive) dendrites, one with AMPA and NMDA (active) dendrites.
using SpikingNeuralNetworks
using CairoMakie, Statistics
SNN.@load_units

gaba = SNN.Receptor(name = "GABAa", E_rev = -70.0, τr = 0.1, τd = 15.0, g0 = 0.38)
# passive dendrite: AMPA only (larger conductance)
passive_syn = SNN.ReceptorSynapse(
    syn = SNN.ReceptorArray([SNN.Receptor(name = "AMPA", E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 2.73), gaba]),
    NMDA = SNN.EyalNMDA, glu_receptors = [1], gaba_receptors = [2])
# active dendrite: AMPA and voltage-dependent NMDA
active_syn = SNN.ReceptorSynapse(
    syn = SNN.ReceptorArray([SNN.Receptor(name = "AMPA", E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.73),
                             SNN.Receptor(name = "NMDA", E_rev = 0.0, τr = 8.0, τd = 35.0, g0 = 1.31, nmda = 1.0),
                             gaba]),
    NMDA = SNN.EyalNMDA, glu_receptors = [1, 2], gaba_receptors = [3])

param = SNN.BallAndStickParameter(ds = [150um])
E_active = SNN.BallAndStick(N = 200; name = "Active", param, dend_syn = active_syn)
E_passive = SNN.BallAndStick(N = 200; name = "Passive", param, dend_syn = passive_syn)

input_param = SNN.PoissonLayer(N = 100, rate = 20Hz)
projection = (p = 0.5, μ = 2nS)
I_active = SNN.Stimulus(input_param, E_active, :glu, :d, conn = projection)
I_passive = SNN.Stimulus(input_param, E_passive, :glu, :d, conn = projection)
EE_passive = SNN.SpikingSynapse(E_passive, E_passive, :glu, :d; conn = (μ = 2, p = 0.01))
EE_active = SNN.SpikingSynapse(E_active, E_active, :glu, :d; conn = (μ = 2, p = 0.01))

model = SNN.compose(; E_active, E_passive, I_active, I_passive, EE_active, EE_passive)
SNN.monitor!(model.pop, [:fire])
SNN.monitor!(model.pop, [(:v_d, 1:1), (:v_s, 1:1)], sr = 2kHz)
SNN.sim!(model, duration = 1second)

rates, _, names = SNN.firing_rate(model.pop, interval = 0.1s:10ms:1s, pop_average = true)
println(join(["$n: $(round(mean(r), digits = 2)) Hz" for (n, r) in zip(names, rates)], ", "))

fig = Figure(size = (800, 600))
SNN.raster!(Axis(fig[1, 1:2], ylabel = "Neuron"), model.pop, 0.1s:1s)
for (k, pop) in enumerate((E_active, E_passive))
    ax = Axis(fig[2, k], title = pop.name, xlabel = "Time (s)", ylabel = "V (mV)")
    SNN.vecplot!(ax, pop, :v_d, neurons = 1, interval = 0.5s:0.5ms:1s, label = "dendrite")
    SNN.vecplot!(ax, pop, :v_s, neurons = 1, interval = 0.5s:0.5ms:1s, label = "soma")
    axislegend(ax)
end
save(joinpath(mktempdir(), "passive_vs_active_dendrites.png"), fig)
fig
