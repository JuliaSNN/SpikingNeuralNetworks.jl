# Ball-and-stick neuron: AdEx soma and one passive dendrite with AMPA, NMDA, GABAa and GABAb
# receptors, driven by Poisson input onto the dendrite.
using SpikingNeuralNetworks
using CairoMakie, Random
SNN.@load_units
Random.seed!(1234)

SomaSynapse = SNN.ReceptorSynapse(
    glu_receptors = [1], gaba_receptors = [2],
    syn = SNN.Receptors(
        SNN.Receptor(E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.73, target = :glu),
        SNN.Receptor(E_rev = -70.0, τr = 0.1, τd = 15.0, g0 = 0.38, target = :gaba),
    ),
)
DendReceptors = SNN.Receptors(
    SNN.Receptor(name = "AMPA", E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.73, target = :glu),
    SNN.Receptor(name = "NMDA", E_rev = 0.0, τr = 8, τd = 35.0, g0 = 1.31, nmda = 1.0f0, target = :glu),
    SNN.Receptor(name = "GABAa", E_rev = -70.0, τr = 4.8, τd = 29.0, g0 = 0.27, target = :gaba),
    SNN.Receptor(name = "GABAb", E_rev = -90.0, τr = 30, τd = 400.0, g0 = 0.0006, target = :gaba),
)
DendSynapse = SNN.ReceptorSynapse(
    glu_receptors = [1, 2], gaba_receptors = [3, 4], syn = DendReceptors,
    NMDA = SNN.NMDAVoltageDependency(mg = 1.0, b = 3.36, k = -0.077),   # Eyal et al. 2018
)

E = SNN.BallAndStick(
    N = 1,
    param = SNN.BallAndStickParameter(ds = [160um], physiology = SNN.human_dend),
    adex = SNN.AdExParameter(C = 281pF, gl = 40nS, Vr = -55.6mV, El = -70.6mV, ΔT = 2mV,
                             Vt = -50.4mV, a = 4nS, b = 80.5pA, τw = 144ms),
    spike = SNN.PostSpike(At = 10.0, τA = 30.0),
    soma_syn = SomaSynapse,
    dend_syn = DendSynapse,
)

stim_exc = SNN.Stimulus(SNN.PoissonLayer(rate = 10.2Hz, N = 1000), E, :glu, :d;
                        conn = (p = 1, μ = 1.0), name = "noiseE")
stim_inh = SNN.Stimulus(SNN.PoissonLayer(rate = 3Hz, N = 1000), E, :gaba, :d;
                        conn = (p = 1, μ = 4.0), name = "noiseI")

model = SNN.compose(; E, stim_exc, stim_inh, silent = true)
SNN.monitor!(E, [:v_s, :v_d, :fire], sr = 1000Hz)
SNN.monitor!(E, :g, variables = :synvars_d, sr = 1000Hz)
SNN.sim!(model, 3s)

interval = 1:2ms:SNN.get_time(model)
fig = Figure(size = (800, 700))
ax1 = Axis(fig[1, 1], ylabel = "Voltage (mV)", title = "Ball-and-stick neuron")
SNN.vecplot!(ax1, E, :v_d; interval, neurons = 1, label = "Dendrite")
SNN.vecplot!(ax1, E, :v_s; interval, neurons = 1, add_spikes = true, label = "Soma")
axislegend(ax1)
ax2 = Axis(fig[2, 1], ylabel = "Dendritic conductance (nS)", xlabel = "Time (s)")
for (i, r) in enumerate(DendReceptors)
    SNN.vecplot!(ax2, E, :g; variables = :synvars_d, sym_id = i, interval, neurons = 1,
                 label = r.name, factor = r.gsyn)
end
axislegend(ax2)
save(joinpath(mktempdir(), "ballandstick_neuron.png"), fig)
