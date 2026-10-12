# Tripod neuron: AdEx soma and two passive dendrites (160 and 400 um) with AMPA, NMDA, GABAa and
# GABAb receptors, driven by Poisson input onto both dendrites.
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

E = SNN.Tripod(
    N = 1,
    param = SNN.TripodParameter(ds = [160um, 400um]),
    adex = SNN.AdExParameter(C = 281pF, gl = 40nS, Vr = -55.6mV, El = -70.6mV, ΔT = 2mV,
                             Vt = -50.4mV, a = 4nS, b = 80.5pA, τw = 144ms),
    spike = SNN.PostSpike(At = 10.0, τA = 30.0),
    soma_syn = SomaSynapse,
    dend_syn = DendSynapse,
)

poisson_exc = SNN.PoissonLayer(rate = 10.2Hz, N = 1000)
poisson_inh = SNN.PoissonLayer(rate = 3Hz, N = 1000)
stims = (
    exc_d1 = SNN.Stimulus(poisson_exc, E, :glu, :d1, conn = (μ = 0.1, ρ = 1), name = "noiseE_d1"),
    inh_d1 = SNN.Stimulus(poisson_inh, E, :gaba, :d1, conn = (μ = 0.1, ρ = 1), name = "noiseI_d1"),
    exc_d2 = SNN.Stimulus(poisson_exc, E, :glu, :d2, conn = (μ = 0.1, ρ = 1), name = "noiseE_d2"),
    inh_d2 = SNN.Stimulus(poisson_inh, E, :gaba, :d2, conn = (μ = 0.1, ρ = 1), name = "noiseI_d2"),
)

model = SNN.compose(; E, stims..., silent = true)
SNN.monitor!(E, [:v_s, :v_d1, :v_d2, :fire], sr = 1000Hz)
SNN.monitor!(E, :g, variables = :synvars_d1, sr = 1000Hz)
SNN.sim!(model, 3s)

interval = 1:2ms:SNN.get_time(model)
fig = Figure(size = (800, 700))
ax1 = Axis(fig[1, 1], ylabel = "Voltage (mV)", title = "Tripod neuron")
SNN.vecplot!(ax1, E, :v_d1; interval, neurons = 1, label = "Dendrite 1")
SNN.vecplot!(ax1, E, :v_d2; interval, neurons = 1, label = "Dendrite 2")
SNN.vecplot!(ax1, E, :v_s; interval, neurons = 1, add_spikes = true, label = "Soma")
axislegend(ax1)
ax2 = Axis(fig[2, 1], ylabel = "Conductance, dendrite 1 (nS)", xlabel = "Time (s)")
for (i, r) in enumerate(DendReceptors)
    SNN.vecplot!(ax2, E, :g; variables = :synvars_d1, sym_id = i, interval, neurons = 1,
                 label = r.name, factor = r.gsyn)
end
axislegend(ax2)
save(joinpath(mktempdir(), "tripod_neuron.png"), fig)
