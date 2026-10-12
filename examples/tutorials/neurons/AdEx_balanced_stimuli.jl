# AdEx neuron in the high-conductance state: excitatory and inhibitory Poisson layers on the
# receptors :glu and :gaba of a double-exponential conductance synapse.
using SpikingNeuralNetworks
using CairoMakie
SNN.@load_units

neuron_parameter = (
    param = SNN.AdExParameter(R = 0.5GΩ, Vt = -50mV, ΔT = 2mV, El = -70mV, τm = 20ms, Vr = -55mV),
    synapse = SNN.DoubleExpSynapse(),
    spike = SNN.PostSpike(τabs = 5ms),
)
E = SNN.Population(; neuron_parameter..., N = 1)

poisson_exc = SNN.PoissonLayer(rate = 1Hz, N = 1000)
poisson_inh = SNN.PoissonLayer(rate = 10Hz, N = 1000)
conn = (p = 1, μ = 5nS)
stim_exc = SNN.Stimulus(poisson_exc, E, :glu; conn, name = "Exc Noise")
stim_inh = SNN.Stimulus(poisson_inh, E, :gaba; conn, name = "Inh Noise")

model = SNN.compose(; E, stim_exc, stim_inh, silent = true)
SNN.monitor!(E, [:v, :fire], sr = 2kHz)
SNN.monitor!(E, [:ge, :gi], sr = 2kHz, variables = :synvars)
SNN.monitor!(model.stim, [:fire])
SNN.sim!(; model, duration = 1000ms)

fig = Figure(size = (800, 800))
SNN.raster!(Axis(fig[1, 1], ylabel = "Input neuron"), model.stim)
ax2 = Axis(fig[2, 1], ylabel = "Conductance (nS)")
SNN.vecplot!(ax2, E, :ge, variables = :synvars, label = "exc")
SNN.vecplot!(ax2, E, :gi, variables = :synvars, label = "inh")
axislegend(ax2)
ax3 = Axis(fig[3, 1], ylabel = "Membrane potential (mV)", xlabel = "Time (s)")
SNN.vecplot!(ax3, E, :v, add_spikes = true, color = :black)
save(joinpath(mktempdir(), "balanced_stimuli.png"), fig)
