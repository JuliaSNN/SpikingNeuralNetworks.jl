# AdEx/IF network driven by a Poisson input whose rate is changed between runs
# (`PoissonVariable` and `set_variable!`).
using SpikingNeuralNetworks
using CairoMakie, Statistics
SNN.@load_units

E = SNN.Population(SNN.AdExParameter(; El = -70mV), synapse = SNN.DoubleExpSynapse(); N = 800, name = "Excitatory")
I = SNN.Population(SNN.IFParameter(), synapse = SNN.SingleExpSynapse(); N = 200,
                   name = "Inhibitory", spike = SNN.PostSpike())
EE = SNN.SpikingSynapse(E, E, :he; conn = (μ = 7, p = 0.02))
EI = SNN.SpikingSynapse(E, I, :ge; conn = (μ = 30, p = 0.02))
IE = SNN.SpikingSynapse(I, E, :hi; conn = (μ = 50, p = 0.02))
II = SNN.SpikingSynapse(I, I, :gi; conn = (μ = 10, p = 0.02))

# external drive: every excitatory neuron receives Poisson spikes at rate `vars[:rate]`
ext_variable = SNN.PoissonVariable(variables = Dict{Symbol,Any}(:rate => 0.0Hz),
                                   rate = (t, vars) -> vars[:rate], μ = 5.0)
ext = SNN.Stimulus(ext_variable, E, :ge)
model = SNN.compose(; E, I, EE, EI, IE, II, ext)
SNN.monitor!(model.pop, [:fire])
SNN.monitor!(E, [(:v, 1:3)])

SNN.sim!(model = model; duration = 1second)           # no input
SNN.set_variable!(ext, :rate, 3kHz)
SNN.sim!(model = model; duration = 2second)           # input on
SNN.set_variable!(ext, :rate, 0.0Hz)
SNN.sim!(model = model; duration = 1second)           # input off

## Plot
rates, r, names = SNN.firing_rate(model.pop, interval = 0:10ms:4s, pop_average = true)
fig = Figure(size = (800, 600))
ax1 = Axis(fig[1, 1], ylabel = "Firing rate (Hz)")
for (fr, name) in zip(rates, names)
    lines!(ax1, r ./ 1000, fr, label = name)
end
vspan!(ax1, [1], [3], color = (:gray, 0.2))
axislegend(ax1)
ax2 = Axis(fig[2, 1], ylabel = "Neuron")
SNN.raster!(ax2, model.pop, 0:4s)
ax3 = Axis(fig[3, 1], xlabel = "Time (s)", ylabel = "V (mV)")
SNN.vecplot!(ax3, E, :v, neurons = 1, interval = 0:1ms:4s, add_spikes = true)
save(joinpath(mktempdir(), "AdExVarStimuli.png"), fig)
fig
