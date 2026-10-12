using SpikingNeuralNetworks
using Statistics
using CairoMakie
SNN.@load_units

## E/I network: STP on the excitatory synapses, inhibitory STDP on I -> E
E = SNN.AdEx(; N = 800, param = SNN.AdExParameter(; El = -50mV))
I = SNN.IF(; N = 200, param = SNN.IFParameter())
EE = SNN.SpikingSynapse(E, E, :he; conn = (μ = 2, p = 0.02), STPParam = SNN.MarkramSTPParameter())
EI = SNN.SpikingSynapse(E, I, :ge; conn = (μ = 30, p = 0.02), STPParam = SNN.MarkramSTPParameter())
IE = SNN.SpikingSynapse(I, E, :hi; conn = (μ = 50, p = 0.02), LTPParam = SNN.iSTDPRate(r = 5Hz))
II = SNN.SpikingSynapse(I, I, :gi; conn = (μ = 10, p = 0.02))
model = SNN.compose(; E, I, EE, EI, IE, II, silent = true)

## Recording setup. Buffers are allocated just before the run.
# Continuous variables (v, conductances, synaptic variables) are stored in a dense array
# (neurons x samples); spikes (:fire) as a list of (time, neuron) events. Plasticity variables
# are recorded at their value at the sampling time.
SNN.monitor!(EE, [:x, :u], :STPVars; sr = 10Hz)
SNN.monitor!(IE, [:tpost]; sr = 10Hz, variables = :LTPVars)
SNN.monitor!(EE, [:ρ], sr = 10Hz)
SNN.monitor!(EI, [:W], sr = 10Hz)
SNN.monitor!(model.pop, [:fire])
SNN.monitor!(model.pop, [:v], sr = 200Hz)
SNN.train!(model = model; duration = 5second)

interval = 1s:5s

## Spike times, binned spikes, firing rates
spiketimes = SNN.spiketimes(model.pop.E)                   # all spike times
spiketimes = SNN.spiketimes(model.pop.E; interval)         # spike times in the interval
@info "Spiketimes: $(length(spiketimes)) neurons, neuron 1 has $(length(spiketimes[1])) spikes"
bins, r = SNN.bin_spiketimes(model.pop.E; interval)
@info "Bins: size $(size(bins)), $(length(r)) bins"
fr, r = SNN.firing_rate(model.pop.E; interval)                       # interpolated
fr, r = SNN.firing_rate(model.pop.E; interval, interpolate = false)  # matrix
fr, r, labels = SNN.firing_rate(model.pop; interval)                 # one entry per population
fr = SNN.record(model.pop.E, :fire; interval)                        # same through `record`
fr, r = SNN.record(model.pop.E, :fire; interval, range = true)
st = SNN.record(model.pop.E, :spikes)

## Continuous variables: interpolated in time (call syntax), or raw samples
v = SNN.record(model.pop.E, :v)
v(1, 3.14s)
v(1:10, 2.4s:15ms:3.1s)
v, r = SNN.record(model.pop.E, :v, range = true)
v_raw = SNN.record(model.pop.E, :v, interpolate = false)    # same as getvariable
@info "v: $(size(v_raw)) samples, times $(first(r)) to $(last(r)) ms"

## Per-synapse variables and weight matrices
ρ, r = SNN.record(EE, :ρ, range = true)
ρ_mat1 = SNN.matrix(EE, ρ, 4.5s)               # N_post x N_pre sparse matrix at 4.5 s
ρ_mat2 = SNN.matrix_record(EE, :ρ, 4.5s)
@assert all(ρ_mat1 .== ρ_mat2)
ρ_T = SNN.matrix_record(EE, :ρ, 2.5s:10ms:4.5s)  # N_post x N_pre x T
W = SNN.matrix(EE)                             # current weights
neuron = 1
Is = SNN.postsynaptic(EE, neuron)              # targets of neuron 1
mean(W[Is, neuron])
Js = SNN.presynaptic(EE, neuron)               # inputs of neuron 1
mean(W[neuron, Js])
Js_many = SNN.presynaptic(EE, 1:10)

## Plasticity variables
x, r = SNN.record(EE, :STPVars_x, range = true)
x(1, 4.1s)
tpost = SNN.record(IE, :LTPVars_tpost)

## Figure 1: raster and mean plasticity variables
fig = Figure(size = (800, 600))
SNN.raster!(Axis(fig[1, 1], ylabel = "Neuron"), model.pop, 4s:5s)
ax = Axis(fig[2, 1], xlabel = "Time (s)", ylabel = "Population mean")
lines!(ax, r ./ 1000, vec(mean(SNN.getvariable(EE, :STPVars_u), dims = 1)), label = "u (STP)")
lines!(ax, r ./ 1000, vec(mean(SNN.getvariable(EE, :STPVars_x), dims = 1)), label = "x (STP)")
lines!(ax, r ./ 1000, vec(mean(SNN.getvariable(IE, :LTPVars_tpost), dims = 1)), label = "tpost (iSTDP)")
axislegend(ax)
save(joinpath(mktempdir(), "recordings_variables.png"), fig)

## Figure 2: effect of the kernel width τ on the firing-rate estimate
fig = Figure(size = (800, 900))
interval = 1s:1ms:5s
for (k, τ) in enumerate([1ms, 10ms, 40ms, 100ms])
    fr, r = SNN.firing_rate(model.pop.E; interval, neurons = 1:10, τ, pop_average = true)
    ax = Axis(fig[k, 1], ylabel = "Rate (Hz)", title = "τ = $τ ms, mean $(round(mean(fr), digits = 2)) Hz")
    lines!(ax, r ./ 1000, fr, color = :black)
end
save(joinpath(mktempdir(), "recordings_firing_rate_tau.png"), fig)
