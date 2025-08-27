using SpikingNeuralNetworks
SNN.@load_units

## AdEx neuron with fixed external current connections with multiple receptors
E = SNN.AdEx(; N = 800, param = SNN.AdExParameter(; El = -50mV))

I = SNN.IF(; N = 200, param = SNN.IFParameter())
EE = SNN.SpikingSynapse(E, E, :he; μ = 2, p = 0.02)
EI = SNN.SpikingSynapse(E, I, :ge; μ = 30, p = 0.02)
IE = SNN.SpikingSynapse(I, E, :hi; μ = 50, p = 0.02)
II = SNN.SpikingSynapse(I, I, :gi; μ = 10, p = 0.02)
model = SNN.compose(;  E, I, EE, EI, IE, II)

SNN.monitor!(E, [:ge, :gi, :v])
SNN.monitor!(model.pop, [:fire])
SNN.sim!(model = model; duration = 5second)

default(palette = :okabe_ito)


##
import SpikingNeuralNetworks.SNNPlots: default, plot, histogram, Plots, plot!, savefig

#= the :spikes can be obtained in three ways. 

1. The spiketimes: They are stored in a Vector of Vector.  
such that the first is the neurons, and the second the actual times in milliseconds. ([neurons][times])

2. The binned_spikes: A matrix of size (neurons, bins) where each entry is the number of spikes in that bin. Time is binned in the time points defined by the interval step (keyworded argument).
The function returns the binned spikes and the time points of the bins.

3. The firing rate: An interpolated array that samples a continuous firing rate signal in the time points defined by the interval (keyworded argument, mandatory). The continuous signal is obtained by convolving the binned spike train with an alpha-function kernel with time constant τ (keyword argument, default 10ms).
The binned spike train is computed with the function `bin_spiketimes` and the resolution defined by the interval step. The firing rate is returned as a matrix of size (neurons, time points) where each entry is the firing rate in Hz at that time point. The firing rate is scaled such that the average over time is equal to the average firing rate of the neuron.
=#

interval = 1s:2s

spiketimes = SNN.spiketimes(model.pop.E) ## all spiketimes
spiketimes = SNN.spiketimes(model.pop.E; interval) ## spiketimes in the interval

bins, r = SNN.bin_spiketimes(model.pop.E; interval)

fr, r = SNN.firing_rate(model.pop.E; interval) # interpolated firing rate
# fr, r = SNN.firing_rate(model.pop.E; interval, interpolate=false) # interpolated firing rate ## TODO: Fix, this is bugged
# to have the non-interpolated fr, one can do: 
non_interp_fr = fr[:,r]

fr = SNN.record(model.pop.E, :fire; interval)


## In this case the interval cannot be selected. The resolution is defined by the sampling rate (sr=1kHz default) of the variable. The starting time is assumed to be the moment when the variable is monitored. TODO: This behaviour should be improved so that the interval can be explicity selected.

v = SNN.record(model.pop.E, :v)
v, r = SNN.record(model.pop.E, :v, range=true)
v = SNN.record(model.pop.E, :v, interpolate=false)

## Examples:
# This is to show the effect of the τ parameter in the firing rate estimation. The interval is fixed, the resolution is always 1ms. Only the firing rate measure is affected by τ. Notice that the mean firing rate is not affected by τ.

interval = 1s:1ms:5s
plots = []
for τ in [1ms, 10ms, 40ms, 100ms]
    fr, r = SNN.firing_rate(model.pop.E, interval, neurons=1, τ=τ)
    p1 = vecplot(model.pop.E, :v, neurons=1, r=interval, add_spikes=true)
    plot!(twinx(), interval./1000, fr[1, interval], c=:black, lw=2, label="FR τ = $τ ms; mean $(round(mean(fr[1, interval]), digits=2)) Hz", ylabel="Firing rate (Hz)", legend=:topright)
    bins, r = SNN.bin_spiketimes(model.pop.E, interval)
    p2 = bar(r./1000, bins[1, :], width=1, label="Binned spikes (1 ms)", c=:black, alpha=0.5, )
    p = plot(p1, p2, layout=(2,1), xlims = (1, 5))
    plot!(size=(900,500))
    push!(plots, p)
end
p = plot(plots..., layout=(4,1), leftmargin=10Plots.mm, rightmargin=10Plots.mm, xlabel="", ylabel="")
plot!(p, subplot=8,  xticks=(0:1s:5s, 0:1s:5s), xlabel = "Time (s)")
plot!(p, subplot=7,  ylabel = "Membrane potential (mV)")
plot!(p, subplot=8,  ylabel = "Firing rate (Hz)")
plot!(p, legend=:topleft, size = (900,1200))
# savefig(p, joinpath(SNN.DOCS_ASSETS_PATH, "AdEx_firing_rate_convolution.png"))

p

##  This is to show the effect of the interval parameter in binning and . The τ is fixed to 10ms. The resolution of the firing rate is affected by the interval step. The mean firing rate is not affected by the interval.

plots = []
for ΔT in [1ms, 10ms, 40ms, 100ms]
    interval = 1s:ΔT:5s

    fr, r = SNN.firing_rate(model.pop.E, interval, neurons=1, τ=10ms)
    p1 = vecplot(model.pop.E, :v, neurons=1, r=1s:1ms:5s, add_spikes=true)
    plot!(twinx(), interval./1000, fr[1, interval], c=:black, lw=2, label="FR ΔT = $ΔT ms; mean $(round(mean(fr[1, interval]), digits=2)) Hz", ylabel="Firing rate (Hz)", legend=:topright)
    bins, r = SNN.bin_spiketimes(model.pop.E, interval)
    p2 = bar(r./1000, bins[1, :], width=1, label="Binned spikes ($ΔT ms)", c=:black, alpha=0.5, )
    p = plot(p1, p2, layout=(2,1), xlims = (1, 5))
    plot!(size=(900,500))
    push!(plots, p)
end
p = plot(plots..., layout=(4,1), leftmargin=10Plots.mm, rightmargin=10Plots.mm, xlabel="", ylabel="")
plot!(p, subplot=8,  xticks=(0:1s:5s, 0:1s:5s), xlabel = "Time (s)")
plot!(p, subplot=7,  ylabel = "Membrane potential (mV)")
plot!(p, subplot=8,  ylabel = "Firing rate (Hz)")
plot!(p, legend=:topleft, size = (900,1200))
# savefig(p, joinpath(SNN.DOCS_ASSETS_PATH, "AdEx_firing_rate_interval.png"))

p