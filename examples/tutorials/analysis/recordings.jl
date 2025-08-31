using SpikingNeuralNetworks
using Statistics
SNN.@load_units
import SpikingNeuralNetworks.SNNPlots: default, plot, histogram, Plots, plot!, savefig

## AdEx neuron with fixed external current connections with multiple receptors
E = SNN.AdEx(; N = 800, param = SNN.AdExParameter(; El = -50mV))
I = SNN.IF(; N = 200, param = SNN.IFParameter())
EE = SNN.SpikingSynapse(E, E, :he; μ = 2, p = 0.02, STPParam = SNN.MarkramSTPParameter())
EI = SNN.SpikingSynapse(E, I, :ge; μ = 30, p = 0.02)
IE = SNN.SpikingSynapse(I, E, :hi; μ = 50, p = 0.02, LTPParam = SNN.iSTDPRate(r=5Hz))
II = SNN.SpikingSynapse(I, I, :gi; μ = 10, p = 0.02)
model = SNN.compose(;  E, I, EE, EI, IE, II)


SNN.monitor!(E, [:ge, :gi, :v], sr=200Hz)
SNN.monitor!(model.pop, :fire)
SNN.train!(model = model; duration = 5second, pbar=true)

##
SNN.monitor!(EE, [:x, :u], :STPVars; sr=10Hz)
SNN.monitor!(IE, [:tpost]; sr=10Hz, variables=:LTPVars)
SNN.monitor!(EE, [:ρ], sr=10Hz)
SNN.monitor!(EI, [:W], sr=10Hz)
SNN.train!(model = model; duration = 5second)

interval = 1s:5s

## Spiketimes 
spiketimes = SNN.spiketimes(model.pop.E) ## all spiketimes
@info "Spiketimes is: type $(nameof(typeof(spiketimes))), size $(size(spiketimes)), neuron 1 has $(length(spiketimes[1])) spikes"

spiketimes = SNN.spiketimes(model.pop.E; interval) ## spiketimes in the interval
@info "Spiketimes is: type $(nameof(typeof(spiketimes))), size $(size(spiketimes)), neuron 1 has $(length(spiketimes[1])) spikes"

## Binned spikes
bins, r = SNN.bin_spiketimes(model.pop.E; interval)
@info "Bins is: type $(nameof(typeof(bins))), size $(size(bins)), r size: $(size(r))"

## Firing rate
fr, r = SNN.firing_rate(model.pop.E; interval) # interpolated firing rate
@info "Fr is: type $(nameof(typeof(fr))), size $(size(fr)), r size: $(size(r))"

fr, r = SNN.firing_rate(model.pop.E; interval, interpolate=false) # non interpolated firing rate 
@info "Fr is: type $(nameof(typeof(fr))), size $(size(fr)), r size: $(size(r))"


fr, r, labels = SNN.firing_rate(model.pop; interval) # non interpolated firing rate 
@info "Fr is: type $(nameof(typeof(fr))), size $(size(fr)), r size: $(size(r))"
fr
r
labels

fr = SNN.record(model.pop.E, :fire; interval)
@info "Fr is: type $(nameof(typeof(fr))), size $(size(fr))"

fr, r = SNN.record(model.pop.E, :fire; interval, range=true)
@info "Fr is: type $(nameof(typeof(fr))), size $(size(fr)), r size: $(size(r))"

SNN.record(model.pop.E, :spikes)

fr = SNN.record(model.pop.E, :fire; interval, interpolate=false)
@info "Fr is: type $(nameof(typeof(fr))), size $(size(fr))"

plot(mean(fr, dims=1)[1,:])


fr, r = SNN.record(model.pop.E, :fire; interval)

## In this case the interval cannot be selected. The resolution is defined by the sampling rate (sr=1kHz default) of the variable. The starting time is the time of the model when the variable was monitored. The ending time is the time of the model when the simulation ended. Indeed, it s not possible to de-activate the recordings while maintaining the variable in the monitored pool; this behaviour may be improved in the future.  

v = SNN.record(model.pop.E, :v)
@info "V is: type $(nameof(typeof(v))), size $(size(v))"
v[1,3.14s]
v[1:10, 2.4s:15ms:3.1s]

v, r = SNN.record(model.pop.E, :v, range=true)
@info "V is: type $(nameof(typeof(v))), size $(size(v)), r size: $(size(r))"

v = SNN.record(model.pop.E, :v, interpolate=false)
@info "V is: type $(nameof(typeof(v))), size $(size(v))"

## The interpolated variable is then accessible only in the recorded time range. In our case, the model was simulated for 7s, and the variable was monitored from 2s to 7s. The variable can be accessed only in this range.
interval = 1s:5s
v = SNN.record(model.pop.E, :v; interval)

interval = 3s:5s
v = SNN.record(model.pop.E, :v; interval)
v, r = SNN.record(model.pop.E, :v, range=true)

# similarly
v = SNN.record(model.pop.E, :v)
v[:, 1.02s] # it will return an error 
v[:, 3.54321s] # it will return the value at 3.54321s

## Synaptic variables


## The matrix of the synaptic weights at time point t can be obtained as follows. The matrix is a sparse matrix of size (N_E, N_E) where N_E is the number of neurons in the pre- and post-synaptic populations.
# 1. Get the sparse vector ρ at time point t, this returns only the non-zero elements of the matrix.
# 
ρ, r = SNN.record(EE, :ρ, range=true) 
histogram(ρ[:,6.5s])

# 2. Reconstruct the matrix from the sparse vector ρ at time point t. This operation reverses the sparse representation and returns the full matrix. The user can pass either the vector obtained from SNN.record or the synapse object and the symbol of the variable. 
ρ_mat1 = SNN.matrix(EE, ρ, 6.5s)
ρ_mat2 = SNN.matrix(EE, :ρ, 6.5s)
all(ρ_mat1 .== ρ_mat2) # true

#3. Get the matrix at multiple time points. This returns a 3D array of size (N_E, N_E, T) where T is the number of time points in the range.

ρ_T1 = SNN.matrix(EE, :ρ, 6.5s:10ms:7s)
ρ_T2 = SNN.matrix(EE, ρ, 6.5s:10ms:7s)


## Synaptic connectivity is stored in a sparse format of a matrix with dimensions (N_post, N_pre). The user can also access the synaptic weights of the connections.

W = SNN.matrix(EE) # the default is to return the synaptic strength matrix at the last time point
W = SNN.matrix(EE, :W)
ρ = SNN.record(EE, :ρ)


##  The user can access the pre- and post-synaptic neurons of a given neuron, or a set of neurons.
## Single neuron
neuron = 1
Is = SNN.postsynaptic(EE, neuron)
W[Is, neuron] |> mean

Js = SNN.presynaptic(EE, neuron)
W[neuron, Js] |> mean

## Multiple neurons
neurons = 1:10

W = SNN.matrix(EE)

Is = SNN.presynaptic(EE, neurons)
Js = SNN.postsynaptic(EE, neurons)


# default(palette = :okabe_ito)
SNN.raster(model.pop, 5s:10s, xlabel="Time (s)", ylabel="Neuron index", size=(800,400))

mean(SNN.record(EE, :STPVars_u), dims=1)[1,:] |> plot
mean(SNN.record(EE, :STPVars_x), dims=1)[1,:] |> plot!
mean(SNN.record(IE, :LTPVars_tpost), dims=1)[1,:] |> plot!


x,r = SNN.record(EE, :STPVars_x, range=true)
@info "x is: type $(nameof(typeof(x))), size $(size(x)), r size: $(size(r))"
x[1, 6.14s]
x[1:10, 6.4s:15ms:9.1s]

x, r = SNN.record(EE, :STPVars_x, range=true)
@info "x is: type $(nameof(typeof(x))), size $(size(x)), r size: $(size(r))"
x = SNN.record(EE, :STPVars_x, interpolate=false)
@info "x is: type $(nameof(typeof(x))), size $(size(x))"




## Examples:
# This is to show the effect of the τ parameter in the firing rate estimation. The interval is fixed, the resolution is always 1ms. Only the firing rate measure is affected by τ. Notice that the mean firing rate is not affected by τ.

fr, r = SNN.firing_rate(model.pop.E, interval, neurons=1:100, τ=100ms, pop_average=true)
plot(r, fr)
    # plot!(twinx(), interval./1000, fr, c=:black, lw=2, label="FR τ = $τ ms; mean $(round(mean(fr[1, interval]), digits=2)) Hz", ylabel="Firing rate (Hz)", legend=:topright)
import SpikingNeuralNetworks: vecplot
using Plots
interval = 1s:1ms:5s
neurons =1:10
plots = []
for τ in [1ms, 10ms, 40ms, 100ms]
    fr, r = SNN.firing_rate(model.pop.E, interval, neurons=neurons, τ=τ, pop_average=true)
    p1 = vecplot(model.pop.E, :v, neurons=neurons, r=interval, pop_average=true)
    plot!(twinx(), interval./1000, fr, c=:black, lw=2, label="FR τ = $τ ms; mean $(round(mean(fr), digits=2)) Hz", ylabel="Firing rate (Hz)", legend=:topright, xlims=(1,5))
    # bins, r = SNN.bin_spiketimes(model.pop.E, interval)
    p2 = bar(r./1000, bins[1, :], width=1, label="Binned spikes (1 ms)", c=:black, alpha=0.5, )
    p = plot(p1, p2, layout=(2,1), xlims = (1, 5))
    plot!(size=(900,500))
    push!(plots, p1)
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