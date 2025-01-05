using Plots
using SpikingNeuralNetworks
SNN.@load_units
import SpikingNeuralNetworks: AdExParameter
using Statistics, Random



##
stdp_param = STDPParameter(A_pre =-5e-1, 
                           A_post=-5e-1,
                           τpre  =20ms,
                           τpost =15ms)
# istdp_param = iSTDPParameterTime(η=0.2, τy=20ms)
ΔTs = -97.5:5:100ms
ΔWs = zeros(Float32, length(ΔTs))
Threads.@threads for i in eachindex(ΔTs)
    ΔT = ΔTs[i]
    spiketime = [2000ms, 2000ms+ΔT]
    neurons = [[1], [2]]
    inputs = SpikeTime(spiketime, neurons)
    w = zeros(Float32, 2,2)
    w[1, 2] = 1f0
    st = Identity(N=max_neurons(inputs))
    stim = SpikeTimeStimulusIdentity(st, :g, param=inputs)
    syn = SpikingSynapse(st, st, :h, w = w,  param = stdp_param)
    model = merge_models(pop=st, stim=stim, syn=syn, silent=true)
    SNN.monitor(model.pop..., [:fire])
    SNN.monitor(model.syn..., [:tpre, :tpost])
    train!(model=model, duration=3000ms, dt=0.1ms)
    ΔWs[i] = model.syn[1].W[1] - 1
end

n_plus = findall(ΔTs .>= 0)
n_minus = findall(ΔTs .< 0)
# R(X; f=maximum) = [f([x,0]) for x in X]
plot(ΔTs[n_minus],ΔWs[n_minus] , legend=false, fill=true,xlabel="ΔT", ylabel="ΔW", title="STDP", size=(500, 300), alpha=0.5)
plot!(ΔTs[n_plus],ΔWs[n_plus] , legend=false, fill=true,xlabel="T_pre - T_post ", ylabel="ΔW", title="STDP", size=(500, 300), alpha=0.5)
##

st = Poisson(N=50, param=PoissonParameter(rate=10Hz))
syn = SpikingSynapse( st, st, nothing,p=1.f0,  param = STDPParameter(A_pre=5e-2, A_post=-5e-2, τpre=15ms, τpost=15ms))
model = merge_models(pop=st, syn=syn, silent=true)
SNN.monitor(model.pop..., [:fire])
# SNN.monitor(model.syn..., [:W])
train!(model=model, duration=20000ms, dt=0.1ms)
# plot(SNN.getvariable(model.syn..., :W)[1,:])
# SNN.getvariable(model.syn..., :W)
mean(model.syn[1].W)

SNN.spiketimes(model.pop[1])

SNN.raster(model.pop..., [0s, 3s])

for i in 1:50
    for j in 1:50
        if i != j
            t_post = SNN.spiketimes(model.pop...)[i]
            t_pre = SNN.spiketimes(model.pop...)[j]
        end
    end
end

using Statistics, StatsPlots
violin(model.syn[1].W, legend=false, xlabel="Neuron", ylabel="ΔW", title="STDP", size=(500, 300))
# SNN.raster(model.pop, [0s, 3s])
##


## 
using SparseArrays

# Parameters
T = 10.0s               # Total time

# Example spike trains as floating-point times
t_pre = sort(rand(40) .* T)  # Post-synaptic spikes (50 random times in [0, T])
# t_pre = sort(rand(30) .* T)  # Post-synaptic spikes (50 random times in [0, T])
t_post = t_pre .+rand(length(t_pre))*10 .+200   # Pre-synaptic spikes (75 random times in [0, T])



r, cv = compute_covariance_density(t_pre, t_post, T, sr=40Hz, τ=300ms)
bar(r, cv, legend=false, fill=true, xlabel="ΔT(ms)", ylabel="C(τ)", size=(500, 300), alpha=0.5, color=:black, margin=5Plots.mm)
plot!(frame=:origin, yticks=:none, )


#

taus = autocorrelogram(SNN.spiketimes(model.pop...)[1], 200ms, 50Hz)
histogram(taus, bins=100, legend=false, fill=true, xlabel="ΔT(ms)", ylabel="Autocorrelogram", size=(500, 300), alpha=0.5, color=:black, margin=5Plots.mm, frame=:origin, yticks=:none)
