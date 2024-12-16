using DrWatson
using Plots
using Revise
using SpikingNeuralNetworks
using BenchmarkTools
SNN.@load_units;

### Define neurons and synapses in the network
N = 1
dendrites = [400um, ]
E = SNN.Multipod(dendrites, N=N)# dend_syn=SNNUtils.quaresima2022_nar(1.0, 35ms).dend_syn)

param = SNN.BalancedStimulusParameter(
    kIE = 2.5,
    wIE = 1.f0,
    r0 = 10kHz,
    β =0,
)
stim = Dict(Symbol("stimE_$n") =>SNN.BalancedStimulus(E, :he, :hi, n,  param=param, name="stimE_$n" ) for n in 1:length(dendrites))

model = merge_models(E=E, stim)

SNN.monitor(E, [:v_d, :v_s, :g_d])
sim!(model=model, duration=10s, pbar=true)

g_d = E.records[:g_d]
plot([E.records[:g_d][i][1,1,2] for i in eachindex(g_d)])
v = SNN.getrecord(model.pop.E, :v_d)
v_d, r_t = SNN.interpolated_record(model.pop.E, :v_d)

p1=plot()
vecplot!(p1, model.pop.E, :v_d, neurons=1,r=1:0.1:1000, sym_id=1)
vecplot!(p1, model.pop.E, :v_s, neurons=1,r=1:0.1:1000, sym_id=3)
plot!(ylims=:auto, title="Multipod")
##
v_d, r_t = SNN.interpolated_record(model.pop.E, :v_d)
plot(r_t,v_d[1,1,r_t])
cor(v_d[1,1,r_t], v_d[1,3,r_t])
##
@unpack dend_syn = SNNUtils.quaresima2022_nar(1.28, 15ms) 

N = 1
dendrites = [200um, 300um]
E = SNN.Tripod(dendrites..., N=N, dend_syn=dend_syn)
Ie = SNN.PoissonStimulus(E, :he, :d1, cells=:ALL, param=5.8kHz)
Ii = SNN.PoissonStimulus(E, :hi, :d1, cells=:ALL, param=1.8kHz)
SNN.monitor(E, [:g_d1, :v_d1, :v_s, :g_d1])
SNN.monitor(E, [:fire])

model = merge_models(E=E, I1=Ie, I2=Ii)
train!(model=model, duration=1s, pbar=true)
p2 = plot()
vecplot!(p2,model.pop.E, :v_d1, neurons=1,r=1:0.1:1000,)
vecplot!(p2, model.pop.E, :v_s, neurons=1,r=1:0.1:1000,)
plot!(ylims=:auto, title="Tripod")

##
plot(p1,p2)


##
@unpack νs, kie = SNNUtils.tripod_balance.dend

scatter(νs, kie.nmda[:,10], xscale=:log)


