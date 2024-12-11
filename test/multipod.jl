using DrWatson
using Plots
using Revise
using SpikingNeuralNetworks
using BenchmarkTools
SNN.@load_units;

### Define neurons and synapses in the network
N = 1
dendrites = [300um, 300um]
E = SNN.MultipodNeurons(dendrites, N)
Ie = SNN.PoissonStimulus(E, :he, 1, cells=:ALL, param=2.8kHz)
Ii = SNN.PoissonStimulus(E, :hi, 1, cells=:ALL, param=3.8kHz)
I1.W

model = merge_models(E=E, I1=Ie, I2=Ii)
SNN.monitor(E, [:v_d, :v_s, :g_d])
@btime sim!(model=model, duration=10s, pbar=true)

g_d = E.records[:g_d]
plot([E.records[:g_d][i][1,1,2] for i in eachindex(g_d)])
v = SNN.getrecord(model.pop.E, :v_d)
v_d, r_t = SNN.interpolated_record(model.pop.E, :v_d)

p1=plot()
vecplot!(p1, model.pop.E, :v_d, neurons=1,r=1:0.1:1000, sym_id=1)
vecplot!(p1, model.pop.E, :v_s, neurons=1,r=1:0.1:1000, sym_id=3)
plot!(ylims=:auto, title="Multipod")
##

N = 1
dendrites = [300um, 300um]
E = SNN.Tripod(dendrites..., N=N)
Ie = SNN.PoissonStimulus(E, :he, :d1, cells=:ALL, param=2.8kHz)
Ii = SNN.PoissonStimulus(E, :hi, :d1, cells=:ALL, param=3.8kHz)
SNN.monitor(E, [:g_d1, :v_d1, :v_s, :g_d1])
SNN.monitor(E, [:fire])
I.W

model = merge_models(E=E, I1=Ie, I2=Ii)
@btime train!(model=model, duration=1s, pbar=true)
p2 = plot()
vecplot!(p2,model.pop.E, :v_d1, neurons=1,r=1:0.1:1000,)
vecplot!(p2, model.pop.E, :v_s, neurons=1,r=1:0.1:1000,)
plot!(ylims=:auto, title="Tripod")

plot(p1,p2)