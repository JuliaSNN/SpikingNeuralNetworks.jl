using DrWatson
using SNNPlots
using Revise
using SpikingNeuralNetworks
using BenchmarkTools
SNN.@load_units;

### Define neurons and synapses in the network
using Random
Random.seed!(1234)
N = 1
dendrites = [200um, 300um]
E = SNN.Multipod(dendrites, N = N)# dend_syn=SNNUtils.quaresima2022_nar(1.0, 35ms).dend_syn)
stimE = Dict(
    Symbol("stimE_$n") => SNN.PoissonStimulus(
        E,
        :he,
        n,
        neurons = :ALL,
        param = 3.5kHz,
        name = "stimE_$n",
    ) for n = 1:length(dendrites)
)
stimI = Dict(
    Symbol("stimI_$n") =>
        SNN.PoissonStimulus(E, :hi, n, neurons = :ALL, param = 3kHz, name = "stimI_$n")
    for n = 1:length(dendrites)
)

model = merge_models(E = E, stimE, stimI)

SNN.monitor!(E, [:v_d, :v_s, :g_d])
sim!(model = model, duration = 7s, pbar = true)
p1 = plot()
vecplot!(p1, model.pop.E, :v_d, neurons = 1, r = 1s:0.1:7s, sym_id = 1)
vecplot!(p1, model.pop.E, :v_s, neurons = 1, r = 1s:0.1:7s, sym_id = 3)
plot!(ylims = :auto, title = "Multipod")

# ##
# g_d = E.records[:g_d]
# plot([E.records[:g_d][i][1,1,2] for i in eachindex(g_d)])
# v = SNN.getrecord(model.pop.E, :v_d)
# v_d, r_t = SNN.interpolated_record(model.pop.E, :v_d)
# v_d, r_t = SNN.interpolated_record(model.pop.E, :v_d)
# plot(r_t,v_d[1,1,r_t])
# cor(v_d[1,1,r_t], v_d[1,3,r_t])
##

using Random
Random.seed!(1234)
dendrites = [200um, 300um]
E = SNN.Tripod(dendrites..., N = N)
stimE = Dict(
    Symbol("stimE_$n") => SNN.PoissonStimulus(
        E,
        :he,
        Symbol("d$n"),
        neurons = :ALL,
        param = 3.5kHz,
        name = "stimE_$n",
    ) for n = 1:length(dendrites)
)
stimI = Dict(
    Symbol("stimI_$n") => SNN.PoissonStimulus(
        E,
        :hi,
        Symbol("d$n"),
        neurons = :ALL,
        param = 3kHz,
        name = "stimI_$n",
    ) for n = 1:length(dendrites)
)
SNN.monitor!(E, [:g_d1, :v_d1, :v_s, :g_d1])
SNN.monitor!(E, [:fire])

model = merge_models(E = E, stimE, stimI)
sim!(model = model, duration = 10s, pbar = true)
p2 = plot()
vecplot!(p2, model.pop.E, :v_d1, neurons = 1, r = 1s:0.1:7s, sym_id = 1)
vecplot!(p2, model.pop.E, :v_s, neurons = 1, r = 1s:0.1:7s, sym_id = 3)
plot!(ylims = :auto, title = "Tripod")

##
plot(p1, p2, layout = (2, 1), size = (800, 800), margin = 5SNNPlots.mm)


##
@unpack νs, kie = SNNUtils.tripod_balance.dend

scatter(νs, kie.nmda[:, 10], xscale = :log)



## Compare synapse parameters
nar = SNNUtils.quaresima2022_nar(1.8, 35ms).dend_syn
base = SNN.TripodDendSynapse
syn_dend = EyalEquivalentNAR(1.8)

for r in [:AMPA, :NMDA, :GABAa, :GABAb]
    @info r
    for field in fieldnames(typeof(getfield(nar, r)))
        println(
            "$field:    ",
            round(getfield(getfield(nar, r), field), digits = 3),
            " ",
            round(getfield(getfield(base, r), field), digits = 3),
            " ",
            round(getfield(getfield(syn_dend, r), field), digits = 3),
        )
    end
end
