using DrWatson
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils
using Plots
using Statistics
using StatsBase
using Distributions
using LaTeXStrings

##
include("../../parameters/Mongillo_WM2008.jl")
model, assemblies = Mongillo2008(n_assemblies = 2)

peak_rate = 2kHz
stim_parameters = Dict(:decay=>1ms, :peak=>peak_rate, :start=>peak_rate)
# intervals =[[3s, 3s+0.3s],
#             [5.5s, 5.8s ]]
intervals = [[-1, -1], [-1, -1]]
stim_assembly = Dict(
    assembly.name=>begin
        variables = merge(stim_parameters, Dict(:intervals=>[intervals[assembly.id]]))
        param = PSParam(rate = attack_decay, variables = variables)
        SNN.PoissonStimulus(
            model.pop.E,
            :ge,
            μ = 1pF,
            neurons = assembly.neurons,
            param = param,
            name = string(assembly.name),
        )
    end for assembly in assemblies
)

model = SNN.merge_models(model, stim_assembly)
#
SNN.monitor([model.pop...], [:fire, :v], sr = 50Hz)
SNN.monitor(model.syn.EE, [:u, :x], sr = 50Hz)
w_rec = [assemblies[1].indices..., indices(model.syn.EE, 81:160, 81:160)...]
SNN.monitor(model.syn.EE, [(:ρ, w_rec), (:W, w_rec)], sr = 20Hz)

## Training
simtime = SNN.train!(model = model, duration = 1.3s, dt = 0.125, pbar = true)

root = datadir("working_memory", "Mongillo2008") |> x -> (mkpath(dirname(x)); x)
path = SNN.save_model(
    path = root,
    model = model,
    name = "8000_neurons_oneitem_nostim",
    assemblies = assemblies,
    simtime = simtime,
)
interval = 0s:10ms:get_time(simtime)

stp_plot(model, interval, assemblies)
##
@unpack model, simtime, assemblies = SNN.load_data(path)
simtime =
    SNN.train!(model = model, duration = 0.5s, dt = 0.125, pbar = true, time = simtime)
μee_assembly = 0.48 * 8000/model.pop.E.N * 2.5
update_weights!(model.syn.EE, assemblies[1].neurons, assemblies[1].neurons, μee_assembly)
update_weights!(model.syn.EE, assemblies[2].neurons, assemblies[2].neurons, μee_assembly)

using ProgressBars
input = 0.08pA
stimuli = []
for x in ProgressBar(1:15)
    my_stim_int = [copy(get_time(simtime))]
    model.stim.E.I_base .= input
    simtime = SNN.train!(model = model, duration = 0.10s, dt = 0.125, time = simtime)
    my_stim_int = push!(my_stim_int, copy(get_time(simtime)))
    model.stim.E.I_base .= 0.0f0
    push!(stimuli, my_stim_int)
    SNN.train!(model = model, duration = 0.20s, dt = 0.125, time = simtime)

end
interval = 2s:10ms:get_time(simtime)
stp_plot(model, interval, assemblies, stimuli)
