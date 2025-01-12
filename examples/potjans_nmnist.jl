using Revise
using DrWatson
using SpikingNeuralNetworks
SNN.@load_units
import SpikingNeuralNetworks: AdExParameter, IFParameter
using Statistics, Random
using Plots
using SparseArrays
using ProgressMeter
using Plots
using SpikingNeuralNetworks
using SNNUtils
using JLD2
using Distributions
include("Tonic_NMNIST_Stimulus.jl")
using .Tonic_NMNIST_Stimulus



"""
Auxiliary Potjans parameters for neural populations with scaled cell counts
"""
function potjans_neurons(scale=1.0)
    ccu = Dict(
        :E23 => trunc(Int32, 20683 * scale), 
        :E4 => trunc(Int32, 21915 * scale),
        :E5 => trunc(Int32, 4850 * scale),  
        :E6 => trunc(Int32, 14395 * scale),
        :I6 => trunc(Int32, 2948 * scale),  
        :I23 => trunc(Int32, 5834 * scale),
        :I5 => trunc(Int32, 1065 * scale),  
        :I4 => trunc(Int32, 5479 * scale)
    )

    neurons = Dict{Symbol, SNN.AbstractPopulation}()
    for (k, v) in ccu
        if occursin("E", String(k))
            neurons[k] = IF(N = v, param=LKD2014SingleExp.PV, name=string(k))
        else
            neurons[k] = IF(N = v, param=LKD2014SingleExp.PV, name=string(k))
        end
    end
    return neurons
end

"""
Define Potjans parameters for neuron populations and connection probabilities
"""
function potjans_conn(Ne)

    function j_from_name(pre, post)
        if occursin("E", String(pre)) && occursin("E", String(post))
            syn_weight_dist = Normal(0.15, 0.1)
            
            return rand(syn_weight_dist)
        elseif occursin("I", String(pre)) && occursin("E", String(post))
            syn_weight_dist = Normal(0.15, 0.1)
            return -4.0*rand(syn_weight_dist)

        elseif occursin("E", String(pre)) && occursin("I", String(post))
            syn_weight_dist = Normal(0.15, 0.1)
            
            return rand(syn_weight_dist)
        elseif occursin("I", String(pre)) && occursin("I", String(post))
            syn_weight_dist = Normal(0.15, 0.1)
            return -4.0*rand(syn_weight_dist)
        else 
            throw(ArgumentError("Invalid pre-post combination: $pre-$post"))
        end
    end

    

    layer_names = [:E23, :I23, :E4, :I4, :E5, :I5, :E6, :I6]



    total_cortical_thickness = 1500.0
    N_full = [20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948] 
    N_E_total = N_full[1]+N_full[3]+N_full[5]+N_full[7]
    dimensions_3D = Dict(
        "x_dimension"=> 1000,
        "z_dimension"=>  1000,
        "total_cortical_thickness"=> total_cortical_thickness,

        # Have the thicknesses proportional to the numbers of E cells in each layer
        "layer_thicknesses"=> Dict(
        "L23"=> total_cortical_thickness*N_full[1]/N_E_total,
        "L4" => total_cortical_thickness*N_full[3]/N_E_total,
        "L5" => total_cortical_thickness*N_full[5]/N_E_total,
        "L6" => total_cortical_thickness*N_full[7]/N_E_total,
        "thalamus" => 100
        )
    )
    
    net_dict = Dict{String, Any}(
        "PSP_e"=> 0.15,
        # Relative standard deviation of the postsynaptic potential.
        "PSP_sd"=> 0.1,
        # Relative inhibitory synaptic strength (in relative units).
        "g"=> -4,
        # Rate of the Poissonian spike generator (in Hz).
        "bg_rate"=> 8.,
        # Turn Poisson input on or off (True or False).
        "poisson_input"=> true,
        # Delay of the Poisson generator (in ms).
        "poisson_delay"=> 1.5,
        # Mean delay of excitatory connections (in ms).
        "mean_delay_exc"=> 1.5,
        # Mean delay of inhibitory connections (in ms).
        "mean_delay_inh"=> 0.75,
        # Relative standard deviation of the delay of excitatory and
        # inhibitory connections (in relative units).
        "rel_std_delay"=> 0.5
    )
    # Replace static matrix with a regular matrix for `conn_probs`
    # ! the convention is j_post_pre. This is how the matrices `w` are built. Are you using that when defining the parameters?
    conn_probs = Float32[
        0.1009  0.1689 0.0437 0.0818 0.0323 0.0     0.0076 0.0    
        0.1346  0.1371 0.0316 0.0515 0.0755 0.0     0.0042 0.0    
        0.0077  0.0059 0.0497 0.135  0.0067 0.0003  0.0453 0.0    
        0.0691  0.0029 0.0794 0.1597 0.0033 0.0     0.1057 0.0    
        0.1004  0.0622 0.0505 0.0057 0.0831 0.3726  0.0204 0.0    
        0.0548  0.0269 0.0257 0.0022 0.06   0.3158  0.0086 0.0    
        0.0156  0.0066 0.0211 0.0166 0.0572 0.0197  0.0396 0.2252
        0.0364  0.001  0.0034 0.0005 0.0277 0.008   0.0658 0.1443
    ]


    conn_j = zeros(Float32, size(conn_probs))
    for pre in eachindex(layer_names)
        for post in eachindex(layer_names)
            
            conn_j[post, pre ] = j_from_name(layer_names[pre], layer_names[post])     

        end
    end
    return layer_names, conn_probs, conn_j,net_dict
end


"""
Main function to setup Potjans layer with memory-optimized connectivity
"""
function potjans_layer(scale)

    ## Create the neuron populations
    neurons = potjans_neurons(scale)
    exc_pop = filter(x -> occursin("E", String(x)), keys(neurons))
    inh_pop = filter(x -> occursin("I", String(x)), keys(neurons))
    Ne = trunc(Int32, sum([neurons[k].N for k in exc_pop]))
    Ni = trunc(Int32, sum([neurons[k].N for k in inh_pop]))
    layer_names, conn_probs, conn_j,net_dict = potjans_conn(Ne)
    syn_weight_dist = Normal(0.15, 0.1)
    delay_dist_exc = Normal(1.5, 0.5)
    delay_dist_inh = Normal( 0.75, 0.5)

    ## Create the synaptic connections based on the connection probabilities and synaptic weights assigned to each pre-post pair
    connections = Dict()
    stdp_param = STDPParameter(A_pre =5e-1, 
        A_post=-5e-1,
        τpre  =20ms,
        τpost =15ms)

    for i in eachindex(layer_names)
        for j in eachindex(layer_names)
            pre = layer_names[i]
            post = layer_names[j]
            p = conn_probs[j, i]
            J = conn_j[j, i]
            sym = J>=0 ? :ge : :gi
            μ = abs(J)
            if J>=0        
                s = SNN.SpikingSynapse(neurons[pre], neurons[post], sym; μ = μ, p=p, σ=0,param=stdp_param, delay_dist=delay_dist_exc)
            else
                s = SNN.SpikingSynapse(neurons[pre], neurons[post], sym; μ = -μ, p=p, σ=0, delay_dist=delay_dist_inh)
            
            end
            connections[Symbol(string(pre,"_", post))] = s
        end
    end

    full_mean_rates = [0.971, 2.868, 4.746, 5.396, 8.142, 9.078, 0.991, 7.523]
    stimuli = Dict()
    for (ind,pop) in enumerate(exc_pop)
        νe = full_mean_rates[ind]kHz
        post = neurons[pop]
        s = SNN.PoissonStimulus(post, :ge; param = νe, cells=:ALL, μ=1.f0, name="PoissonE_$(post.name)")
        stimuli[Symbol(string("PoissonE_", pop))] = s
    end
    return merge_models(neurons,connections, stimuli),neurons,connections,stimuli,net_dict
end

#if !isfile("cached_potjans_model.jld2")
    model,neurons_,connections_,stimuli_,net_dict = potjans_layer(0.085)
    @save "cached_potjans_model.jld2" model neurons_ connections_ stimuli_ net_dict
#else
    @load "cached_potjans_model.jld2" model neurons_ connections_ stimuli_ net_dict
#end
#@show(connections_.vals)

before_learnning_weights = model.syn[1].W
#=
ΔTs = -100:1:100ms
ΔWs = zeros(Float32, length(ΔTs))
Threads.@threads for i in eachindex(ΔTs)
    ΔT = ΔTs[i]
    #spiketime = [2000ms, 2000ms+ΔT]
    #neurons = [[1], [2]]
    #inputs = SpikeTime(spiketime, neurons)
    w = zeros(Float32, 2,2)
    w[1, 2] = 1f0
    st = Identity(N=max_neurons(inputs))
    stim = SpikeTimeStimulusIdentity(st, :g, param=inputs)
    syn = SpikingSynapse( st, st, nothing, w = w,  param = stdp_param)
    
    model = merge_models(pop=st, stim=stim, syn=syn, silent=true)
    SNN.monitor(model.pop..., [:fire])
    SNN.monitor(model.syn..., [:tpre, :tpost])
    train!(model=model, duration=3000ms, dt=0.1ms)
    ΔWs[i] = model.syn[1].W[1] - 1
end
=#

duration = 15000ms
SNN.monitor([model.pop...], [:fire])
#SNN.monitor([model.pop...], [:v], sr=200Hz)
SNN.sim!(model=model, duration=3000ms, dt=0.125, pbar = true)#, dt = 0.125)

#=
# Example data: Spike times and trial start times
s#pike_times = [0.1, 0.15, 0.2, 0.4, 0.6, 0.65, 0.8, 1.0, 1.1, 1.3, 1.5]
t#rial_starts = [0.0, 1.0, 2.0]  # Start times of each trial
num_trials = length(trial_starts)

# Parameters
bin_width = 0.1  # Bin width in seconds
time_window = (0.0, 1.0)  # Time window relative to each trial start

# Create bins
edges = collect(time_window[1]:bin_width:time_window[2])
num_bins = length(edges) - 1

# Initialize a matrix to hold spike counts
spike_counts = zeros(num_trials, num_bins)

# Bin spikes for each trial
for (i, trial_start) in enumerate(trial_starts)
    aligned_spikes = spike_times .- trial_start  # Align spikes to trial start
    filtered_spikes = aligned_spikes[aligned_spikes .>= time_window[1] .& aligned_spikes .< time_window[2]]
    spike_counts[i, :] = histcounts(filtered_spikes, edges)
end

# Compute average spike activity (optional)
average_spike_activity = mean(spike_counts, dims=1)

# Plot heatmap
heatmap(
    edges[1:end-1] .+ bin_width / 2,  # Bin centers
    1:num_trials,  # Trial indices
    spike_counts,
    xlabel="Time (s)",
    ylabel="Trial",
    color=:viridis,
    title="Spike Activity Heatmap"
)
=#
SNN.train!(model=model, duration=3000ms, dt=0.125, pbar = true)#, dt = 0.125)

#SNN.sim!(model=model; duration = duration, pbar = true, dt = 0.125)
display(SNN.raster(model.pop, [1.0s, 2s]))
savefig("without_stimulus.png")




after_learnning_weights0 = model.syn[1].W

training_order = shuffle(0:60000-1)
cached_spikes=[]
for i in 1:40
    push!(cached_spikes,Tonic_NMNIST_Stimulus.getNMNIST(training_order[i]))
end

"""
Filter for spike packets that belong to provided labels that pertain to single numeral characters.
"""


if !isfile("labeled_packets.jld2")
    labeled_packets = Tonic_NMNIST_Stimulus.spike_packets_by_labels(cached_spikes)
    filter_label = 0
    (population_code,time_and_offset) = labeled_packets[filter_label]
    for filter_label in range(0,9)
        (population_code_,time_and_offset_) = labeled_packets[filter_label]
        append!(population_code,population_code_)
        append!(time_and_offset,time_and_offset_)
    end
    @save "labeled_packets.jld2" population_code time_and_offset
else
    @load "labeled_packets.jld2" population_code time_and_offset
end



p1 = Plots.scatter()
scatter!(p1,
    time_and_offset,
    population_code,    
    ms = 0.5,  # Marker size
    ylabel = "Neuron Index" ,
    xlabel ="Time (ms)",
    title = "Spiking Activity with Distinct Characters", 
    legend=false
)
display(plot(p1))
savefig("stimulus.png")

neurons_as_nested_array = [ Vector{Int64}([n]) for n in population_code]
inputs = SpikeTime(time_and_offset,neurons_as_nested_array)

st = neurons_[:E4] #Identity(N=max_neurons(inputs))
w = ones(Float32,neurons_[:E4].N,max_neurons(inputs))*15


st = Identity(N=max_neurons(inputs))
stim = SpikeTimeStimulusIdentity(st, :g, param=inputs)


syn = SpikingSynapse( st, neurons_[:E4], nothing, w = w)#,  param = stdp_param)
model2 = merge_models(pop=[st,model], stim=[stim,stimuli_], syn=[syn,connections_], silent=false)

duration = 15000ms
SNN.monitor([model2.pop...], [:fire])
SNN.monitor([model2.pop...], [:v], sr=200Hz)
SNN.train!(model=model2; duration = duration, pbar = true, dt = 0.125)
#display(SNN.raster(model2.pop, [0s, 15s]))

after_learnning_weights1 = model.syn[1].W

@show(mean(before_learnning_weights))
@show(mean(after_learnning_weights0))
@show(mean(after_learnning_weights1))

#mean(model2.syn[1].W)

SNN.spiketimes(model.pop[1])

#x, y, y0 = SNN._raster(model2.pop.pop_2_E5,[1.95s, 2s]))

display(SNN.raster(model2.pop, [1.75s, 2s]))

savefig("with_stimulus.png")

Trange = 0:10:15s
frE, interval, names_pop = SNN.firing_rate(model2.pop, interval = Trange)
plot(mean.(frE), label=hcat(names_pop...), xlabel="Time [ms]", ylabel="Firing rate [Hz]", legend=:topleft)
savefig("firing_rate.png")

##

vecplot(model2.pop.E4, :v, neurons =1, r=0s:15s,label="soma")
savefig("vector_vm_plot.png")
layer_names, conn_probs, conn_j = potjans_conn(4000)
pj = heatmap(conn_j, xticks=(1:8,layer_names), yticks=(1:8,layer_names), aspect_ratio=1, color=:bluesreds,  title="Synaptic weights", xlabel="Presynaptic", ylabel="Postsynaptic", size=(500,500), clims=(-maximum(abs.(conn_j)), maximum(abs.(conn_j))))
pprob=heatmap(conn_probs, xticks=(1:8,layer_names), yticks=(1:8,layer_names), aspect_ratio=1, color=:viridis,  title="Connection probability", xlabel="Presynaptic", ylabel="Postsynaptic", size=(500,500))
plot(pprob, pj, layout=(1,2), size=(1000,500), margin=5Plots.mm)
