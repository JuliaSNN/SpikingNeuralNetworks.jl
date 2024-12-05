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
include("Tonic_NMNIST_Stimulus.jl")
using .Tonic_NMNIST_Stimulus




""" 
https://github.com/OpenSourceBrain/PotjansDiesmann2014/blob/master/PyNN/network_params.py#L139-L146
Creates matrix containing the delay of all connections.

Arguments
---------
mean_delay_exc
    Delay of the excitatory connections.
mean_delay_inh
    Delay of the inhibitory connections.
number_of_pop
    Number of populations.

Returns
-------
mean_delays
    Matrix specifying the mean delay of all connections.

"""
function get_mean_delays(mean_delay_exc, mean_delay_inh, number_of_pop)

    dim = number_of_pop
    mean_delays = np.zeros((dim, dim))
    mean_delays[:, 0:dim:2] = mean_delay_exc
    mean_delays[:, 1:dim:2] = mean_delay_inh
    return mean_delays

""" 
https://github.com/OpenSourceBrain/PotjansDiesmann2014/blob/master/PyNN/network_params.py#L139-L146
Creates matrix containing the standard deviations of all delays.

Arguments
---------
std_delay_exc
    Standard deviation of excitatory delays.
std_delay_inh
    Standard deviation of inhibitory delays.
number_of_pop
    Number of populations in the microcircuit.

Returns
-------
std_delays
    Matrix specifying the standard deviation of all delays.

"""
function get_std_delays(std_delay_exc, std_delay_inh, number_of_pop)

    dim = number_of_pop
    std_delays = np.zeros((dim, dim))
    std_delays[:, 0:dim:2] = std_delay_exc
    std_delays[:, 1:dim:2] = std_delay_inh
    return std_delays

""" 
https://github.com/OpenSourceBrain/PotjansDiesmann2014/blob/master/PyNN/network_params.py#L139-L146
Creates a matrix of the mean evoked postsynaptic potential.

The function creates a matrix of the mean evoked postsynaptic
potentials between the recurrent connections of the microcircuit.
The weight of the connection from L4E to L23E is doubled.

Arguments
---------
PSP_e
    Mean evoked potential.
g
    Relative strength of the inhibitory to excitatory connection.
number_of_pop
    Number of populations in the microcircuit.

Returns
-------
weights
    Matrix of the weights for the recurrent connections.

"""

function get_mean_PSP_matrix(PSP_e, g, number_of_pop)
    dim = number_of_pop
    weights = zeros((dim, dim))
    exc = PSP_e
    inh = PSP_e * g
    weights[:, 0:dim:2] = exc
    weights[:, 1:dim:2] = inh
    weights[1, 3] = exc * 2
    return weights

""" 
https://github.com/OpenSourceBrain/PotjansDiesmann2014/blob/master/PyNN/network_params.py#L139-L146
Relative standard deviation matrix of postsynaptic potential created.

The relative standard deviation matrix of the evoked postsynaptic potential
for the recurrent connections of the microcircuit is created.

Arguments
---------
PSP_rel
    Relative standard deviation of the evoked postsynaptic potential.
number_of_pop
    Number of populations in the microcircuit.

Returns
-------
std_mat
    Matrix of the standard deviation of postsynaptic potentials.

"""

function get_std_PSP_matrix(PSP_rel, number_of_pop)
    dim = number_of_pop
    std_mat = zeros((dim, dim))
    std_mat[:, :] = PSP_rel
    return std_mat


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
            neurons[k] = AdEx(N = v, param=LKD2014SingleExp.AdEx, name=string(k))
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

    function j_from_name(pre, post, g)
        if occursin("E", String(pre)) && occursin("E", String(post))
            return g.jee
        elseif occursin("I", String(pre)) && occursin("E", String(post))
            return g.jei
        elseif occursin("E", String(pre)) && occursin("I", String(post))
            return g.jie
        elseif occursin("I", String(pre)) && occursin("I", String(post))
            return g.jii
        else 
            throw(ArgumentError("Invalid pre-post combination: $pre-$post"))
        end
    end

    

    layer_names = [:E23, :I23, :E4, :I4, :E5, :I5, :E6, :I6]



    total_cortical_thickness = 1500.0
    N_full = [20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948] 
    N_E_total = N_full[0]+N_full[2]+N_full[4]+N_full[6]
    dimensions_3D = Dict(
        "x_dimension"=> 1000,
        "z_dimension"=>  1000,
        "total_cortical_thickness"=> total_cortical_thickness,

        # Have the thicknesses proportional to the numbers of E cells in each layer
        "layer_thicknesses"=> Dict(
        "L23"=> total_cortical_thickness*N_full[0]/N_E_total,
        "L4" => total_cortical_thickness*N_full[2]/N_E_total,
        "L5" => total_cortical_thickness*N_full[4]/N_E_total,
        "L6" => total_cortical_thickness*N_full[6]/N_E_total,
        "thalamus" => 100
        )
    )
    net_dict = Dict(
        "PSP_e"=> 0.15,
        # Relative standard deviation of the postsynaptic potential.
        "PSP_sd"=> 0.1,
        # Relative inhibitory synaptic strength (in relative units).
        "g"=> -4,
        # Rate of the Poissonian spike generator (in Hz).
        "bg_rate"=> 8.,
        # Turn Poisson input on or off (True or False).
        "poisson_input"=> True,
        # Delay of the Poisson generator (in ms).
        "poisson_delay"=> 1.5,
        # Mean delay of excitatory connections (in ms).
        "mean_delay_exc"=> 1.5,
        # Mean delay of inhibitory connections (in ms).
        "mean_delay_inh"=> 0.75,
        # Relative standard deviation of the delay of excitatory and
        # inhibitory connections (in relative units).
        "rel_std_delay"=> 0.5,
    )
    updated_dict = Dict(
        # PSP mean matrix.
        "PSP_mean_matrix"=> get_mean_PSP_matrix(
            net_dict['PSP_e'], net_dict['g'], len(layer_names)
            ),
        # PSP std matrix.
        "PSP_std_matrix"=> get_std_PSP_matrix(
            net_dict['PSP_sd'], len(layer_names)
            ),
        # mean delay matrix.
        "mean_delay_matrix"=> get_mean_delays(
            net_dict['mean_delay_exc'], net_dict['mean_delay_inh'],
            len(layer_names)
            ),
        # std delay matrix.
        "std_delay_matrix"=> get_std_delays(
            net_dict['mean_delay_exc'] * net_dict['rel_std_delay'],
            net_dict['mean_delay_inh'] * net_dict['rel_std_delay'],
            len(layer_names])
            ),
    )
    merge!(net_dict, updated_dict)

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

    # !Assign dimensions to these parameters, and some reference
    pree = 0.1
    g = 1.0
    tau_meme = 10.0   # (ms)
    # Synaptic strengths for each connection type
        #=
    K = round(Int, Ne * pree)
    sqrtK = sqrt(K)

    # !Same, the convention is j_post_pre
    je = 2.0 / sqrtK * tau_meme * g
    ji = 2.0 / sqrtK * tau_meme * g 
    jee = 0.15 * je
    jie = je
    jei = -0.75 * ji
    jii = -ji
    =#

    jii = jie = -4.0
    jee = jei = 0.15
    g_strengths = dict2ntuple(SNN.@symdict jee jie jei jii)
    
    conn_j = zeros(Float32, size(conn_probs))
    for pre in eachindex(layer_names)
        for post in eachindex(layer_names)
            conn_j[post, pre ] = j_from_name(layer_names[pre], layer_names[post], g_strengths)
        end
    end

    return layer_names, conn_probs, conn_j
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
    @show exc_pop, Ne, Ni
    layer_names, conn_probs, conn_j = potjans_conn(Ne)

    ## Create the synaptic connections based on the connection probabilities and synaptic weights assigned to each pre-post pair
    connections = Dict()
    for i in eachindex(layer_names)
        for j in eachindex(layer_names)
            pre = layer_names[i]
            post = layer_names[j]
            p = conn_probs[j, i]
            J = conn_j[j, i]
            sym = J>=0 ? :ge : :gi
            μ = abs(J)
            s = SNN.SpikingSynapse(neurons[pre], neurons[post], sym; μ = μ, p=p, σ=0)
            connections[Symbol(string(pre,"_", post))] = s
        end
    end

    #'full_mean_rates':
    full_mean_rates = [0.971, 2.868, 4.746, 5.396, 8.142, 9.078, 0.991, 7.523]
    stimuli = Dict()
    for (ind,pop) in enumerate(exc_pop)
        νe = full_mean_rates[ind]kHz
        post = neurons[pop]
        s = SNN.PoissonStimulus(post, :ge; param = νe, cells=:ALL, μ=1.f0, name="PoissonE_$(post.name)")
        stimuli[Symbol(string("PoissonE_", pop))] = s
    end
    return merge_models(neurons,connections, stimuli),neurons,connections,stimuli
end

if !isfile("cached_potjans_model.jld2")
    model,neurons_,connections_,stimuli_ = potjans_layer(0.125)
    @save "cached_potjans_model.jld2" model neurons_ connections_ stimuli_
else
    @load "cached_potjans_model.jld2" model neurons_ connections_ stimuli_
end
duration = 15000ms
SNN.monitor([model.pop...], [:fire])
SNN.monitor([model.pop...], [:v], sr=200Hz)
SNN.sim!(model=model; duration = duration, pbar = true, dt = 0.125)
display(SNN.raster(model.pop, [0s, 15s]))
savefig("without_stimulus.png")





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
ms = 1,  # Marker size
ylabel = "Time (ms)",
xlabel = "Neuron Index",
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
stdp_param = STDPParameter(A_pre =5e-1, 
                           A_post=-5e-1,
                           τpre  =20ms,
                           τpost =15ms)

syn = SpikingSynapse( st, neurons_[:E4], nothing, w = w,  param = stdp_param)
model2 = merge_models(pop=[st,model], stim=[stim,stimuli_], syn=[syn,connections_], silent=false)

duration = 15000ms
SNN.monitor([model2.pop...], [:fire])
SNN.monitor([model2.pop...], [:v], sr=200Hz)
SNN.sim!(model=model2; duration = duration, pbar = true, dt = 0.125)
display(SNN.raster(model2.pop, [0s, 15s]))
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

##

    #Tonic_NMNIST_Stimulus.spike_chara
        #labels = cached_spikes[5][:]
        #unique_labels = unique(labels)
        #labeled_packets = Dict()
        #for filter_label in unique_labels
        #    population_code,time_and_offset = Tonic_NMNIST_Stimulus.spike_packets_by_labels(filter_label,cached_spikes)
        #    labeled_packets[filter_label] = (population_code,time_and_offset)
        #    Tonic_NMNIST_Stimulus.spike_character_plot(filter_label,population_code,time_and_offset)
        
        #end
        
        #=
        labels = cached_spikes[5][:]
        unique_labels = unique(labels)
        labeled_packets = Dict()
        for filter_label in unique_labels
            population_code,time_and_offset = Tonic_NMNIST_Stimulus.spike_packets_by_labels(filter_label,cached_spikes)
            labeled_packets[filter_label] = (population_code,time_and_offset)
            Tonic_NMNIST_Stimulus.spike_character_plot(filter_label,population_code,time_and_offset)
        
        end
        =#
        #scatter_plot(filter_label,population_code,time_and_offset)



#array_size = maximum(unique_neurons)
#w = zeros(Float32, array_size,array_size)

#w[population_code, population_code] = 1f0

#stparam = SpikeTimeStimulusParameter(time_and_offset,population_code)
#SpikeTimeStimulus(stim_size,neurons,param=stparam)
#=
function times_x_neurons(spikes::Spiketimes)
	all_times = Dict{Float64, Vector{Int}}()
	for n in eachindex(spikes)
		if !isempty(spikes[n])
			for tt in spikes[n]
				# tt = round(Int,t*1000/dt) ## from seconds to timesteps
				if haskey(all_times,tt)
					push!(all_times[tt], n)
				else
					push!(all_times, tt=>[n])
				end
			end
		end
	end
	return collect(keys(all_times)), collect(values(all_times))
end
=#
#neurons = convert(Vector{Int64},population_code)


#stimulus_spikes_dist_neurons = cached_spikes[6][:]
#neurons = convert(Vector{Int64},stimulus_spikes_dist_neurons)

#times = cached_spikes[3][:]

#inputs = SpikeTime(time_and_offset,neurons)
#stim_size = length(unique(stimulus_spikes_dist_neurons))
#display(plot(scatter(neurons,times)))
#stparam = SpikeTimeStimulusParameter(times,stimulus_spikes_dist_neurons)
#SpikeTimeStimulus(stim_size,neurons,param=stparam)

#=
Extreme levels of fatigue excerbated by medication.

ΔTs = -100:1:100ms
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
    syn = SpikingSynapse( st, st, nothing, w = w,  param = stdp_param)
    model = merge_models(pop=st, stim=stim, syn=syn, silent=true)
    SNN.monitor(model.pop..., [:fire])
    SNN.monitor(model.syn..., [:tpre, :tpost])
    train!(model=model, duration=3000ms, dt=0.1ms)
    ΔWs[i] = model.syn[1].W[1] - 1
end
=#

#SpikingNeuralNetworks.SpikeTimeStimulus
#@show(help(SpikingNeuralNetworks.stimulate))
