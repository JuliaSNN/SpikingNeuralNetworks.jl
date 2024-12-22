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


#using Graphs
#using Karnak
#using NetworkLayout
#using Colors

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
            neurons[k] = SNN.AdEx(; N = 1, param = SNN.AdExParameter(; El = -49mV), name=string(k))

        else

            neurons[k] = SNN.IF(; N = 1, param = SNN.IFParameter(; El = -49mV), name=string(k))
        end
    end
    return neurons
end

"""
Define Potjans parameters for neuron populations and connection probabilities
    # Names of the simulated populations.
    'populations': ['L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I'],
    # Number of neurons in the different populations. The order of the
    # elements corresponds to the names of the variable 'populations'.
    'N_full': np.array([20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948]),
    # Mean rates of the different populations in the non-scaled version
    # of the microcircuit. Necessary for the scaling of the network.
    # The order corresponds to the order in 'populations'.
    'full_mean_rates':
        np.array([0.971, 2.868, 4.746, 5.396, 8.142, 9.078, 0.991, 7.523]),
    # Connection probabilities. The first index corresponds to the targets
    # and the second to the sources.
    # Number of external connections to the different populations.
    # The order corresponds to the order in 'populations'.
    'K_ext': np.array([1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100]),
    # Factor to scale the indegrees.
    'K_scaling': 0.1,
    # Factor to scale the number of neurons.
    'N_scaling': 0.1,
    # Mean amplitude of excitatory postsynaptic potential (in mV).
    'PSP_e': 0.15,
    # Relative standard deviation of the postsynaptic potential.
    'PSP_sd': 0.1,
    # Relative inhibitory synaptic strength (in relative units).
    'g': -4,
    # Rate of the Poissonian spike generator (in Hz).
    'bg_rate': 8.,
    # Turn Poisson input on or off (True or False).
    'poisson_input': True,
    # Delay of the Poisson generator (in ms).
    'poisson_delay': 1.5,
    # Mean delay of excitatory connections (in ms).
    'mean_delay_exc': 1.5,
    # Mean delay of inhibitory connections (in ms).
    'mean_delay_inh': 0.75,
    # Relative standard deviation of the delay of excitatory and
    # inhibitory connections (in relative units).
    'rel_std_delay': 0.5,
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
    pree = 0.2
    g = 1.0
    tau_meme = 10.0   # (ms)
    # Synaptic strengths for each connection type
    K = round(Int, Ne * pree)
    sqrtK = sqrt(K)

    # !Same, the convention is j_post_pre
    je = 2.0 / sqrtK * tau_meme * g
    ji = 2.0 / sqrtK * tau_meme * g 
    jee = 0.15 * je
    jie = je
    jei = -0.75 * ji
    jii = -ji
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
    #@show exc_pop, Ne, Ni
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


            #S = SNN.SpikingSynapse(
            #    inputs,
            #    neurons,
            #    :ge;
            #    μ = 0.01,
            #    p = 1.0,
            #    param = SNN.vSTDPParameter(; Wmax = 0.01),
            #)
            
            s = SNN.SpikingSynapse(neurons[pre], neurons[post], sym; μ = μ*10, p=p, σ=0)#,param = SNN.vSTDPParameter(; Wmax = 0.5))
            connections[Symbol(string(pre,"_", post))] = s
        end
    end

    ## Create the Poisson stimulus for each population
    stimuli = Dict()
    for pop in exc_pop
        νe = 3.5kHz
        post = neurons[pop]
        s = SNN.PoissonStimulus(post, :ge; param = νe, cells=:ALL, μ=1.f0, name="PoissonE_$(post.name)")
        stimuli[Symbol(string("PoissonE_", pop))] = s
    end
    return merge_models(neurons,connections, stimuli)
end


model = potjans_layer(0.1)
duration = 15000ms
SNN.monitor([model.pop...], [:fire])
SNN.monitor([model.pop...], [:v])#, sr=200Hz)
SNN.sim!(model=model; duration = duration, pbar = true, dt = 0.125)
display(SNN.raster(model.pop))#, [10s, 15s]))

#Trange = 0:10:15s
frE, interval = SNN.firing_rate(model.pop, interval = Trange)
display(plot(mean.(frE), xlabel="Time [ms]", ylabel="Firing rate [Hz]", legend=:topleft))
##

#vecplot(model.pop.E23, :v, neurons =1, r=0s:15s,label="soma")
layer_names, conn_probs, conn_j = potjans_conn(4000)
pj = heatmap(conn_j, xticks=(1:8,layer_names), yticks=(1:8,layer_names), aspect_ratio=1, color=:bluesreds,  title="Synaptic weights", xlabel="Presynaptic", ylabel="Postsynaptic", size=(500,500), clims=(-maximum(abs.(conn_j)), maximum(abs.(conn_j))))
pprob=heatmap(conn_probs, xticks=(1:8,layer_names), yticks=(1:8,layer_names), aspect_ratio=1, color=:viridis,  title="Connection probability", xlabel="Presynaptic", ylabel="Postsynaptic", size=(500,500))
display(plot(pprob, pj, layout=(1,2), size=(1000,500), margin=5Plots.mm))
##