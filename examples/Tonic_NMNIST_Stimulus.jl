module Tonic_NMNIST_Stimulus
    using JLD2
    using PyCall
    using DataFrames

    using Random
    using Plots 
    """
    NMNIST data set is a 3D labelled data set played out onto a 400 by 400 matrix of pixels over time. For any time instant there can be positive and negative spikes.
    Here the 2D pixel matrix is collapsed into 1D so that it the matrix can be more easily projected onto the synapses of a biological network population.
    """

    """
    Call Pythons tonic module to get NMNIST a spiking data set of numerals.
    """

    function getNMNIST(ind)#, cnt, input_shape)
        pushfirst!(PyVector(pyimport("sys")."path"), "")
        nmnist_module = pyimport("batch_nmnist_motions")
        dataset::PyObject = nmnist_module.NMNIST("./")    
        A = zeros((35,35))
        I = LinearIndices(A)
        pop_stimulation= Vector{Int32}([])#Vector{UInt32}([])
        l = -1
        events,l = dataset._dataset[ind]#._dataset(ind)#;limit=15)
        l = convert(Int32,l)
        x = Vector{Int32}([e[1] for e in events])
        y = Vector{Int32}([e[2] for e in events])
        ts = Vector{Float32}([e[3]/100000.0 for e in events])
        p = Vector{Int8}([e[4] for e in events])
        for (x_,y_) in zip(x,y)
            push!(pop_stimulation,Int32(I[CartesianIndex(convert(Int32,x_+1),convert(Int32,y_+1))]))
        end
        (ts,p,l,pop_stimulation)
    end

    function spike_packets_by_labels(cached_spikes)
        ts = [cached_spikes[i][1] for (i,l) in enumerate(cached_spikes)]
        p = [cached_spikes[i][2] for (i,l) in enumerate(cached_spikes)]
        labels = [cached_spikes[i][3] for (i,l) in enumerate(cached_spikes)]
        pop_stimulation = [cached_spikes[i][4] for (i,l) in enumerate(cached_spikes)]
    
        # Create DataFrame
        df = DataFrame(
            ts = ts,
            p = p,
            labels = labels,
            pop_stimulation = pop_stimulation
        )
    
        # Sort the DataFrame by `labels`
        sorted_df = sort(df, :labels)
    
        # Group by `labels` to create indexed/enumerated groups
        grouped_df = groupby(sorted_df, :labels)
        unique_labels = unique(labels)
        labeled_packets = Dict()
        for filter_label in unique_labels
            time_and_offset=[]
            population_code=[]
            next_offset = 0 
            group = grouped_df[filter_label+1]
            for row in eachrow(group)
                append!(population_code,row["pop_stimulation"])
                append!(time_and_offset,next_offset.+row["ts"])
                windowb = minimum(row["ts"])
                windowa = maximum(row["ts"])
                next_offset = windowa-windowb
    
    
            labeled_packets[filter_label] = (population_code,time_and_offset)
    
            end
        end
        labeled_packets 
    end
    
    """
    For a given numeral characters label plot the corresponding spike packets.
    """
    function spike_character_plot(filter_label,population_code,time_and_offset)
        p1 = Plots.scatter()
        scatter!(p1,
        population_code,    
        time_and_offset,
        ms = 1,  # Marker size
        ylabel = "Time (ms)",
        xlabel = "Neuron Index",
        title = "Spiking Activity with Distinct Characters", 
        legend=false
        )
        plot(p1)
        savefig("label$filter_label.png")
    end
end 



