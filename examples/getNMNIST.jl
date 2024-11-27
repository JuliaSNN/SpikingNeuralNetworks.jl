using JLD2



function bds!()

    pushfirst!(PyVector(pyimport("sys")."path"), "")
    nmnist_module = pyimport("batch_nmnist_motions")
    dataset::PyObject = nmnist_module.NMNIST("./")
    training_order = shuffle(0:dataset.get_count()-1)
    #storage = Array{Array}([])
    storage::Array{Tuple{Vector{UInt32}, Vector{UInt32}, Vector{Float32}, Vector{Int8}, Vector{UInt32}, Vector{Any}}} = []
    storage = []
    input_shape = dataset.get_element_dimensions()
    cnt = 0
    
    @time @inbounds for batch in 1:100:length(training_order)
        if cnt<1400

            events = dataset.get_dataset_item(training_order[batch:batch+100-1])
            cnt,did_it_exec = build_data_set_native(Odesa.ConvOdesa,events,storage,cnt,input_shape)
            push!(storage,did_it_exec)
        end
    end
    #@load "yeshes_events.jld" events_cache input_shape

    @save "all_mnmist.jld" storage

    
end


function expected_spike_format(empty_spike_cont,nodes1,times1,maxt)
    nodes1 = [i+1 for i in nodes1]

    @inbounds for i in collect(1:1220)
        @inbounds for (neuron, t) in zip(nodes1,times1)
            if i == neuron
                push!(empty_spike_cont[Int32(i)],Float32(t)+Float32(maxt))
            end            
        end
    end
    empty_spike_cont,minimum(empty_spike_cont),maximum(empty_spike_cont)
end

function NMNIST_pre_process_spike_data(temp_container_store;duration=25)
    spike_packet_lists = Vector{Any}([])
    labelsl = Vector{Any}([])
    packet_window_boundaries = Vector{Any}([])
    maxt = 0
    empty_spike_cont =  []
    @inbounds for i in collect(1:1220)
        push!(empty_spike_cont,[])
    end
    cnt = 0
    @inbounds @showprogress for (ind,s) in enumerate(temp_container_store)
        (times,labels,nodes) = (s[1],s[2],s[3]) 
        maxt = maximum(times)
        if length(times) != 0
            if cnt<duration

                empty_spike_cont,min_,maxt = expected_spike_format(empty_spike_cont,nodes,times,maxt)
                maxt += maxt

                push!(labelsl,labels)
                push!(packet_window_boundaries,(min_,maxt))
                cnt+=1
            end
        end
    end
    return empty_spike_cont,labelsl,packet_window_boundaries
end

#if !isfile("NMNIST_spike_packet_conc_v.jld")

#@time (correct_class,wrong_class,no_class) = 
bds!()

@load "../data2/all_mnmist_complete.jld" storage
empty_spike_cont, labelsl, packet_window_boundaries = NMNIST_pre_process_spike_data(storage;duration=125)
#empty_spike_cont, labelsl, packet_window_boundaries= NMNIST_pre_process_spike_data(storage)
@save "NMNIST_spike_packet_conc_v.jld"  empty_spike_cont labelsl packet_window_boundaries


