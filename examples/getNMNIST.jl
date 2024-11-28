using JLD2
using ProgressMeter
using PyCall
using Random



function build_data_set_native(events,cnt,input_shape)#,l_change_cnt,l_old)
    xx = Vector{Int32}([])
    yy = Vector{Int32}([])
    tts = Vector{Float32}([])
    polarity = Vector{Int8}([])
    label = Vector{Int32}([])
    #A = zeros(input_shape[1]+1,input_shape[2]+1,input_shape[3]+1)
    A = zeros((400,400))
    I = LinearIndices(A)
    pop_stimulation= Vector{Int32}([])#Vector{UInt32}([])
    @inbounds for (ind_,ev) in enumerate(events)      
        cnt+=1
        (x,y,ts,p,l) = ev
        #@show(x,y,ts,p,l)
        #rows, cols = size(pop_stimulation)
        #if 1 <= x <= rows && 1 <= y <= cols
            #index = Int32(I[CartesianIndex(Int(x), Int(y))])
        #if 1 <= x && 1 <= y            
        #if 1 <= x  && 1 <= y 
        push!(pop_stimulation,Int32(I[CartesianIndex(convert(Int32,x),convert(Int32,y))]))

        #else
        #    @warn "Index out of bounds: x=$x, y=$y"
        #end

        #push!(pop_stimulation,Int32(I[CartesianIndex(convert(Int32,x),convert(Int32,y))]))
        push!(xx,convert(Int32,x))
        push!(yy,convert(Int32,y))
        ts = Float32(convert(Float32,ts)/1000.0)
        push!(tts,ts)
        push!(polarity,convert(Int8,p))
        l = convert(Int32,l)
        push!(label,l)
    end
    spikes_typed::Tuple{Vector{Int32}, Vector{Int32}, Vector{Float32}, Vector{Int8}, Vector{Int32},Vector{Any}} = (xx,yy,tts,polarity,label,pop_stimulation)
    (cnt,spikes_typed)#,l_change_cnt,l_old)
end

function getNMNIST()

    pushfirst!(PyVector(pyimport("sys")."path"), "")
    nmnist_module = pyimport("batch_nmnist_motions")
    dataset::PyObject = nmnist_module.NMNIST("./")
    training_order = shuffle(0:dataset.get_count()-1)
    #cached_spikes::Array{Tuple{Vector{UInt32}, Vector{UInt32}, Vector{Float32}, Vector{Int8}, Vector{UInt32}, Vector{Any}}} = []
    cached_spikes = []
    input_shape = dataset.get_element_dimensions()
    cnt = 0    
    @time @inbounds for batch in 1:100:length(training_order)
        if cnt<400

            events = dataset.get_dataset_item(training_order[batch:batch+100-1])
            cnt,batch_nmnist = build_data_set_native(events,cnt,input_shape)
            push!(cached_spikes,batch_nmnist)
        end
    end
    @save "partial_mnmist.jld" cached_spikes    
    cached_spikes
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

cached_spikes = getNMNIST()

@load "partial_mnmist.jld" cached_spikes
empty_spike_cont, labelsl, packet_window_boundaries = NMNIST_pre_process_spike_data(cached_spikes;duration=1250)
@save "NMNIST_spike_packet_conc_v.jld"  empty_spike_cont labelsl packet_window_boundaries


