using SparseArrays
using StaticArrays
using ProgressMeter
using LinearAlgebra

"""
This file contains a function stack that creates a network with Potjans and Diesmann wiring likeness in Julia using SpikingNeuralNetworks.jl to simulate
electrical neural network dynamics.
This code draws heavily on the PyNN OSB Potjans implementation code found here:
https://github.com/OpenSourceBrain/PotjansDiesmann2014/blob/master/PyNN/network_params.py#L139-L146
However the translation of this Python PyNN code into performant and scalable Julia was not trivial.
Hard coded Potjans parameters follow.
and then the function outputs adapted Potjans parameters.
"""
function potjans_params(ccu)
    # a cummulative cell count
    #cumulative = Dict{String, Vector{Int64}}()  
    layer_names = @SVector ["23E","23I","4E","4I","5E", "5I", "6E", "6I"] 
    # Probabilities for >=1 connection between neurons in the given populations. 
    # The first index is for the target population; the second for the source population
    #             2/3e      2/3i    4e      4i      5e      5i      6e      6i
    conn_probs = @SMatrix [0.1009  0.1689 0.0437 0.0818 0.0323 0.0     0.0076 0.    
                                    0.1346   0.1371 0.0316 0.0515 0.0755 0.     0.0042 0.    
                                    0.0077   0.0059 0.0497 0.135  0.0067 0.0003 0.0453 0.    
                                    0.0691   0.0029 0.0794 0.1597 0.0033 0.     0.1057 0.    
                                    0.1004   0.0622 0.0505 0.0057 0.0831 0.3726 0.0204 0.    
                                    0.0548   0.0269 0.0257 0.0022 0.06   0.3158 0.0086 0.    
                                    0.0156   0.0066 0.0211 0.0166 0.0572 0.0197 0.0396 0.2252
                                    0.0364   0.001  0.0034 0.0005 0.0277 0.008  0.0658 0.1443 ]


    # https://github.com/shimoura/ReScience-submission/blob/ShimouraR-KamijiNL-PenaRFO-CordeiroVL-CeballosCC-RomaroC-RoqueAC-2017/code/netParams.py
    #= conn_probs = @SMatrix [0.101  0.169 0.044 0.082 0.032, 0.     0.008 0.     0.    
                                    0.135  0.137 0.032, 0.052, 0.075, 0.,     0.004 0.     0.    
                                    0.008  0.006 0.050, 0.135, 0.007, 0.0003, 0.045 0.     0.0983
                                    0.069  0.003 0.079, 0.160, 0.003, 0.,     0.106 0.     0.0619
                                    0.100  0.062 0.051, 0.006, 0.083, 0.373,  0.020 0.     0.    
                                    0.055  0.027 0.026, 0.002, 0.060, 0.316,  0.009 0.     0.    
                                    0.016  0.007 0.021, 0.017, 0.057, 0.020,  0.040 0.225  0.0512
                                    0.036  0.001 0.003, 0.001, 0.028, 0.008,  0.066 0.144  0.0196 ]
    =#
    # hard coded network wiring parameters are manipulated below:

    syn_pol = Vector{Bool}([])
    for (i,syn) in enumerate(layer_names)
        if occursin("E",syn) 
            push!(syn_pol,true)
        else
            push!(syn_pol,false)
        end
    end
    #syn_pol = syn_pol # synaptic polarity vector.
    return (conn_probs,syn_pol)
end


"""
Auxillary method, NB, this acts like the connectome constructor, so change function name to something more meaningful, like construct PotjanAndDiesmon
A mechanism for scaling cell population sizes to suite hardware constraints.
While Int64 might seem excessive when cell counts are between 1million to a billion Int64 is required.
Only dealing with positive count entities so Usigned is fine.
"""
function potjans_constructor(scale::Float64)
	ccu = Dict{String, UInt32}("23E"=>20683,
		    "4E"=>21915, 
		    "5E"=>4850, 
		    "6E"=>14395, 
		    "6I"=>2948, 
		    "23I"=>5834,
		    "5I"=>1065,
		    "4I"=>5479)

	ccu = Dict{String, UInt32}((k,ceil(Int64,v*scale)) for (k,v) in pairs(ccu))
    v_old=1
    K = length(keys(ccu))
    cum_array = []# Vector{Array{UInt32}}(undef,K)

    for (k,v) in pairs(ccu)
        ## update the cummulative cell count
        push!(cum_array,v_old:v+v_old)
        v_old=v+v_old

    end    

    cum_array = SVector{8,Array{UInt32}}(cum_array) # cumulative population counts array.
	Ncells = UInt64(sum([i for i in values(ccu)])+1)
	Ne = UInt64(sum([ccu["23E"],ccu["4E"],ccu["5E"],ccu["6E"]]))
    Ni = UInt64(Ncells - Ne)
    (Ncells, Ne, Ni, ccu, cum_array)
end
"""
The entry point to building the whole Potjans model in SNN.jl
Also some of the current density parameters needed to adjust synaptic gain initial values.
Some of the following calculations and parameters are borrowed from this repository:
https://github.com/SpikingNetwork/TrainSpikingNet.jl/blob/master/src/param.jl
"""
function potjans_layer(scale::Float64)
    (Ncells, Ne, Ni, ccu, cum_array)= potjans_constructor(scale)    
    (conn_probs,syn_pol) = potjans_params(ccu)    

    pree = 0.1
    K = round(Int, Ne*pree)
    sqrtK = sqrt(K)
    g = 1.0
    tau_meme = 10   # (ms)
    je = 2.0 / sqrtK * tau_meme * g
    ji = je#2.0 / sqrtK * tau_meme * g 
    jee = 0.15je 
    jei = je 
    jie = -0.75ji 
    jii = -ji
    #g_strengths = Vector{Float32}([jee,jie,jei,jii])
    Lxx = spzeros(Float32, (Ncells, Ncells))


    cell_index_to_layer::Vector{UInt64} = zeros(Ncells)
    build_matrix_prot!(cell_index_to_layer,jee,jei,jie,jii,Lxx,cum_array,conn_probs,syn_pol)#,g_strengths)
    Lxx,cell_index_to_layer

end
export potjans_layer




"""
This function contains synapse selection logic seperated from iteration logic for readability only.
Used inside the nested iterator inside build_matrix.
Ideally iteration could flatten to support the readability of subsequent code.
"""
function build_matrix_prot!(cell_index_to_layer,jee::Real,jei::Real,jie::Real,jii::Real,Lxx::SparseMatrixCSC{Float32, Int64},cum_array::SVector{8, Array{UInt32}}, conn_probs::StaticArraysCore.SMatrix{8, 8, Float64, 64}, syn_pol::Vector{Bool})#, g_strengths::Vector{Float32})
    # excitatory weights.

    @inbounds @showprogress for (i,v) in enumerate(cum_array)
        @inbounds for (j,v1) in enumerate(cum_array)
            prob = conn_probs[i,j]
            @inbounds for src in v
                cell_index_to_layer[src] = i
                @inbounds for tgt in v1
                    if src!=tgt
                        if rand()<prob
                            syn1 = syn_pol[j]
                            syn0 = syn_pol[i]
                            if syn0==true
                                if syn1==true

                                    setindex!(Lxx,jee, src,tgt)
                                else# meaning if the same as a logic: Inhibitory post synapse  is true                   
                                    setindex!(Lxx,jei, src,tgt)

                                end
                            else syn0==false     
                                if syn1==true

                                    setindex!(Lxx,jie, src,tgt)
                                else
                                    setindex!(Lxx,jii, src,tgt)
                                end
                            end 
                        end
                    end
                end
            end
        end
    end
end


"""
Syntactically necesarry for TrainingSpikeNet package, I am not sure what the reason is I suspect the reason is dense formats.

function build_w0Index(edge_dict,Ncells)
    nc0Max = 0
    # what is the maximum out degree of this ragged array?
    for (k,v) in pairs(edge_dict)
        templength = length(v)
        if templength>nc0Max
            nc0Max=templength
        end
    end
    # outdegree
    nc0 = Int.(nc0Max*ones(Ncells))
    ##
    # Force ragged array into smallest dense rectangle (contains zeros for undefined synapses) 
    ##
    w0Index = zeros(Int64, (nc0Max,Ncells))

    #w0Index = spzeros(Int64,nc0Max,Ncells)
    for pre_cell = 1:Ncells
        post_cells = edge_dict[pre_cell]
        stride_length = length(edge_dict[pre_cell])
        w0Index[1:stride_length,pre_cell] = post_cells
    end
    nc0,w0Index
end
"""
function genStaticWeights(args)
    (edge_dict,w0Weights,Ne,Ni,Lexc,Linh) = potjans_weights(args)
    #dropzeros!(w0Weights)
    Ncells = args[1]
    #nc0,w0Index = build_w0Index(edge_dict,Ncells)
    return w0Index, w0Weights, nc0
end