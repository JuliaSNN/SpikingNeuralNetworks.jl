

"""
    graph(model)

Generate a graph representation of the model.

## Arguments
- `model`: The model to generate the graph from.

## Returns
A `MetaGraphs.MetaDiGraph` object representing the graph.

## Details
- Each vertex represents either a population ('pop'), a normalization synapse ('norm'), or a stimulus pre-target ('pre'). 
  Its metadata includes:
    - `name`: Actual name of the population, 'norm' for a SynapseNormalization, or the pre-target's name for a stimulus.
    - `id`: Identifier of the population, SynapseNormalization, or stimulus.
    - `key`: Key from the original 'pop', 'syn', or 'stim' dictionary in the model.

- Each edge represents a synaptic connection or a stimulus. 
  Its metadata includes:
    - `type`: Type of the edge, ':fire_to_g' for SpikingSynapse, ':norm' for SynapseNormalization, or ':stim' for a stimulus.
    - `name`: Name of the edge, formatted as "from_vertex_name to to_vertex_name".
    - `key`: Key from the original 'syn' or 'stim' dictionary in the model.
    - `id`: Identifier of the synapse or stimulus.    
    
- The function iterates over the populations, synapses, and stimuli in the model.

`AbstractPopulation` items are added as vertices.

For each connection it checks the type of the synapse and adds an edge between the pre-synaptic population and the post-synaptic population. 
    - `SNN.SpikingSynapse`: the edge represents a connection from the firing population to the receiving population.   
    - `SNN.SynapseNormalization`: the edge represents a normalization of synapses between populations.
    - `SNN.PoissonStimulus`: the edge represents a stimulus from the pre-synaptic population to the post-synaptic population.

For each stimulus, it adds a vertex to the graph representing an implicit pre-synaptic population [:fire] an edge between it and the post-synaptic population [:g].

Returns a MetaGraphs.MetaDiGraph where:

# Errors
Throws ArgumentError when the synapse type is neither SNN.SpikingSynapse nor SNN.SynapseNormalization.
"""
function graph(model)
    graph = MetaGraphs.MetaDiGraph()
    @unpack pop, syn, stim = model
    norms = Dict()
    for (k, pop) in pairs(pop)
        name = pop.name
        id = pop.id
        add_vertex!(graph, Dict(:name => name, :id => id, :key => k))
    end
    for (k, syn) in pairs(syn)
        if isa(syn, SNN.SpikingSynapse)
            pre = syn.targets[:fire]
            post = syn.targets[:g]
            pre_node = find_id_vertex(graph, pre)
            post_node = find_id_vertex(graph, post)
            pre_name = get_prop(graph, pre_node, :name)
            post_name = get_prop(graph, post_node, :name)
            syn_name = "$(pre_name) to $(post_name)"
            add_edge!(graph, pre_node, post_node, Dict(:type => :fire_to_g, :name => syn_name, :key => k, :id => syn.id, :norm => nothing))
        elseif isa(syn, SNN.SynapseNormalization)
            push!(norms, k=>syn)
        else
            throw(ArgumentError("Only SpikingSynapse is supported"))
        end
    end
    for (k, stim) in pairs(stim)
        pre = stim.targets[:pre]
        add_vertex!(graph, Dict(:name => "$pre", :id => stim.id, :key => k))
        pre_node = find_id_vertex(graph, stim.id)
        post = stim.targets[:g]
        post_node = find_id_vertex(graph, post)
        pre_name = get_prop(graph, pre_node, :name)
        post_name = get_prop(graph, post_node, :name)
        stim_name = "$(pre_name) to $(post_name)"
        add_edge!(graph, pre_node, post_node, Dict(:type => :stim, :name => stim_name, :key => k, :id => stim.id))
    end
    for (k,v) in norms
        for id in v.targets[:synapses]
            e = find_id_edge(graph, id)
            props(graph, e.src, e.dst )[:norm] = k
        end
        # # t = join(targets, ", ")
        # # syn_name = "normalize: $t"
        # syn_group = "Normalization_$n"
        # # add_edge!(graph, pre_node, post_node, Dict(:type => :norm, :name => syn_name, :key => k, :id => syn.id))
    end

    return graph
end

function filter_first_vertex(g::AbstractMetaGraph, fn::Function)
    for v in vertices(g)
        fn(g, v) && return v
    end
    error("No vertex matching conditions found")
end

function filter_first_edge(g::AbstractMetaGraph, fn::Function)
    for e in edges(g)
        fn(g, e) && return e
    end
    error("No edge matching conditions found")
end

function find_id_vertex(g::AbstractMetaGraph, id)
    v = filter_first_vertex(g, (g, v) -> get_prop(g, v, :id) == id)
    isnothing(v) && error("Vertex not found")
    return v
end

function find_id_edge(g::AbstractMetaGraph, id)
    v = filter_first_edge(g, (g, v) -> get_prop(g, v, :id) == id)
    isnothing(v) && error("Vertex not found")
    return v
end

function find_key_graph(g::AbstractMetaGraph, id)
    v = filter_first_vertex(g, (g, v) -> get_prop(g, v, :key) == id)
    isnothing(v) && isnothing(e) && error("Vertex or edge not found")
    return insothing(v) ? e : v
end


export graph