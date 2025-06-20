

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

For each stimulus, it adds a vertex to the graph representing an implicit pre-synaptic population [:fire] an edge between it and the post-synaptic population [:post].

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
        if typeof(syn) <: AbstractNormalization
            push!(norms, k => syn)
        elseif haskey(syn.targets, :type)
            pre_id = syn.targets[:fire]
            post_id = syn.targets[:post]
            type = syn.targets[:type]
            add_connection!(graph, pre_id, post_id, k, syn, type)
        else
            throw(ArgumentError("Only SpikingSynapse is supported"))
        end
    end
    for (k, stim) in pairs(stim)
        # verterx and edge for the stimulus have the same id
        pre_id = stim.id
        post_id = stim.targets[:post]
        add_vertex!(graph, Dict(:name => stim.name, :id => pre_id, :key => k))
        type = :fire_to_g
        add_connection!(graph, pre_id, post_id, k, stim, type)
    end
    for (k, v) in norms
        for id in v.targets[:synapses]
            _edges, _ids = filter_edge_props(graph, :id, id)
            for (e, i) in zip(_edges, _ids)
                props(graph, e.src, e.dst)[:norm][i] = k
            end
        end
    end
    return graph
end

function add_connection!(graph, pre_id, post_id, k, syn, type)
    pre_node = find_id_vertex(graph, pre_id)
    post_node = find_id_vertex(graph, post_id)
    pre_name = get_prop(graph, pre_node, :name)
    post_name = get_prop(graph, post_node, :name)
    sym = haskey(syn.targets, :sym) ? syn.targets[:sym] : :soma
    syn_name = "$(syn.name)"
    syn_pop = "$(pre_name) -> $(post_name).$sym"
    id = syn.id
    if !has_edge(graph, pre_node, post_node)
        add_edge!(
            graph,
            pre_node,
            post_node,
            Dict(
                :type => [type],
                :name => [syn_name],
                :pop => [syn_pop],
                :key => [k],
                :id => [id],
                :norm => [:none],
                :target => [sym],
                :count => [1],
                :multi => 1,
            ),
        )
    else
        multi_dict = props(graph, pre_node, post_node)
        _multi = multi_dict[:multi] + 1
        multi_dict[:multi] = _multi
        push!(multi_dict[:name], syn_name)
        push!(multi_dict[:pop], syn_pop)
        push!(multi_dict[:type], type)
        push!(multi_dict[:key], k)
        push!(multi_dict[:id], id)
        push!(multi_dict[:norm], :none)
        push!(multi_dict[:target], sym)
        push!(multi_dict[:count], _multi)
        set_props!(graph, pre_node, post_node, multi_dict)
    end
end

function filter_first_vertex(g::AbstractMetaGraph, fn::Function)
    for v in vertices(g)
        fn(g, v) && return v
    end
    # error("No vertex matching conditions found")
    return nothing
end

function filter_edge_props(g::AbstractMetaGraph, key, value)
    _edges = []
    _ids = []
    for e in edges(g)
        prop = props(g, e)
        for i in eachindex(prop[key])
            if prop[key][i] == value
                push!(_edges, e)
                push!(_ids, i)
            end
        end
    end
    if isempty(_edges)
        # error("No edge matching conditions found")
        return []
    end
    return _edges, _ids
end

function find_id_vertex(g::AbstractMetaGraph, id)
    v = filter_first_vertex(g, (g, v) -> get_prop(g, v, :id) == id)
    isnothing(v) && error("Population $id not found")
    return v
end


function find_key_graph(g::AbstractMetaGraph, id)
    v = filter_first_vertex(g, (g, v) -> get_prop(g, v, :key) == id)
    isnothing(v) && isnothing(e) && error("Vertex or edge not found")
    return insothing(v) ? e : v
end


export graph
