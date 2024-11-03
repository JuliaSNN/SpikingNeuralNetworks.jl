function connect!(c, j, i, μ = 1e-6)
    W = sparse(c.I, c.J, c.W, length(c.rowptr) - 1, length(c.colptr) - 1)
    W[i, j] = μ * randn(Float32)
    c.rowptr, c.colptr, c.I, c.J, c.index, c.W = dsparse(W)
    # c.tpre, c.tpost, c.Apre, c.Apost = zero(c.W), zero(c.W), zero(c.W), zero(c.W)
    return nothing
end

# """function dsparse

function dsparse(A)
    # them in a special data structure leads to savings in space and execution time, compared to dense arrays.
    At = sparse(A') # Transposes the input sparse matrix A and stores it as At.
    colptr = A.colptr # Retrieves the column pointer array from matrix A
    rowptr = At.colptr # Retrieves the column pointer array from the transposed matrix At
    I = rowvals(A) # Retrieves the row indices of non-zero elements from matrix A
    V = nonzeros(A) # Retrieves the values of non-zero elements from matrix A
    J = zero(I) # Initializes an array J of the same size as I filled with zeros.
    index = zeros(Int, size(I)) # Initializes an array index of the same size as I filled with zeros.


    # FIXME: Breaks when A is empty
    for j = 1:(length(colptr)-1) # Starts a loop iterating through the columns of the matrix.
        J[colptr[j]:(colptr[j+1]-1)] .= j # Assigns column indices to J for each element in the column range.
    end
    coldown = zeros(eltype(index), length(colptr) - 1) # Initializes an array coldown with a specific type and size.
    for i = 1:(length(rowptr)-1) # Iterates through the rows of the transposed matrix At.
        for st = rowptr[i]:(rowptr[i+1]-1) # Iterates through the range of elements in the current row.
            j = At.rowval[st] # Retrieves the column index from the transposed matrix At.
            index[st] = colptr[j] + coldown[j] # Computes an index for the index array.
            coldown[j] += 1 # Updates coldown for indexing.
        end
    end
    # Test.@test At.nzval == A.nzval[index]
    rowptr, colptr, I, J, index, V # Returns the modified rowptr, colptr, I, J, index, and V arrays.
end


@inline function exp32(x::Float32)
    x = ifelse(x < -10.0f0, -32.0f0, x)
    x = 1.0f0 + x / 32.0f0
    x *= x
    x *= x
    x *= x
    x *= x
    x *= x
    return x
end

@inline function exp256(x::Float32)
    x = ifelse(x < -10.0f0, -256.0f0, x)
    x = 1.0f0 + x / 256.0f0
    x *= x
    x *= x
    x *= x
    x *= x
    x *= x
    x *= x
    x *= x
    x *= x
    return x
end



"""

    merge_models(kwargs...; syn=nothing, pop=nothing)

Merge multiple models into a single model.

## Arguments
- `kwargs...`: List of `kwarg` elements, i.e., dictionary or named tuples, containing the models to be merged.
    - if `kwarg` has elements with `:pop` and `:syn` entries, the function copies them into the merged model.
    - if `kwarg` has no `:pop` and `:syn` entries, the function iterates over all the elements contained in `kwarg` and merge them into the model.
- `syn`: Optional dictionary of synapses to be merged.
- `pop`: Optional dictionary of populations to be merged.
- `stim`: Optional dictionary of stimuli to be merged.

## Returns
A tuple `(pop, syn)` representing the merged populations and synapses.

## Details
This function takes in multiple models represented as keyword arguments and merges them into a single model. The models can be specified using the `pop` and `syn` fields in the keyword arguments. If the `pop` and `syn` fields are not present, the function expects the keyword arguments to have elements with `:pop` or `:syn` fields.

The merged populations and synapses are stored in dictionaries `populations` and `synapses`, respectively. The function performs type assertions to ensure that the elements being merged are of the correct types (`AbstractPopulation` for populations and `AbstractConnection` for synapses).

If `syn` and/or `pop` and/or `stim` arguments are provided, they are merged into the respective dictionaries.

## Example
"""
function merge_models(args...;silent=false, kwargs...)
    pop = Dict{Symbol, Any}()
    syn = Dict{Symbol, Any}()
    stim= Dict{Symbol, Any}()
    for v in args
        extract_items(Symbol(""),v, pop=pop, syn=syn, stim=stim)
    end
    for (k,v) in kwargs
        extract_items(k,v, pop=pop, syn=syn, stim=stim)
    end
    pop = DrWatson.dict2ntuple(sort(pop))
    syn = DrWatson.dict2ntuple(sort(syn))
    stim = DrWatson.dict2ntuple(sort(stim))
    if !silent
        print_model((pop=pop, syn=syn, stim=stim))
    end
    return (pop=pop, syn=syn, stim=stim)
end

"""
    print_model(model)

Prints the details of the given model. 
The model is expected to have three components: `pop` (populations), `syn` (synapses), and `stim` (stimuli).

The function displays a graph representation of the model, followed by detailed information about each component.

# Arguments
- `model`: The model containing populations, synapses, and stimuli to be printed.

# Outputs
Prints the graph of the model, along with the name, key, type, and parameters of each component in the populations, synapses, and stimuli.

# Exception
Raises an assertion error if any component in the populations is not a subtype of `SNN.AbstractPopulation`, if any component in the synapses is not a subtype of `SNN.AbstractConnection`, or if any component in the stimuli is not a subtype of `SNN.AbstractStimulus`.

"""
function print_model(model)
    model_graph = graph(model)
    @show model_graph
    @unpack pop, syn, stim = model
    @info "================"
    @info "Model:"
    @info "----------------"
    @info "Populations:"
    for k in keys(pop)
        v = filter_first_vertex(model_graph, (g, v) -> get_prop(model_graph, v, :key) == k)
        name = props(model_graph, v)[:name]
        @info "$name ($k): $(nameof(typeof(getfield(pop,k)))): $(nameof(typeof(getfield(pop,k).param)))"
        @assert typeof(getfield(pop, k)) <: SNN.AbstractPopulation "Expected neuron, got $(typeof(getfield(network.pop,k)))"
    end
    @info "----------------"
    @info "Synapses:"
    for k in keys(syn)
        e = filter_first_edge(model_graph, (g, e) -> get_prop(model_graph, e, :key) == k)
        name = props(model_graph, e)[:name]
        @info "$name ($k): $(nameof(typeof(getfield(syn,k)))): $(nameof(typeof(getfield(syn,k).param)))"
        @assert typeof(getfield(syn, k)) <: SNN.AbstractConnection "Expected synapse, got $(typeof(getfield(network.syn,k)))"
    end
    @info "----------------"
    @info "Stimuli:"
    for k in keys(stim)
        e = filter_first_edge(model_graph, (g, e) -> get_prop(model_graph, e, :key) == k)
        name = props(model_graph, e)[:name]
        @info "$name ($k): $(nameof(typeof(getfield(stim,k)))): $(nameof(typeof(getfield(stim,k).param)))"
        @assert typeof(getfield(stim, k)) <: SNN.AbstractStimulus "Expected stimulus, got $(typeof(getfield(network.stim,k)))"
    end
    @info "================"
end

"""
    extract_items(root::Symbol, container; pop::Dict{Symbol,Any}, syn::Dict{Symbol, Any}, stim::Dict{Symbol,Any})

Extracts items from a container and adds them to the corresponding dictionaries based on their type.

## Arguments
- `root::Symbol`: The root symbol for the items being extracted.
- `container`: The container from which to extract items.
- `pop::Dict{Symbol,Any}`: The dictionary to store population items.
- `syn::Dict{Symbol, Any}`: The dictionary to store synapse items.
- `stim::Dict{Symbol,Any}`: The dictionary to store stimulus items.

## Returns
- `true`: Always returns true.

## Details
- If the type of the item in the container is `AbstractPopulation`, it is added to the `pop` dictionary.
- If the type of the item in the container is `AbstractConnection`, it is added to the `syn` dictionary.
- If the type of the item in the container is `AbstractStimulus`, it is added to the `stim` dictionary.
- If the type of the item in the container is none of the above, the function is recursively called to extract items from the nested container.
"""
function extract_items(root::Symbol, container; pop::Dict{Symbol,Any}, syn::Dict{Symbol, Any}, stim::Dict{Symbol,Any})
    v = container
    if typeof(v) <: AbstractPopulation
        @assert !haskey(pop, root) "Population $(root) already exists"
        push!(pop, root => v)
    elseif typeof(v) <: AbstractConnection
        @assert !haskey(syn, root) "Synapse $(root) already exists"
        push!(syn, root => v)
    elseif typeof(v) <: AbstractStimulus
        @assert !haskey(stim, root) "Stimulus $(root) already exists"
        push!(stim, root => v)
    else
        for k in keys(container)
            v = getindex(container, k)
            (k == :pop || k == :syn || k == :stim) && (extract_items(root, v, pop=pop, syn=syn, stim=stim)) && continue
            new_key = isempty(string(root)) ? k : Symbol(string(root) * "_" * string(k))
            if typeof(v) <: AbstractPopulation
                @assert !haskey(pop, new_key) "Population $(new_key) already exists"
                push!(pop, new_key => v)
            elseif typeof(v) <: AbstractConnection
                @assert !haskey(syn, new_key) "Synapse $(new_key) already exists"
                push!(syn, new_key => v)
            elseif typeof(v) <: AbstractStimulus
                @assert !haskey(stim, new_key) "Stimulus $(new_key) already exists"
                push!(stim, new_key => v)
            else
                extract_items(new_key, v, pop=pop, syn=syn, stim=stim)
            end
        end
    end
    return true
end

function remove_element(model, key)
    pop = Dict(pairs(model.pop))
    syn = Dict(pairs(model.syn))
    stim = Dict(pairs(model.stim))
    if haskey(model.pop, key)
        delete!(pop, key)
    elseif haskey(model.syn, key)
        delete!(syn, key)
    elseif haskey(model.stim, key)
        delete!(stim, key)
    else
        throw(ArgumentError("Element not found"))
    end
    merge_models(pop, syn, stim)
end

export connect!,
    model, dsparse, record!, monitor, getrecord, clear_records, clear_monitor, merge_models, remove_element, graph
