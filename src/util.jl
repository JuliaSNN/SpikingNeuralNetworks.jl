function connect!(c, j, i, μ = 1e-6)
    W = matrix(c)
    W[i, j] = μ * randn(Float32)
    replace_sparse_matrix!(c, W)
    return nothing
end

function matrix(c::C) where C <: AbstractConnection
    return sparse(c.I, c.J, c.W, length(c.rowptr) - 1, length(c.colptr) - 1)
end


function matrix(c::C, sym::Symbol) where C <: AbstractConnection
    return sparse(c.I, c.J, getfield(c,sym), length(c.rowptr) - 1, length(c.colptr) - 1)
end


function update_weights!(c::C, j, i, w) where C <: AbstractConnection
    @unpack colptr, I,  W = c
    for s in colptr[j]:(colptr[j+1]-1)
        if I[s] == i
            W[s] = w
            break
        end
    end
end

function update_weights!(c::C, js::Vector, is::Vector, w) where C <: AbstractConnection
    @unpack colptr, I,  W = c
    for j in js
        for s in colptr[j]:(colptr[j+1]-1)
            if I[s] ∈ is
                W[s] = w
            end
        end
    end
end

function indices(c::C, js::AbstractVector, is::AbstractVector) where C <: AbstractConnection
    @unpack colptr, I,  W = c
    indices = Int[]
    for j in js
        for s in colptr[j]:(colptr[j+1]-1)
            if I[s] ∈ is
                push!(indices, s)
            end
        end
    end
    return indices
end

function set_plasticity!(synapse::AbstractConnection, bool::Bool )
        synapse.param.active[1] = bool
end
function has_plasticity(synapse::AbstractConnection)
        synapse.param.active[1] |> Bool
end

function replace_sparse_matrix!(c::S, W::SparseMatrixCSC) where S <: AbstractConnection
    rowptr, colptr, I, J, index, W = dsparse(W)
    @assert length(rowptr) == length(c.rowptr) "Rowptr length mismatch"
    @assert length(colptr) == length(c.colptr) "Colptr length mismatch"

    resize!(c.I, length(I))
    resize!(c.J, length(I))
    resize!(c.W, length(I))
    resize!(c.index, length(I))

    @assert length(c.I) == length(c.J) == length(c.index) == length(c.W) == length(I) == length(J) == length(index) == length(W) "Length mismatch"
    
    @inbounds @simd for i in eachindex(I)
        c.I[i] = I[i]
        c.J[i] = J[i]
        c.W[i] = W[i]
        c.index[i] = index[i]
    end
    return nothing
end

# """function dsparse

function sparse_matrix(w, Npre, Npost, dist, μ, σ, ρ)
    if isnothing(w)
        # if w is not defined, construct a random sparse matrix with `dist` with `μ` and `σ`. 
        w = rand(dist(μ, σ), Npost, Npre) # Construct a random dense matrix with dimensions post.N x pre.N
        w[[n for n in eachindex(w[:]) if rand() > ρ]] .= 0
        w[w .< 0] .= 0 
        w = sparse(w)
    else
        # if w is defined, convert it to a sparse matrix
        w = sparse(w)
    end
    @assert size(w) == (Npost, Npre) "The size of the synaptic weight is not correct"
    return w
end

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
function merge_models(args...; name=randstring(10), silent=false, kwargs...)
    pop = Dict{Symbol, Any}()
    syn = Dict{Symbol, Any}()
    stim= Dict{Symbol, Any}()
    for v in args
        v isa String && continue
        extract_items(Symbol(""),v, pop=pop, syn=syn, stim=stim)
    end
    for (k,v) in kwargs
        v isa String && continue
        extract_items(k,v, pop=pop, syn=syn, stim=stim)
    end
    pop = DrWatson.dict2ntuple(sort(pop, by =x->x))
    syn = DrWatson.dict2ntuple(sort(syn, by =x->x))
    stim = DrWatson.dict2ntuple(sort(stim, by =x->stim[x].name))
    name = haskey(kwargs, :name) ? args.name : name
    model = (pop=pop, syn=syn, stim=stim, name=name)
    if !silent
        print_model(model)
    end
    return model
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
function print_model(model, get_keys=false)
    model_graph = graph(model)
    @unpack pop, syn, stim = model
    populations = Vector{String}()
    for k in keys(pop)
        v = filter_first_vertex(model_graph, (g, v) -> get_prop(model_graph, v, :key) == k)
        name = props(model_graph, v)[:name]
        _k = get_keys ? "($k)" : ""
        @assert typeof(getfield(pop, k)) <: SNN.AbstractPopulation "Expected neuron, got $(typeof(getfield(network.pop,k)))"
        push!(populations,  "$name $(_k): $(nameof(typeof(getfield(pop,k)))): $(nameof(typeof(getfield(pop,k).param)))")

    end
    synapses = Vector{String}()
    for k in keys(syn)
        isa(syn[k], SNN.SynapseNormalization) && continue
        _edges, _ids = filter_edge_props(model_graph, :key, k)
        for (e, i) in zip(_edges, _ids)
            name = props(model_graph, e)[:name][i]
            _k = get_keys ? "($k)" : ""
            norm = props(model_graph, e)[:norm][i] !== :none ? "<-> norm: $(props(model_graph, e)[:norm][i])" : ""
            # @info "$name $(_k) $norm: \n $(nameof(typeof(getfield(syn,k)))): $(nameof(typeof(getfield(syn,k).param)))"
            @assert typeof(getfield(syn, k)) <: SNN.AbstractConnection "Expected synapse, got $(typeof(getfield(network.syn,k)))"
            push!(synapses,"$name $(_k) $norm: $(nameof(typeof(getfield(syn,k).param)))")
        end
    end
    stimuli = Vector{String}()
    for k in keys(stim)
        _edges, _ids = filter_edge_props(model_graph, :key, k)
        for (e, i) in zip(_edges, _ids)
            name = props(model_graph, e)[:name][i]
            _k = get_keys ? "($k)" : ""
            # @info "$name $(_k): $(nameof(typeof(getfield(stim,k)))): $(nameof(typeof(getfield(stim,k).param)))"
            @assert typeof(getfield(stim, k)) <: SNN.AbstractStimulus "Expected stimulus, got $(typeof(getfield(network.stim,k)))"
            push!(stimuli, "$name $(_k): $(nameof(typeof(getfield(stim,k))))")
        end
    end
    sort!(stimuli)
    sort!(synapses)
    sort!(populations)

    @info "================"
    @info "Model: $(model.name)"
    @info "----------------"
    @info "Populations ($(length(populations))):"
    for p in populations
        @info p
    end
    @info "----------------"
    @info "Synapses ($(length(synapses))): "
    for s in synapses
        @info s
    end
    @info "----------------"
    @info "Stimuli ($(length(stimuli))):"
    for s in stimuli
        @info s
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
            k == :name && continue
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


function load_data(path="", name=nothing, info=nothing)
    isfile(path) && (return dict2ntuple(DrWatson.load(path)))
    if isnothing(name)
        throw(ArgumentError("$path is not file, config is required"))
    end
    path = joinpath(path, savename(name, info, "data.jld2", connector="-"))
    @info "Loading model from $(path)"
    tic = time()
    DATA = DrWatson.load(path)
    @info "Data loaded in $(time()-tic) seconds"
    return dict2ntuple(DATA)
end

# load_data(path=""; name="", info=nothing) = load_data(path, name, info)

function load_model(path="", name=nothing, info=nothing)
    isfile(path) && (return dict2ntuple(DrWatson.load(path)))
    if isnothing(name)
        throw(ArgumentError("If path is not file, config is required"))
    end
    path = joinpath(path, savename(name, info, "model.jld2", connector="-"))
    tic = time()
    @info "Loading model from $(path)"
    DATA = DrWatson.load(path)
    @info "Model loaded in $(time()-tic) seconds"
    return dict2ntuple(DATA)
end
export load_data, load_model, save_model, savemodel

function save_model(;path, model, name=randstring(10),info=nothing, kwargs...)
    @info "Model: `$(savename(name, info, connector="-"))` \nsaved at $(path)"
    isdir(path) || mkpath(path)

    data_path = joinpath(path, savename(name, info, "data.jld2", connector="-"))
    Logging.LogLevel(0) == Logging.Error
    @time DrWatson.save(data_path, merge((@strdict model=model), kwargs))
    Logging.LogLevel(0) == Logging.Info
    @info "-> Data ($(filesize(data_path) |> Base.format_bytes))"

    _model = deepcopy(model)
    clear_monitor(_model.pop)
    clear_monitor(_model.syn)
    clear_monitor(_model.stim)

    model_path = joinpath(path, savename(name, info, "model.jld2", connector="-"))
    Logging.LogLevel(0) == Logging.Error
    @time DrWatson.save(model_path, merge((@strdict model=_model), kwargs))
    Logging.LogLevel(0) == Logging.Info
    @info "-> Model ($(filesize(model_path) |> Base.format_bytes))"
    return data_path
end

function save_parameters(;path, parameters, name=randstring(10),info=nothing, file_path, force=false)
    @info "Parameters: `$(savename(name, info, connector="-"))` \nsaved at $(path)"

    isdir(path) || mkpath(path)

    params_path = joinpath(path, savename(name, info, "params.jld2", connector="-"))
    DrWatson.save(params_path, @strdict parameters)  # Here you are saving a Julia object to a file

    params_path = joinpath(path, savename(name, info, "params.jl.script", connector="-"))
    isfile(params_path) && !force && throw("File already exists, use force=true to overwrite")
    !isfile(params_path) &&  cp(file_path, params_path)
    return 
end

export save_model, load_model, load_data, save_parameters

   

"""
    print_summary(p)

    Prints a summary of the given element.
"""
function print_summary(p)
    println("Type: $(nameof(typeof(p))) $(nameof(typeof(p.param)))")
    println("  Name: ", p.name)
    println("  Number of Neurons: ", p.N)
    for k in fieldnames(typeof(p.param))
        println("   $k: $(getfield(p.param,k))")
    end
end




export connect!,
    model, dsparse, record!, monitor, getrecord, clear_records, clear_monitor, merge_models, remove_element, graph, matrix,  print_model,  extract_items, sparse_matrix, replace_sparse_matrix!, exp32, exp256, print_summary, update_weights!, indices, set_plasticity!, has_plasticity
