function connect!(c, j, i, σ = 1e-6)
    W = sparse(c.I, c.J, c.W, length(c.rowptr) - 1, length(c.colptr) - 1)
    W[i, j] = σ * randn(Float32)
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

## Returns
A tuple `(pop, syn)` representing the merged populations and synapses.

## Details
This function takes in multiple models represented as keyword arguments and merges them into a single model. The models can be specified using the `pop` and `syn` fields in the keyword arguments. If the `pop` and `syn` fields are not present, the function expects the keyword arguments to have elements with `:pop` or `:syn` fields.

The merged populations and synapses are stored in dictionaries `populations` and `synapses`, respectively. The function performs type assertions to ensure that the elements being merged are of the correct types (`AbstractPopulation` for populations and `AbstractConnection` for synapses).

If `syn` and/or `pop` arguments are provided, they are merged into the respective dictionaries.

## Example
"""
function merge_models(kwargs...; syn = nothing, pop = nothing)
    populations = Dict{String,Any}()
    synapses = Dict{String,Any}()
    for kwarg in kwargs
        if haskey(kwarg, :pop) && haskey(kwarg, :syn)
            for (k) in keys(kwarg.pop)
                @assert typeof(kwarg.pop[k]) <: AbstractPopulation "$(typeof(pop[k])) is not a population"
                push!(populations, "$(k)" => getfield(kwarg.pop, k))
            end
            for (k) in keys(kwarg.syn)
                @assert typeof(kwarg.syn[k]) <: AbstractConnection "$(typeof(syn[k])) is not a synapse"
                push!(synapses, "$(k)" => getfield(kwarg.syn, k))
            end
        else
            for k in keys(kwarg)
                v = kwarg[k]
                @assert haskey(v, :pop) || haskey(v, :syn) "$k element must have a :pop or :syn field"
                if haskey(v, :pop)
                    for (k1) in keys(v.pop)
                        @assert typeof(getfield(v.pop, k1)) <: AbstractPopulation "$(typeof(getfield(v.pop, k1))) is not a population"
                        push!(populations, "$(k)_$(k1)" => getfield(v.pop, k1))
                    end
                end
                if haskey(v, :syn)
                    for (k1) in keys(v.syn)
                        @assert typeof(getfield(v.syn, k1)) <: AbstractConnection "$(typeof(getfield(v.syn, k1))) is not a synapse"
                        push!(synapses, "$(k)_$(k1)" => getfield(v.syn, k1))
                    end
                end
            end
        end
    end
    if !isnothing(syn)
        for k in keys(syn)
            @assert typeof(syn[k]) <: AbstractConnection "$(typeof(syn[k])) is not a synapse"
            push!(synapses, k => syn[k])
        end
    end
    if !isnothing(pop)
        for k in keys(pop)
            @assert typeof(pop[k]) <: AbstractPopulation "$(typeof(pop[k])) is not a population"
            push!(populations, k => pop[k])
        end
    end
    pop = DrWatson.dict2ntuple(sort(populations))
    syn = DrWatson.dict2ntuple(sort(synapses))
    @info "Merging models"
    @info "Populations"
    for k in keys(pop)
        @info "$(k) => $(typeof(getfield(pop,k)))"
        @assert typeof(getfield(pop, k)) <: SNN.AbstractPopulation "Expected neuron, got $(typeof(getfield(network.pop,k)))"
    end
    @info "Synapses"
    for k in keys(syn)
        @info "$(k) => $(typeof(getfield(syn,k)))"
        @assert typeof(getfield(syn, k)) <: SNN.AbstractConnection "Expected synapse, got $(typeof(getfield(network.syn,k)))"
    end
    return (pop = pop, syn = syn)
end


export connect!,
    model, dsparse, record!, monitor, getrecord, clear_records, clear_monitor, merge_models
