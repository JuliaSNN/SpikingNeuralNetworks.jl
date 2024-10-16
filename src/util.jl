function connect!(c, j, i, σ = 1e-6)
    W = sparse(c.I, c.J, c.W, length(c.rowptr) - 1, length(c.colptr) - 1)
    W[i, j] = σ * randn(Float32)
    c.rowptr, c.colptr, c.I, c.J, c.index, c.W = dsparse(W)
    # c.tpre, c.tpost, c.Apre, c.Apost = zero(c.W), zero(c.W), zero(c.W), zero(c.W)
    return nothing
end

function model(elements)
    elements = isa(elements, Array) ? elements : [elements]
    P = vcat(map(elements) do e
        e.pop
    end)
    C = vcat(map(elements) do e
        e.syn
    end)
    return P, C
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


function merge_models(kwargs)
    populations = Dict{String, Any}()
    synapses = Dict{String,Any}()
    for (k,v) in kwargs
        @assert haskey(v, :pop) && haskey(v, :syn) "Each element must have a :pop and :syn field"
        for (k1) in keys(v.pop)
            push!(populations, "$(k)_$(k1)" => getfield(v.pop, k1))
        end
        for (k1) in keys(v.syn)
            push!(synapses, "$(k)_$(k1)" => getfield(v.syn, k1))
        end
        if haskey(v, :norm) 
            if isa(v.norm, NamedTuple)
                for (k1) in keys(v.norm)
                    push!(synapses, "$(k)_$(k1)" => getfield(v.syn, k1))
                end
            else
                push!(synapses, "$(k)_norm" => getfield(v.syn, :norm))
            end

        end
    end
    pop = DrWatson.dict2ntuple(populations)
    syn = DrWatson.dict2ntuple(synapses)
    return (pop = pop, syn = syn)
end

export connect!, model, dsparse, record!, monitor, getrecord, clear_records, clear_monitor, merge_models