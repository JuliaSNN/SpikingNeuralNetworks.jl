@snn_kw struct Time
    t::Vector{Float32}= [0.0f0]
    tt::Vector{Int} = [0]
end

get_time(T::Time)::Float32 = T.t[1]
get_step(T::Time)::Float32 = T.tt[1]

function update_time!(T::Time, dt::Float32) 
    T.t[1] += dt
    T.tt[1] += 1
end


function record_sym(obj, key, T::Time, ind::Vector{Int})
    ind = isempty(ind) ? collect(axes(getfield(obj,key), 1)) : ind
    if key == :fire
        sum(obj.fire[ind]) == 0 && return
        t = get_time(T)
        push!(obj.records[:fire][:time], t)
        push!(obj.records[:fire][:neurons], [i for i in ind if obj.fire[i]])
    else
        push!(obj.records[key], getindex(getfield(obj, key), ind))
    end
end

function record_sym(obj, key, T::Time)
    if key == :fire
        sum(obj.fire) == 0 && return
        t = get_time(T)
        push!(obj.records[:fire][:time], t)
        push!(obj.records[:fire][:neurons], findall(obj.fire))
    else
        # getindex returns a copy of `getfield(obj, sym)` at the given index `ind`
        push!(obj.records[key], copy(getfield(obj, key)))
    end
end

"""
Store values into the dictionary named `records` in the object given 

# Arguments
- `obj`: An object whose values are to be recorded

"""
function record!(obj, T::Time)
    for key in keys(obj.records)
        (key == :indices) && (continue)
        if haskey(obj.records[:indices], key)
            indices = get(obj.records, :indices, nothing)
            ind = get(indices, key, nothing)
            record_sym(obj, key, T, ind)
        else
            record_sym(obj, key, T)
        end
    end
end

"""
Initialize dictionary records for the given object, by assigning empty vectors to the given keys

# Arguments
- `obj`: An object whose variables will be monitored
- `keys`: The variables to be monitored

"""
function monitor(obj, keys)

    if !haskey(obj.records, :indices)
        obj.records[:indices] = Dict{Symbol,Vector{Int}}()
    end
    for key in keys
        # @info key
        ## If the key is a tuple, then the first element is the symbol and the second element is the list of neurons to record.
        if isa(key, Tuple)
            sym, ind = key
            push!(obj.records[:indices], sym => ind)
        else
            sym = key
        end

        ## If the then assign a Spiketimes object to the dictionary `records[:fire]`, add as many empty vectors as the number of neurons in the object as in [:indices][:fire]
        if sym == :fire
            obj.records[:fire] = Dict{Symbol, AbstractVector}(:time=>Vector{Float32}(), :neurons=>Vector{Vector{Int}}())
        ## If the object has the field `sym`, then assign an empty vector of the same type to the dictionary `records`
        elseif hasfield(typeof(obj), sym)
            typ = typeof(getfield(obj, sym))
            obj.records[sym] = Vector{typ}()
        ## If the object `sym` is in :plasticity, then assign an empty vector of the same type to the dictionary `records[:plasticity]
        elseif hasfield(typeof(obj), :plasticity) && hasfield(typeof(obj.plasticity), sym)
            typ = typeof(getfield(obj.plasticity, sym))
            obj.records[:plasticity] = Dict{Symbol, AbstractVector}()
            obj.records[:plasticity][sym] = Vector{typ}()
        else
            @debug "Field $sym not found in $(typeof(obj))"
        end
    end
end



function monitor(objs::Array, keys)
    """
    Function called when more than one object is given, which then calls the above monitor function for each object
    """
    for obj in objs
        monitor(obj, keys)
    end
end

function getrecord(p, sym)
    key = sym
    for (k, val) in p.records
        isa(k, Tuple) && k[1] == sym && (key = k)
    end
    p.records[key]
end

function clear_records(obj)
    for (key, val) in obj.records
        key == :indices && continue
        empty!(val)
    end
end

function clear_records(obj, sym::Symbol)
    for (key, val) in obj.records
        (key == sym) && (empty!(val))
    end
end

function clear_records(objs::AbstractArray)
    for obj in objs
        clear_records(obj)
    end
end

function clear_monitor(obj)
    for (k, val) in obj.records
        delete!(obj.records, k)
    end
end
