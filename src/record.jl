@snn_kw mutable struct Time
    t::Vector{Float32} = [0.0f0]
    tt::Vector{Int} = [0]
    dt::Float32 = 0.125f0
end

get_time(T::Time)::Float32 = T.t[1]
get_step(T::Time)::Float32 = T.tt[1]

function update_time!(T::Time, dt::Float32)
    T.t[1] += dt
    T.tt[1] += 1
end

function record_plast!(obj::ST, plasticity::PT, key::Symbol, T::Time, indices::Dict{Symbol,Vector{Int}}, name_plasticity::Symbol) where {ST <: AbstractConnection, PT <: PlasticityVariables}
    ind::Vector{Int} = haskey(indices, key) ? indices[key] : collect(eachindex(getfield(plasticity, key)))
    push!(obj.records[name_plasticity][key], getfield(plasticity, key)[ind])
end

function record_fire!(obj::PT, T::Time, indices::Dict{Symbol,Vector{Int}}) where {PT <: AbstractPopulation}
    sum(obj.fire) == 0 && return
    ind::Vector{Int} = haskey(indices, :fire) ? indices[:fire] : collect(eachindex(obj.fire))
    t::Float32 = get_time(T)
    push!(obj.records[:fire][:time], t)
    push!(obj.records[:fire][:neurons], findall(obj.fire[ind]))
end

function record_sym!(obj, key::Symbol, T::Time, indices::Dict{Symbol,Vector{Int}}) 
    ind::Vector{Int} = haskey(indices, key) ? indices[key] : collect(eachindex(getfield(obj,key)))
    push!(obj.records[key], getfield(obj, key)[ind])
end

"""
Store values into the dictionary named `records` in the object given 

# Arguments
- `obj`: An object whose values are to be recorded

"""
function record!(obj, T::Time)
    records::Dict{Symbol,Any} = obj.records
    for key in keys(records)
        (key == :indices) && (continue)
        if key == :fire
            record_fire!(obj, T, records[:indices])
        elseif key == :plasticity
            for name_plasticity in keys(records[:plasticity])
                for p_k in records[:plasticity][name_plasticity]
                    record_plast!(obj, obj.plasticity, p_k, T, records[:indices], name_plasticity)
                end
            end
        elseif key ∈ keys(records[:plasticity])
            continue
        else
            record_sym!(obj, key, T, records[:indices])
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
        ## If the key is a tuple, then the first element is the symbol and the second element is the list of neurons to record.
        if isa(key, Tuple)
            sym, ind = key
            push!(obj.records[:indices], sym => ind)
        else
            sym = key
        end
        ## If the then assign a Spiketimes object to the dictionary `records[:fire]`, add as many empty vectors as the number of neurons in the object as in [:indices][:fire]
        if sym == :fire
            obj.records[:fire] = Dict{Symbol,AbstractVector}(
                :time => Vector{Float32}(),
                :neurons => Vector{Vector{Int}}(),
            )
        ## If the object has the field `sym`, then assign an empty vector of the same type to the dictionary `records`
        elseif hasfield(typeof(obj), sym)
            typ = typeof(getfield(obj, sym))
            obj.records[sym] = Vector{typ}()
        ## If the object `sym` is in :plasticity, then assign an empty vector of the same type to the dictionary `records[:plasticity]
        elseif hasfield(typeof(obj), :plasticity) && has_plasticity_field(obj.plasticity, sym)
            monitor_plast(obj, obj.plasticity, sym)
        else
            @error "Field $sym not found in $(typeof(obj))"
        end
    end
end

function has_plasticity_field(plasticity::T, key) where {T<:PlasticityVariables}
    return hasfield(typeof(plasticity), key)
end

function monitor_plast(obj, plasticity, sym) 
    name =nameof(typeof(plasticity))
    if !haskey(obj.records, :plasticity)
       obj.records[:plasticity] = Dict{Symbol,Vector{Symbol}}()
    end
    if !haskey(obj.records[:plasticity], name)
       obj.records[:plasticity][name] = Vector{Symbol}()
    end
    push!(obj.records[:plasticity][name], sym)
    typ = typeof(getfield(plasticity, sym))
    if !haskey(obj.records, name)
       obj.records[name] = Dict{Symbol,AbstractVector}()
    end
    obj.records[name][sym] = Vector{typ}()
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
    if haskey(p.records, key) 
        p.records[key]
    elseif haskey(p.records[:plasticity], key)
        p.records[:plasticity][sym]
    else
        throw(ArgumentError("The record is not found"))
    end
end

function clear_records(obj)
    function clean(z)
        for (key, val) in z
            if isa(val, Dict) 
                clean(val)
            else
                empty!(val)
            end
        end
    end
    clean(obj.records)
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
