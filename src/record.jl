import Interpolations: scale, interpolate, BSpline, Linear

"""
    struct Time

A mutable struct representing time. 

# Fields
- `t::Vector{Float32}`: A vector containing the current time.
- `tt::Vector{Int}`: A vector containing the current time step.
- `dt::Float32`: The time step size.

"""
Time
@snn_kw mutable struct Time{VFT = Vector{Float32}, VIT = Vector{Int32}, FT = Float32}
    t::VFT = [0.0f0]
    tt::VIT = Int32[0]
    dt::FT = 0.125f0
end

"""
    get_time(T::Time)

Get the current time.

# Arguments
- `T::Time`: The Time object.

# Returns
- `Float32`: The current time.

"""
get_time(T::Time)::Float32 = T.t[1]

"""
    get_step(T::Time)

Get the current time step.

# Arguments
- `T::Time`: The Time object.

# Returns
- `Float32`: The current time step.

"""
get_step(T::Time)::Float32 = T.tt[1]

"""
    get_dt(T::Time)

Get the time step size.

# Arguments
- `T::Time`: The Time object.

# Returns
- `Float32`: The time step size.

"""
get_dt(T::Time)::Float32 = T.dt

"""
    get_interval(T::Time)

Get the time interval from 0 to the current time.

# Arguments
- `T::Time`: The Time object.

# Returns
- `StepRange{Float32}`: The time interval.

"""
get_interval(T::Time) = Float32(T.dt):Float32(T.dt):get_time(T)

"""
    update_time!(T::Time, dt::Float32)

Update the current time and time step.

# Arguments
- `T::Time`: The Time object.
- `dt::Float32`: The time step size.

"""
function update_time!(T::Time, dt::Float32)
    T.t[1] += dt
    T.tt[1] += 1
end

"""
    record_fire!(obj::PT, T::Time, indices::Dict{Symbol,Vector{Int}}) where {PT <: Union{AbstractPopulation, AbstractStimulus}}

Record the firing activity of the `obj` object into the `obj.records[:fire]` array.

# Arguments
- `obj::PT`: The object to record the firing activity from.
- `T::Time`: The time at which the recording is happening.
- `indices::Dict{Symbol,Vector{Int}}`: A dictionary containing indices for each variable to record.

"""
function record_fire!(fire::Vector{Bool}, record::Dict{Symbol,AbstractVector}, T::Time, indices::Dict{Symbol,Vector{Int}})
    # @unpack fire = obj
    # @unpack records = obj
    sum(fire) == 0 && return
    ind::Vector{Int} = haskey(indices, :fire) ? indices[:fire] : collect(eachindex(fire))
    t::Float32 = get_time(T)
    push!(record[:time], t)
    push!(record[:neurons], findall(fire[ind]))
end

"""
Store values into the dictionary named `records` in the object given 

# Arguments
- `obj`: An object whose values are to be recorded

"""
function record!(obj::P, T::Time) where {P<:Union{AbstractPopulation, AbstractStimulus}}
    @unpack records = obj
    haskey(records, :fire) && record_fire!(obj.fire, obj.records[:fire], T, records[:indices])
    # timestamp::Vector{Float32} = records[:timestamp]
    # if (get_step(T) % round(Int, 1/get_dt(T)) == 0)
    #     push!(timestamp, get_time(T))
    # end
    @inbounds for key::Symbol in keys(records)
        (key == :indices) && (continue)
        (key == :sr) && (continue)
        (key == :timestamp) && (continue)
        (key == :fire) && (continue)
        record_sym!(obj, key, T, records[:indices], records[:sr][key])
    end
end

"""
    record_sym!(obj, key::Symbol, T::Time, indices::Dict{Symbol,Vector{Int}})

Record the variable `key` of the `obj` object into the `obj.records[key]` array.

# Arguments
- `obj`: The object to record the variable from.
- `key::Symbol`: The key of the variable to record.
- `T::Time`: The time at which the recording is happening.
- `indices::Dict{Symbol,Vector{Int}}`: A dictionary containing indices for each variable to record.

"""
function record_sym!(obj, key::Symbol, T::Time, indices::Dict{Symbol,Vector{Int}}, sr::Float32) 
    my_record = getfield(obj, key)
    @unpack records = obj
    !record_step(T,sr) && return
    ind::Vector{Int} = haskey(indices, key) ? indices[key] : axes(my_record,1)
    @inbounds _record_sym(my_record, records[key], ind)
end

@inline function _record_sym(my_record::Vector{T}, records::Vector{Vector{T}}, ind::Vector{Int}) where {T<:Real}
    push!(records, my_record[ind])
end

@inline function _record_sym(my_record::Matrix{T}, records::Vector{Matrix{T}}, ind::Vector{Int}) where {T<:Real}
    push!(records, my_record[ind,:])
end

@inline function record_step(T, sr) 
    (get_step(T) % floor(Int, 1.f0/sr/get_dt(T))) == 0
end



function record!(obj::C, T::Time) where {C<:AbstractConnection}
    @unpack records = obj
    # timestamp::Vector{Float32} = records[:timestamp]
    # if (get_step(T) % round(Int, 1/get_dt(T)) == 0)
    #     push!(timestamp, get_time(T))
    # end
    @inbounds if haskey(records, :plasticity)
        for name_plasticity::Symbol in keys(records[:plasticity])
            for key::Symbol in records[:plasticity][name_plasticity]
                record_plast!(obj, obj.plasticity, key, T, records[:indices], records[:sr][key], name_plasticity)
            end
        end
    end
    for key::Symbol in keys(records)
        (key == :indices) && (continue)
        (key == :sr) && (continue)
        (key == :timestamp) && (continue)
        (key == :fire) && (continue)
        (key == :plasticity) && (continue)
        (haskey(records, :plasticity)) && (key ∈ keys(records[:plasticity])) && (continue)
        record_sym!(obj, key, T, records[:indices], records[:sr][key])
    end
end


"""
    record_plast!(obj::ST, plasticity::PT, key::Symbol, T::Time, indices::Dict{Symbol,Vector{Int}}, name_plasticity::Symbol) where {ST <: AbstractConnection, PT <: PlasticityVariables}

Record the plasticity variable `key` of the `plasticity` object into the `obj.records[name_plasticity][key]` array.

# Arguments
- `obj::ST`: The object to record the plasticity variable into.
- `plasticity::PT`: The plasticity object containing the variable to record.
- `key::Symbol`: The key of the variable to record.
- `T::Time`: The time at which the recording is happening.
- `indices::Dict{Symbol,Vector{Int}}`: A dictionary containing indices for each variable to record.
- `name_plasticity::Symbol`: The name of the plasticity object.

"""
function record_plast!(obj::ST, plasticity::PT, key::Symbol, T::Time, indices::Dict{Symbol,Vector{Int}}, sr::Float32, name_plasticity::Symbol) where {ST <: AbstractConnection, PT <: PlasticityVariables}
    # my_record = getfield(obj, key)
    # @unpack records = obj
    !record_step(T,sr) && return
    ind::Vector{Int} = haskey(indices, key) ? indices[key] : collect(eachindex(getfield(plasticity, key)))
    push!(obj.records[name_plasticity][key], getfield(plasticity, key)[ind])
end


"""
Initialize dictionary records for the given object, by assigning empty vectors to the given keys

# Arguments
- `obj`: An object whose variables will be monitored
- `keys`: The variables to be monitored

"""
function monitor(obj, keys; sr=1000Hz, T::Time=Time())
    if !haskey(obj.records, :indices)
        obj.records[:indices] = Dict{Symbol,Vector{Int}}()
    end
    if !haskey(obj.records, :sr)
        obj.records[:sr] = Dict{Symbol,Float32}()
    end
    if !haskey(obj.records, :timestamp)
        obj.records[:timestamp] = Vector{Float32}()
    end
    for key in keys
        ## If the key is a tuple, then the first element is the symbol and the second element is the list of neurons to record.
        if isa(key, Tuple)
            sym, ind = key
            push!(obj.records[:indices], sym => ind)
        else
            sym = key
        end
        push!(obj.records[:sr], sym => sr)
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
            @warn "Field $sym not found in $(nameof(typeof(obj)))"
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
"""
monitor(objs::Array, keys)

Function called when more than one object is given, which then calls the above monitor function for each object
"""
function monitor(objs::Array, keys; sr=200Hz)
    for obj in objs
        monitor(obj, keys, sr=sr)
    end
end

"""
    interpolated_record(p, sym)

    Returns the recording with interpolated time values and the extrema of the recorded time points.

    N.B. 
    ----
    The element can be accessed at whichever time point by using the index of the array. The time point must be within the range of the recorded time points, in r_v.
"""
function interpolated_record(p, sym)
    sr = p.records[:sr][sym]
    v_dt = SNN.getvariable(p, sym)

    # ! adjust the end time to account for the added first element 
    _end = (size(v_dt,)[end]-1)/sr  
    # ! this is the recorded time (in ms), it assumes all recordings are contained in v_dt
    r_v = 0:1/sr:_end 

    # Set NoInterp in the singleton dimensions:
    interp = get_interpolator(v_dt)
    v = interpolate(v_dt, interp)

    ax = map(1:length(size(v_dt))-1) do i
        axes(v_dt, i)
    end
    y = scale(v, ax..., r_v)
    return y, extrema(r_v)
end

function squeeze(A::AbstractArray)
    singleton_dims = tuple((d for d in 1:ndims(A) if size(A, d) == 1)...)
    return dropdims(A, dims=singleton_dims)
end

function get_interpolator(A::AbstractArray)
    singleton_dims = tuple((d for d in 1:ndims(A) if size(A, d) == 1)...)
    interp = repeat(Vector{Any}([BSpline(Linear())]), ndims(A))
    for d in singleton_dims
        interp[d] = NoInterp()
    end
    return Tuple(interp)
end



"""
getvariable(obj, key, id=nothing)

Returns the recorded values for a given object and key. If an id is provided, returns the recorded values for that specific id.
"""
function getvariable(obj, key, id=nothing)
    rec = getrecord(obj, key)
    if isa(rec[1], Matrix)
        array = zeros(size(rec[1])..., length(rec))
        for i in eachindex(rec)
            array[:,:,i] = rec[i]
        end
        return array
    else
        isnothing(id) && return hcat(rec...)
        return hcat(rec...)[id,:]
    end
end

"""
getrecord(p, sym)

Returns the recorded values for a given object and symbol. If the symbol is not found in the object's records, it checks the records of the object's plasticity and returns the values for the matching symbol.
"""
function getrecord(p, sym)
    key = sym
    if haskey(p.records, key) 
        return p.records[key]
    elseif haskey(p.records, :plasticity)
        values = []
        names = []
        for (name, keys) in p.records[:plasticity]
            if sym in keys
                push!(values, p.records[name][sym])
                push!(names, name)
            end
        end
        if length(values) == 1
            return values[1]
        else
            Dict{Symbol,Vector{Any}}(zip(names, values))
        end
    else
        throw(ArgumentError("The record $sym is not found"))
    end
end

"""
clear_records(obj)

Clears all the records of a given object.
"""
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

"""
clear_records(obj, sym::Symbol)

Clears the records of a given object for a specific symbol.
"""
function clear_records(obj, sym::Symbol)
    for (key, val) in obj.records
        (key == sym) && (empty!(val))
    end
end

"""
clear_records(objs::AbstractArray)

Clears the records of multiple objects.
"""
function clear_records(objs::AbstractArray)
    for obj in objs
        clear_records(obj)
    end
end


"""
clear_monitor(obj)

Clears all the records of a given object.
"""
function clear_monitor(obj)
    for (k, val) in obj.records
        delete!(obj.records, k)
    end
end

function clear_monitor(objs::NamedTuple)
    for obj in values(objs)
        clear_monitor(obj)
    end
end


export Time, get_time, get_step, get_dt, get_interval, update_time!, record_plast!, record_fire!, record_sym!, record!, monitor, monitor_plast, getvariable, getrecord, clear_records, clear_monitor