import Interpolations: scale, interpolate, BSpline, Linear, NoInterp
"""
    get_time(T::Time)

Get the current time.

# Arguments
- `T::Time`: The Time object.

# Returns
- `Float32`: The current time.

"""
get_time(T::Time)::Float32 = T.t[1]

get_time(model::NamedTuple)::Float32 = model.time.t[1]

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

function update_time!(T::Time, myT::Time)
    T.t[1] = myT.t[1]
    T.tt[1] = myT.tt[1]
end

function reset_time!(T::Time)
    T.t[1] = 0.0f0
    T.tt[1] = 0
end

function reset_time!(model::NamedTuple)
    model.time.t[1] = 0.0f0
    model.time.tt[1] = 0
end

"""
    record_fire!(obj::PT, T::Time, indices::Dict{Symbol,Vector{Int}}) where {PT <: Union{AbstractPopulation, AbstractStimulus}}

Record the firing activity of the `obj` object into the `obj.records[:fire]` array.

# Arguments
- `obj::PT`: The object to record the firing activity from.
- `T::Time`: The time at which the recording is happening.
- `indices::Dict{Symbol,Vector{Int}}`: A dictionary containing indices for each variable to record.

"""
function record_fire!(
    fire::Vector{Bool},
    record::Dict{Symbol,AbstractVector},
    T::Time,
    indices::Dict{Symbol,Vector{Int}},
)
    # @unpack fire = obj
    # @unpack records = obj
    sum(fire) == 0 && return
    ind::Vector{Int} = haskey(indices, :fire) ? indices[:fire] : collect(eachindex(fire))
    t::Float32 = get_time(T)
    push!(record[:time], t)
    push!(record[:neurons], findall(fire[ind]))
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
function record_sym!(
    my_record,
    obj,
    key::Symbol,
    T::Time,
    indices::Dict{Symbol,Vector{Int}},
    sr::Float32,
)
    !record_step(T, sr) && return
    ind::Vector{Int} = haskey(indices, key) ? indices[key] : axes(my_record, 1)
    @inbounds _record_sym(my_record, obj.records[key], ind)
end

@inline function _record_sym(
    my_record::Vector{T},
    records::Vector{Vector{T}},
    ind::Vector{Int},
) where {T<:Real}
    push!(records, my_record[ind])
end

@inline function _record_sym(
    my_record::T,
    records::Vector{T},
    ind::Vector{Int},
) where {T<:Real}
    push!(records, my_record)
end

@inline function _record_sym(
    my_record::Array{T,3},
    records::Vector{Array{T,3}},
    ind::Vector{Int},
) where {T<:Real}
    push!(records, my_record[ind, :, :])
end

@inline function _record_sym(
    my_record::Vector{Vector{T}},
    records::Vector{Vector{Vector{T}}},
    ind::Vector{Int},
) where {T<:Real}
    push!(records, deepcopy(my_record[ind]))
end

@inline function _record_sym(
    my_record::Matrix{T},
    records::Vector{Matrix{T}},
    ind::Vector{Int},
) where {T<:Real}
    push!(records, my_record[ind, :])
end

@inline function record_step(T, sr)
    (get_step(T) % floor(Int, 1.0f0 / sr / get_dt(T))) == 0
end


function record!(obj, T::Time) 
    @unpack records = obj
    for key::Symbol in keys(records)
        if key == :fire
            record_fire!(obj.fire, obj.records[:fire], T, records[:indices])
            continue
        end
        for v in records[:variables]
            if startswith(string(key), string(v))
                sym = string(key)[length(string(v))+2:end] |> Symbol
                record_sym!(getfield(getfield(obj,v), sym), obj, key, T, records[:indices], records[:sr][key])
            end
        end
        hasfield(typeof(obj), key) && record_sym!(getfield(obj, key), obj, key, T, records[:indices], records[:sr][key])
    end
end


"""
Initialize dictionary records for the given object, by assigning empty vectors to the given keys

# Arguments
- `obj`: An object whose variables will be monitored
- `keys`: The variables to be monitored

"""
function monitor!(
    obj::Item,
    keys;
    sr = 1000Hz,
    T::Time = Time(),
    variables::Symbol = :none,
) where {Item<:Union{AbstractPopulation,AbstractStimulus,AbstractConnection}}
    if !haskey(obj.records, :indices)
        obj.records[:indices] = Dict{Symbol,Vector{Int}}()
    end
    if !haskey(obj.records, :sr)
        obj.records[:sr] = Dict{Symbol,Float32}()
    end
    if !haskey(obj.records, :variables)
        obj.records[:variables] = Vector{Symbol}()
    end
    if !haskey(obj.records, :timestamp)
        obj.records[:timestamp] = Vector{Float32}()
    end
    ## If the key is a tuple, then the first element is the symbol and the second element is the list of neurons to record.
    for key in keys
        sym, ind = isa(key, Tuple) ? key : (key, [])
        if sym == :fire
            ## If the then assign a Spiketimes object to the dictionary `records[:fire]`, add as many empty vectors as the number of neurons in the object as in [:indices][:fire]
            obj.records[:fire] = Dict{Symbol,AbstractVector}(
                :time => Vector{Float32}(),
                :neurons => Vector{Vector{Int}}(),
            )
            continue
        end
        if variables == :none
            if hasfield(typeof(obj), sym)
                typ = typeof(getfield(obj, sym))
                key = sym
                !isempty(ind) && (obj.records[:indices][key] = ind)
                obj.records[:sr][key] = sr
                obj.records[key] = Vector{typ}()
            else
                @warn "Field $sym not found in $(nameof(typeof(obj)))"
                continue
            end
        else
            if hasfield(typeof(obj), variables) && hasfield(typeof(getfield(obj,variables)), sym)
                typ = typeof(getfield(getfield(obj,variables), sym))
                key = Symbol(variables,"_", sym)
                variables ∈ obj.records[:variables] && continue
                push!(obj.records[:variables], variables)
            else
                @warn "Field $variables not found in $(nameof(typeof(obj)))"
            end
        end
        !isempty(ind) && (obj.records[:indices][key] = ind)
        obj.records[:sr][key] = sr
        obj.records[key] = Vector{typ}()
    end
end

"""
monitor!(objs::Array, keys)

Function called when more than one object is given, which then calls the above monitor function for each object
"""
function monitor!(objs::Array, keys; sr = 200Hz, kwargs...)
    for obj in objs
        monitor!(obj, keys, sr = sr; kwargs...)
    end
end

function monitor!(objs::NamedTuple, keys; sr = 200Hz, kwargs...)
    for obj in values(objs)
        monitor!(obj, keys, sr = sr; kwargs...)
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
    if sym == :fire
        return firing_rate(p, τ = 20ms)
    end
    sr = p.records[:sr][sym]
    v_dt = SNN.getvariable(p, sym)

    # ! adjust the end time to account for the added first element 
    _end = (size(v_dt)[end] - 1) / sr
    # ! this is the recorded time (in ms), it assumes all recordings are contained in v_dt
    r_v = 0:(1/sr):_end

    # Set NoInterp in the singleton dimensions:
    interp = get_interpolator(v_dt)
    v = interpolate(v_dt, interp)

    ax = map(1:(length(size(v_dt))-1)) do i
        axes(v_dt, i)
    end
    y = scale(v, ax..., r_v)
    return y, r_v
end

function squeeze(A::AbstractArray)
    singleton_dims = tuple((d for d = 1:ndims(A) if size(A, d) == 1)...)
    return dropdims(A, dims = singleton_dims)
end

function get_interpolator(A::AbstractArray)
    singleton_dims = tuple((d for d = 1:ndims(A) if size(A, d) == 1)...)
    interp = repeat(Vector{Any}([BSpline(Linear())]), ndims(A))
    for d in singleton_dims
        interp[d] = NoInterp()
    end
    return Tuple(interp)
end

function _record(p, sym; interpolate = true, kwargs...)
    if interpolate
        return interpolated_record(p, sym)
    else
        return getvariable(p, sym), []
    end
end

function record(p, sym::Symbol; range=false, interval = nothing, kwargs...)
    if sym == :fire
        @assert !isnothing(interval) "Range must be provided for firing rate recording"
        v, r = firing_rate(p, interval; kwargs...)
        if range
            return v, r
        else
            return v
        end#
    else
        v, r = _record(p, sym; kwargs...)
        if range
            return v, r
        else
            return v
        end
    end
end


function record(p, sym::Symbol, interval::R; kwargs...) where {R<:AbstractRange}
    if sym == :fire
        fr, r = firing_rate(p, interval; kwargs...)
        return fr[:, r]
    else
        v, r = interpolated_record(p, sym)
        return v[:, interval]
    end

end



"""
getvariable(obj, key, id=nothing)

Returns the recorded values for a given object and key. If an id is provided, returns the recorded values for that specific id.
"""
function getvariable(obj, key, id = nothing)
    rec = getrecord(obj, key)
    if isa(rec[1], Matrix)
        @debug "Matrix recording"
        array = zeros(size(rec[1])..., length(rec))
        for i in eachindex(rec)
            array[:, :, i] = rec[i]
        end
        return array
    elseif typeof(rec[1]) <: Vector{Vector{typeof(rec[1][1][1])}} # it is a multipod
        @debug "Multipod recording"
        i = length(rec)
        n = length(rec[1])
        d = length(rec[1][1])
        array = zeros(d, n, i)
        for i in eachindex(rec)
            for n in eachindex(rec[i])
                array[:, n, i] = rec[i][n]
            end
        end
        return array
    else
        @debug "Vector recording"
        isnothing(id) && return hcat(rec...)
        return hcat(rec...)[id, :]
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
clear_records!(obj)

Clears all the records of a given object.
"""
function clear_records!(obj)
    if obj isa AbstractPopulation || obj isa AbstractStimulus || obj isa AbstractConnection
        _clean(obj.records)
    else
        for v in obj
            if v isa AbstractPopulation ||
               v isa AbstractStimulus ||
               v isa AbstractConnection
                @debug "Removing records from $(v.name)"
                _clean(v.records)
            elseif v isa String
                continue
            elseif v isa Time
                continue
            else
                clear_records!(v)
            end
        end
    end

end

function _clean(z)
    for (key, val) in z
        (key == :indices) && (continue)
        (key == :sr) && (continue)
        (key == :timestamp) && (continue)
        (key == :plasticity) && (continue)
        if isa(val, Dict)
            _clean(val)
        else
            empty!(val)
        end
    end
end

"""
clear_records!(obj, sym::Symbol)

Clears the records of a given object for a specific symbol.
"""
function clear_records!(obj, sym::Symbol)
    for (key, val) in obj.records
        (key == sym) && (empty!(val))
    end
end

"""
clear_records!(objs::AbstractArray)

Clears the records of multiple objects.
"""
function clear_records!(objs::AbstractArray)
    for obj in objs
        clear_records!(obj)
    end
end


"""
clear_monitor!(obj)

Clears all the records of a given object.
"""
function clear_monitor!(obj)
    for (k, val) in obj.records
        delete!(obj.records, k)
    end
end

function clear_monitor!(objs::NamedTuple)
    for obj in values(objs)
        try
            clear_monitor!(obj)
        catch
            @warn "Could not clear monitor for $obj"
        end
    end
end


export Time,
    get_time,
    get_step,
    get_dt,
    get_interval,
    update_time!,
    record_plast!,
    record_fire!,
    record_sym!,
    record!,
    monitor!,
    getvariable,
    getrecord,
    clear_records!,
    clear_monitor!,
    record,
    reset_time!
