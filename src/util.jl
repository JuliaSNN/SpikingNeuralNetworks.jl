function connect!(c, j, i, σ = 1e-6)
    W = sparse(c.I, c.J, c.W, length(c.rowptr) - 1, length(c.colptr) - 1)
    W[i, j] = σ * randn(Float32)
    c.rowptr, c.colptr, c.I, c.J, c.index, c.W = dsparse(W)
    c.tpre, c.tpost, c.Apre, c.Apost = zero(c.W), zero(c.W), zero(c.W), zero(c.W)
    return nothing
end

function dsparse(A)
    At = sparse(A')
    colptr = A.colptr
    rowptr = At.colptr
    I = rowvals(A)
    V = nonzeros(A)
    J = zero(I)
    # FIXME: Breaks when A is empty
    for j in 1:(length(colptr) - 1)
        J[colptr[j]:(colptr[j+1] - 1)] .= j
    end
    index = zeros(size(I))
    coldown = zeros(eltype(index), length(colptr) - 1)
    for i in 1:(length(rowptr) - 1)
        for st in rowptr[i]:(rowptr[i+1] - 1)
            j = At.rowval[st]
            index[st] = colptr[j] + coldown[j]
            coldown[j] += 1
        end
    end
    # Test.@test At.nzval == A.nzval[index]
    rowptr, colptr, I, J, index, V
end

function record!(obj)
    for (key, val) in obj.records
        if isa(key, Tuple)
            sym, ind = key
            push!(val, getindex(getfield(obj, sym),ind))
        else
            push!(val, copy(getfield(obj, key)))
        end
    end
end

function monitor(obj, keys)
    for key in keys
        if isa(key, Tuple)
            sym, ind = key
        else
            sym = key
        end
        typ = typeof(getfield(obj, sym))
        obj.records[key] = Vector{typ}()
    end
end

function monitor(objs::Array, keys)
    for obj in objs
        monitor(obj, keys)
    end
end

function getrecord(p, sym)
    key = sym
    for (k,val) in p.records
        isa(k, Tuple) && k[1] == sym && (key = k)
    end
    p.records[key]
end

function clear_records(obj)
    for (key, val) in obj.records
        empty!(val)
    end
end

@inline function exp32(x::Float32)
    x = ifelse(x < -10f0, -32f0, x)
    x = 1f0 + x / 32f0
    x *= x; x *= x; x *= x; x *= x; x *= x
    return x
end

@inline function exp256(x::Float32)
    x = ifelse(x < -10f0, -256f0, x)
    x = 1.0f0 + x / 256.0f0
    x *= x; x *= x; x *= x; x *= x
    x *= x; x *= x; x *= x; x *= x
    return x
end

macro symdict(x...)
    ex = Expr(:block)
    push!(ex.args, :(d = Dict{Symbol,Any}()))
    for p in x
        push!(ex.args, :(d[$(QuoteNode(p))] = $(esc(p))))
    end
    push!(ex.args, :(d))
    return ex
end

snn_kw_str_param(x::Symbol) = (x,)
function snn_kw_str_param(x::Expr)
    if x.head == :(<:)
        return (x.args...,)
    elseif x.head == :(=)
        if x.args[1] isa Expr && x.args[1].head == :(<:)
            return (x.args[1].args..., x.args[2])
        elseif x.args[1] isa Symbol
            return (x.args[1], Any, x.args[2])
        end
    end
    error("Can't handle param Expr: $x")
end
snn_kw_str_field(x::Symbol) = (x,)
function snn_kw_str_field(x::Expr)
    if x.head == :(::)
        return (x.args...,)
    elseif x.head == :(=)
        return (x.args[1].args[1:2]..., x.args[2])
    end
    error("Can't handle field Expr: $x")
end
function snn_kw_str_kws(x::Tuple)
    if 1 <= length(x) <= 2
        return x[1]
    elseif length(x) == 3
        return Expr(:kw, x[1], x[3])
    end
end
function snn_kw_str_kws_types(x::Tuple)
    if 1 <= length(x) <= 2
        return Expr(:(::), x[1], x[2])
    elseif length(x) == 3
        return Expr(:kw, x[1], x[3])
    end
end
struct KwStrSentinel end
function snn_kw_str_sentinels(x)
    if length(x) == 1
        return (x[1], Any, :(KwStrSentinel()))
    elseif length(x) == 2
        return (x[1], Any, :(KwStrSentinel()))
    else
        return x
    end
end
snn_kw_str_sentinel_check(x) =
    :(if $(x[1]) isa KwStrSentinel
        $(x[1]) = $(length(x) > 1 ? x[2] : Any)
    end)
"A minimal implementation of `Base.@kwdef` with default type parameter support"
macro snn_kw(str)
    str_abs = nothing
    if str.args[2] isa Expr && str.args[2].head == :(<:)
        # Lower abstract type
        str_abs = str.args[2].args[2]
        str.args[2] = str.args[2].args[1]
    end
    if str.args[2] isa Symbol
        # No type params
        str_name = str.args[2]
        str_params = []
    else
        # Has type params
        str_name = str.args[2].args[1]
        str_params = map(snn_kw_str_param, str.args[2].args[2:end])
    end
    @assert str_name isa Symbol
    @assert str_abs isa Union{Symbol,Nothing}
    str_fields = map(snn_kw_str_field, filter(x->!(x isa LineNumberNode),
                                          str.args[3].args))

    # Remove default type params
    if length(str_params) > 0
        idx = 1
        for idx in 2:length(str.args[2].args)
            param = str_params[idx-1]
            if length(param) == 1
                str.args[2].args[idx] = param[1]
            else
                str.args[2].args[idx] = Expr(:(<:), param[1:2]...)
            end
        end
    end

    # Remove default field values
    idx = 1
    subidx = 1
    for idx in 1:length(str.args[3].args)
        if !(str.args[3].args[idx] isa LineNumberNode)
            field = str_fields[subidx]
            if length(field) == 1
                str.args[3].args[idx] = field[1]
            else
                str.args[3].args[idx] = Expr(:(::), field[1:2]...)
            end
            subidx += 1
        end
    end

    # Replace abstract type
    if str_abs !== nothing
        str.args[2] = Expr(:(<:), str.args[2], str_abs)
    end

    # Use sentinels to track if type param kwargs are assigned
    ctor_params = snn_kw_str_sentinels.(str_params)
    ctor_params_bodies = snn_kw_str_sentinel_check.(str_params)

    # Constructor accepts field values and type params as kwargs
    ctor_kws = Expr(:parameters,
                    map(snn_kw_str_kws, str_fields)...,
                    map(snn_kw_str_kws_types, ctor_params)...)
    ctor_sig = Expr(:call, str_name, ctor_kws)
    ctor_call = if length(str_params) > 0
        Expr(:curly, str_name, first.(str_params)...)
    else
        str_name
    end
    ctor_body = Expr(:block,
                     ctor_params_bodies...,
                     Expr(:call, ctor_call, first.(str_fields)...))
    ctor = Expr(:function, ctor_sig, ctor_body)

    return quote
        $(esc(str))
        $(esc(ctor))
    end
end
