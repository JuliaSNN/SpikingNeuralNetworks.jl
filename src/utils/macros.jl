
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
snn_kw_str_sentinel_check(x) = :(
    if $(x[1]) isa KwStrSentinel
        $(x[1]) = $(length(x) > 1 ? x[2] : Any)
    end
)

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
    str_fields =
        map(snn_kw_str_field, filter(x -> !(x isa LineNumberNode), str.args[3].args))

    # Remove default type params
    if length(str_params) > 0
        idx = 1
        for idx = 2:length(str.args[2].args)
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
    for idx = 1:length(str.args[3].args)
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
    ctor_kws = Expr(
        :parameters,
        map(snn_kw_str_kws, str_fields)...,
        map(snn_kw_str_kws_types, ctor_params)...,
    )
    ctor_sig = Expr(:call, str_name, ctor_kws)
    ctor_call = if length(str_params) > 0
        Expr(:curly, str_name, first.(str_params)...)
    else
        str_name
    end
    ctor_body =
        Expr(:block, ctor_params_bodies..., Expr(:call, ctor_call, first.(str_fields)...))
    ctor = Expr(:function, ctor_sig, ctor_body)

    return quote
        $(esc(str))
        $(esc(ctor))
    end
end

""" Macro to update fields in a named tuple configuration.
    Usage:
        @update base_config field1.field2 = value
        @update base_config begin
            field1.field2 = value1
            field1.field3.field4 = value2
        end
"""

# Simple update macro for handling multiple updates in a block
macro update(base, update_expr)
    # Verify if the expr is a block or a line
    if update_expr.head == :block
        # if a block, extract the expressions
        updates = update_expr.args

        # Start with the base configuration
        # The :($(esc(base))) is used to ensure the base is evaluated in the correct context (the macro's context)
        current_config = :($(esc(base)))

        # Process each update expression in the block
        for update_expr in updates
            isa(update_expr, LineNumberNode) && continue  # Ensure it's an expression

            # Extract the left-hand side and right-hand side, the left-hand side is the field to update, the right hand side is the new value
            lhs, rhs = update_expr.args

            # Escape the value to ensure it's evaluated in the correct context
            value = :($(esc(rhs)))

            # Assert the left-hand side has the correct structure
            # if isa(lhs, Symbol)
            #     pushfirst!(fields, lhs)  # Add the first part
            # else
                # @assert lhs.head == Symbol(".")
            fields = []
            while !isa(lhs, Symbol)
                pushfirst!(fields, lhs.args[2].value)  # Collect the field names
                lhs = lhs.args[1]  # Move to the next part of the path
            end
            pushfirst!(fields, lhs)  # Add the first part

            # Convert the field names into symbols
            field_syms = [Symbol(f) for f in fields]

            # Apply the update to the current config using the helper function
            current_config = :(update_with_merge($current_config, $field_syms, $value))
        end
        return current_config
    else
        lhs, rhs = update_expr.args  # Extract the left-hand side and right-hand side


        @assert lhs.head == Symbol(".")
        fields = []
        while !isa(lhs, Symbol)
            pushfirst!(fields, lhs.args[2].value)  # Collect the field names
            lhs = lhs.args[1]  # Move to the next part of the path
        end
        pushfirst!(fields, lhs)  # Add the first part

        # Convert the field names into symbols
        field_syms = [Symbol(f) for f in fields]

        # Return the updated expression with deep merge
        return :(update_with_merge($base, $field_syms, $rhs))
    end
    # end
end

# Deep merge function for named tuples
function update_with_merge(base_config::NamedTuple, path::Vector{Symbol}, value)

    if length(path) == 1
        # If it's the final field, update the value
        return merge(base_config, (path[1] => value,))
    else
        key = path[1]
        sub = getfield(base_config, key)
        # Recursively update the nested subfield
        updated_sub = update_with_merge(sub, path[2:end], value)

        # Merge the updated subfield back into the base
        return merge(base_config, (key => updated_sub,))
    end
end


export @symdict, @snn_kw, @update#, update_with_merge
