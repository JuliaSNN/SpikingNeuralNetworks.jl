
macro update(nt, fields_and_value...)
    # Split field accessors and the new value
    fields = fields_and_value[1:end-2]
    last_field = fields_and_value[end-1]
    new_value = fields_and_value[end]

    # Recursive function to rebuild the NamedTuple
    function build_expr(path, value_expr)
        if isempty(path)
            return value_expr
        else
            field = path[1]
            rest = path[2:end]
            return :(merge($nt, (; $field = $(build_expr(:($nt.$field), rest, value_expr)))))
        end
    end

    function build_expr(current_expr, path::Vector{Symbol}, value_expr)
        if length(path) == 1
            # Final field to replace
            field = path[1]
            return :(merge($current_expr, (; $field = $value_expr)))
        else
            # Recursive step: rebuild current NamedTuple with one field replaced
            field = path[1]
            return :(merge($current_expr, (; $field = $(build_expr(:($current_expr.$field), path[2:end], value_expr)))))
        end
    end

    # Construct and return the update expression
    full_path = [fields... , last_field]
    return build_expr(nt, full_path, new_value)
end

macro update_f(nt, block)
    updates = block isa Expr && block.head == :block ? block.args : [block]

    result_expr = esc(nt)  # Start with the original NamedTuple

    for update in updates
        # Collect all assignments per line like: `connection a = 3 b = 4`
        if !(update isa Expr && update.head == :(=))
            error("@update_fields: each line must be like `field1 field2 = value ...`")
        end

        lhs = update.args[1]
        rhs = update.args[2]

        # Get the full path (e.g., `connection a`)
        path = lhs isa Symbol ? [lhs] : lhs.args

        # There might be multiple assignments in one line, like: a = 3 b = 4
        # So we handle that using recursion over `Expr(:=, lhs, rhs)`
        result_expr = build_nested_update(result_expr, path, rhs)
    end

    return result_expr
end

function build_nested_update(base, path::Vector{Any}, value)
    if length(path) == 1
        # Final field to be updated
        field = path[1]
        return :(merge($base, (; $field = $value)))
    else
        # Go deeper
        field = path[1]
        rest = path[2:end]
        sub_expr = build_nested_update(:($base.$field), rest, value)
        return :(merge($base, (; $field = $sub_expr)))
    end
end
# Update examples:
network = @update network connection a =3
network = @update network plasticity d up = Dict{}(:a=>1)

local my_config = (;base_config...,network = (
    base_config.network...,
    connections = (base_config.network.connections...,
            Th_to_E = (p=2, μ=1),
        )
    ),
    plasticity = (base_config.network.plasticity...,
        iSTDP_potential =SNN.iSTDPParameterPotential(η = 0.02, v0 = 3, τy = 100ms, Wmax = 200.0pF, Wmin = 2.78pF)
    ),
    noise =
        (;base_config.network.noise...,
        exc_soma = (param=2.0kHz,  μ=2.0f0,  neurons=:ALL, name="noise_exc_soma"),
        exc_dend = (param=2.0kHz,  μ=2.f0,  neurons=:ALL, name="noise_exc_dend")),
)

config = base_config
config = @update config network connections Th_to_E  (p=1, μ=2)
config = @update config network plasticity iSTDP_potential = SNN.iSTDPParameterPotential(η = 0.02, v0 = 3, τy = 100ms, Wmax = 200.0pF, Wmin = 2.78pF)
config = @update config network noise exc_soma = (param=2.0kHz,  μ=2.0f0,  neurons=:ALL, name="noise_exc_soma")
config = @update config network noise exc_dend = (param=2.0kHz,  μ=2.f0,  neurons=:ALL, name="noise_exc_dend")

my_network.network.connections.Th_to_E
config.network.connections