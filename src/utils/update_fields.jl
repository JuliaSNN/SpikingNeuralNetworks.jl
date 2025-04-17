
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
            @assert lhs.head == Symbol(".")
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

# Example base configuration
base_config = (
    network = (
        connections = (Th_to_E = (p = 1, μ = 0.5),),
        plasticity = (
            iSTDP_potential = (η = 0.01, v0 = 2, τy = 50, Wmax = 100.0, Wmin = 1.0),
        ),
    ),
    noise = (
        exc_soma = (param = 1.0, μ = 1.0, neurons = "ALL", name = "noise_exc_soma"),
        exc_dend = (param = 1.0, μ = 1.0, neurons = "ALL", name = "noise_exc_dend"),
    ),
)


# Use the macro to update specific values
function update_func(new_value, config)
    return @update config begin
        network.connections.Th_to_E.p = new_value
        noise.exc_soma.param = 3.0
        network.plasticity.iSTDP_potential.η = 1000
    end
    return new_config
end

new_config = update_func(10, base_config)
@show new_config.network.connections.Th_to_E.p  ## should be 10
