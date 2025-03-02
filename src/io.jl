
function load_data(path, name = nothing, info = nothing)
    isfile(path) && (return dict2ntuple(DrWatson.load(path)))
    if isnothing(name)
        throw(ArgumentError("$path is not file, config is required"))
    end
    name = savename(name, info, "data.jld2", connector = "-")
    path = joinpath(path, name)
    if !isfile(path) 
        @warn  "Model $(path) not found"
                return nothing
    end
    tic = time()
    # DATA = DrWatson.load(path)
    DATA = JLD2.load(path)
    @info "Data $(name)"
    @info "Loading time:  $(time()-tic) seconds"
    return dict2ntuple(DATA)
end
load_data(;path, name, info) = load_data(path, name, info)


function load_model(path, name = nothing, info = nothing)
    isfile(path) && (return dict2ntuple(DrWatson.load(path)))
    if isnothing(name)
        throw(ArgumentError("If path is not file, config is required"))
    end
    name = savename(name, info, "model.jld2", connector = "-")
    path = joinpath(path, name)
    tic = time()
    if !isfile(path) 
        @warn  "Model $(path) not found"
                return nothing
    end
    DATA = JLD2.load(path)
    @info "Model $(name)"
    @info "Loading time:  $(time()-tic) seconds"
    return dict2ntuple(DATA)
end
load_model(;path, name, info) = load_model(path, name, info,)

function load_or_run(f::Function, path, name, info; exp_config...) 
    loaded = load_model(path, name, info) 
    if isnothing(loaded)
        name = savename(name, info, connector = "-")
        @info "Running simulation for: $name"
        produced = f(info)
        save_model(path=path, model=produced, name=name, info=info, exp_config...)
        return produced
    end
    return loaded
end


export load_data, load_model, save_model, savemodel

function save_model(; path, model, name, info, config, kwargs...)
    @info "Model: `$(savename(name, info, connector="-"))` \nsaved at $(path)"
    isdir(path) || mkpath(path)

    config_path = joinpath(path, savename(name, info, "jl.config", connector = "-"))
    write_config(config_path, info; config, kwargs...)

    data_path = joinpath(path, savename(name, info, "data.jld2", connector = "-"))
    Logging.LogLevel(0) == Logging.Error
    @time DrWatson.save(data_path, merge((@strdict model = model config=config), kwargs))
    Logging.LogLevel(0) == Logging.Info
    @info "-> Data ($(filesize(data_path) |> Base.format_bytes))"

    _model = deepcopy(model)
    clear_records(_model)

    model_path = joinpath(path, savename(name, info, "model.jld2", connector = "-"))
    Logging.LogLevel(0) == Logging.Error
    @time DrWatson.save(model_path, merge((@strdict model = _model config=config), kwargs))
    Logging.LogLevel(0) == Logging.Info
    @info "-> Model ($(filesize(model_path) |> Base.format_bytes))"
    return data_path
end

function data2model(; path, name = randstring(10), info = nothing, kwargs...)
    # Does data file exist? If no return false
    data_path = joinpath(path, savename(name, info, "data.jld2", connector = "-")) 
    !isfile(data_path) && return false
    # Does model file exist? If yes return true
    data = load_data(path, name, info)
    clear_records(data.model)

    model_path = joinpath(path, savename(name, info, "model.jld2", connector = "-"))
    isfile(model_path) && return true
    # If model file does not exist, save model file
    # Logging.LogLevel(0) == Logging.Error
    @time DrWatson.save(model_path, ntuple2dict(data))

    isfile(model_path) && return true
    @error "Model file not saved"
end

function get_path(; path, name = randstring(10), info = nothing, kwargs...)
    model_path = joinpath(path, savename(name, info, "model.jld2", connector = "-"))
    return model_path
end

function save_parameters(;
    path,
    parameters,
    name = randstring(10),
    info = nothing,
    file_path,
    force = false,
)
    @info "Parameters: `$(savename(name, info, connector="-"))` \nsaved at $(path)"

    isdir(path) || mkpath(path)

    params_path = joinpath(path, savename(name, info, "params.jld2", connector = "-"))
    DrWatson.save(params_path, @strdict parameters)  # Here you are saving a Julia object to a file

    params_path = joinpath(path, savename(name, info, "params.jl.script", connector = "-"))
    isfile(params_path) &&
        !force &&
        throw("File already exists, use force=true to overwrite")
    !isfile(params_path) && cp(file_path, params_path)
    return
end

# Helper function to get the current timestamp
function get_timestamp()
    return now()
end

# Helper function to get the current Git commit hash
function get_git_commit_hash()
    return readchomp(`git rev-parse HEAD`)
end

function write_value(file, key, value, indent="", equal_sign="=")
    if isa(value, Number)
        println(file, "$indent$key $(equal_sign) $value,")
    elseif isa(value, String)
        println(file, "$indent$key $(equal_sign) \"$value\",")
    elseif isa(value, Symbol)
        println(file, "$indent$key $(equal_sign) :$value,")
    elseif typeof(value) <: AbstractRange || isa(value,StepRange{Int64, Int64})
        _s = step(value)
        _end = last(value)
        _start = first(value)
        println(file, "$indent$key $(equal_sign) $(_start):$(_s):$(_end),")
    elseif isa(value, Bool)
        println(file, "$indent$key $(equal_sign) $value,")
    elseif isa(value, Array)
        println(file, "$indent$key $(equal_sign) [")
        for v in value
            write_value(file, "", v, indent * "    ", "")
        end
        println(file, "$indent],")
    elseif isa(value, Dict)
        println(file, "$indent$key = Dict(")
        for (k, v) in value
            if isa(v, Number)
                println(file, "$indent    :$k => $v,")
            else
                # println(file, "$indent    $k => $v,")
                write_value(file, k, v, indent * "    ")
            end
        end
        println(file, "$indent),")
    else
            name = isa(value,NamedTuple) ? "" : nameof(typeof(value)) 
        println(file, "$indent$key = $(name)(")
        for field in fieldnames(typeof(value))
            field_value = getfield(value, field)
            write_value(file, field, field_value, indent * "    ")
        end
        println(file, "$indent),")
    end
end

function write_config(path::String, config; name="", kwargs...)
    timestamp = get_timestamp()
    commit_hash = get_git_commit_hash()

    if name !== ""
        config_path = joinpath(path, savename(name, info, "config", connector = "-"))
    else    
        config_path = path
    end
    
    file = open(config_path, "w")

    println(file, "# Configuration file generated on: $timestamp")
    println(file, "# Corresponding Git commit hash: $commit_hash")
    println(file, "")
    println(file, "info = (")
    for (key, value) in pairs(config)
        write_value(file, key, value, "    ")
    end
    println(file, ")")
    for (info_name, info_value ) in pairs(kwargs)
        if isa(info_value, NamedTuple)
            println(file, "$(info_name) = (")
            for (key, value) in pairs(info_value)
                write_value(file, key, value, "        ")
            end
            println(file, "    )")
        end
    end
    close(file)
    @info "Config file saved at $(config_path)"
end

"""
    print_summary(p)

    Prints a summary of the given element.
"""
function print_summary(p)
    println("Type: $(nameof(typeof(p))) $(nameof(typeof(p.param)))")
    println("  Name: ", p.name)
    println("  Number of Neurons: ", p.N)
    for k in fieldnames(typeof(p.param))
        println("   $k: $(getfield(p.param,k))")
    end
end




export save_model, load_model, load_data, save_parameters, get_path, data2model, write_config, print_summary, load_or_run
