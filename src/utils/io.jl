import DrWatson: save, load

function SNNfolder(path, name, info)
    return joinpath(path, savename(name, info, connector = "-"))
end

function SNNfile(type, count)
    count_string = count > 0 ? "-$(count)" : ""
    return "$(type)$count_string.jld2"
end

function SNNpath(path, name, info, type, count)
    return joinpath(SNNfolder(path, name, info), SNNfile(type, count))
end

function SNNload(;
    path::String,
    name::String = "",
    info = nothing,
    count::Int = 1,
    type::Symbol = :model,
)
    ## Check if path is a directory
    if isfile(path)
        @info "Loading $(path)"
        return dict2ntuple(DrWatson.load(path))
    else
        if isempty(name) || isnothing(info)
            throw(
                ArgumentError(
                    "If path is not file, `name::String`` and `info::NamedTuple` are required",
                ),
            )
        end
        root = path
    end

    path = SNNpath(root, name, info, type, count)
    if !isfile(path)
        legacy_name = joinpath(root, savename(name, info, "$(type).jld2", connector = "-"))
        if isfile(legacy_name)
            @warn "Loading legacy file $(legacy_name). Please consider using the new file format."
        else
            @warn "$(path) not found"
        end

    end

    tic = time()
    DATA = JLD2.load(path)
    @info "$type $(name)"
    @info "Loading time:  $(time()-tic) seconds"
    return dict2ntuple(DATA)
end

SNNload(path::String, name::String = "", info = nothing, kwargs...) =
    SNNload(; path = path, name = name, info = info, kwargs..., type = :model)
load_model(path::String, name::String, info::NamedTuple; kwargs...) =
    SNNload(; path = path, name = name, info = info, kwargs..., type = :model)
load_data(path::String, name::String, info::NamedTuple; kwargs...) =
    SNNload(; path = path, name = name, info = info, kwargs..., type = :data)
load_data(path, name, info) = SNNload(; path, name, info, type = :data, kwargs...)

function load_or_run(f::Function; path, name, info, exp_config...)
    loaded = load_model(path, name, info)
    if isnothing(loaded)
        name = savename(name, info, connector = "-")
        @info "Running simulation for: $name"
        produced = f(info)
        save_model(model = produced, path = path, name = name, info = info, exp_config...)
        return produced
    end
    return loaded
end




function SNNsave(
    model;
    path,
    name,
    info,
    config = nothing,
    type = :all,
    count = 1,
    kwargs...,
)

    function store_data(filename, data)
        Logging.LogLevel(0) == Logging.Error
        @time DrWatson.save(filename, data)
        Logging.LogLevel(0) == Logging.Info
        @info "$type stored. It occupies $(filesize(filename) |> Base.format_bytes)"
    end

    @info "Storing $(type)-$count of `$(savename(name, info, connector="-"))`
    at $(path) \n"

    ## Create directory if it does not exist
    root = SNNfolder(path, name, info)
    isdir(root) || mkpath(root)

    ## Write config file
    if count < 2
        write_config(joinpath(root, "config.jl"), info; config, kwargs...)
    end

    if type == :all
        type = :data
        filename = joinpath(root, SNNfile(type, count))
        data = merge((@strdict model = model config = config), kwargs)
        store_data(filename, data)

        type = :model
        _model = deepcopy(model)
        clear_records!(_model)
        filename = joinpath(root, SNNfile(type, count))
        data = merge((@strdict model = _model config = config), kwargs)
        store_data(filename, data)
        return filename
    elseif type == :model
        _model = deepcopy(model)
        clear_records!(_model)
        filename = joinpath(root, SNNfile(type, count))
        data = merge((@strdict model = _model config = config), kwargs)
        store_data(filename, data)
        return filename
    else
        filename = joinpath(root, SNNfile(type, count))
        data = merge((@strdict model = model config = config), kwargs)
        store_data(filename, data)
        return filename
    end

end

export load, save, load_model, load_data, SNNload, SNNsave, SNNpath, SNNfolder, savename

save_model(; model, path, name, info, config = nothing, kwargs...) = SNNsave(
    model;
    path = path,
    name = name,
    info = info,
    config = config,
    type = :all,
    kwargs...,
)
save_model

function data2model(; path, name = randstring(10), info = nothing, kwargs...)
    # Does data file exist? If no return false
    data_path = joinpath(path, savename(name, info, "data.jld2", connector = "-"))
    !isfile(data_path) && return false
    # Does model file exist? If yes return true
    data = load_data(path, name, info)
    clear_records!(data.model)

    model_path = joinpath(path, savename(name, info, "model.jld2", connector = "-"))
    isfile(model_path) && return true
    # If model file does not exist, save model file
    # Logging.LogLevel(0) == Logging.Error
    @time DrWatson.save(model_path, ntuple2dict(data))

    isfile(model_path) && return true
    @error "Model file not saved"
end

function model_path_name(; path, name = randstring(10), info = nothing, kwargs...)
    @warn " `model_path_name` is deprecated, use `SNNpath` instead"
    return SNNpath(path, name, info, :model, 0)
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

function write_value(file, key, value, indent = "", equal_sign = "=")
    if isa(value, Number)
        println(file, "$indent$key $(equal_sign) $value,")
    elseif isa(value, String)
        println(file, "$indent$key $(equal_sign) \"$value\",")
    elseif isa(value, Symbol)
        println(file, "$indent$key $(equal_sign) :$value,")
    elseif isa(value, Tuple)
        println(file, "$indent$key $(equal_sign) (")
        for v in value
            write_value(file, "", v, indent * "    ", "")
        end
        println(file, "$indent),")
    elseif typeof(value) <: AbstractRange || isa(value, StepRange{Int64,Int64})
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
                println(file, "$indent    :$k => $v")#$(write_value(file,"",v,"", ""))")
            else
                isa(v, String)
                println(file, "$indent    :$k => \"$v\",")
            end
            # else
            #     # println(file, "$indent    $k => $v,")
            #     write_value(file, k, v, indent * "    ")
            # end
        end
        println(file, "$indent),")
    else
        isa(value, NamedTuple)
        name = isa(value, NamedTuple) ? "" : nameof(typeof(value))
        println(file, "$indent$key $equal_sign $(name)(")
        for field in fieldnames(typeof(value))
            field_value = getfield(value, field)
            write_value(file, field, field_value, indent * "    ")
        end
        println(file, "$indent),")
    end
end

function write_config(path::String, info; config, name = "", kwargs...)
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
    for (key, value) in pairs(info)
        String(key) == "study" || String(key)=="models" && continue
        write_value(file, key, value, "    ")
    end
    println(file, ")")
    println(file, "config = (")
    for (key, value) in pairs(config)
        String(key) == "study" || String(key)=="models" && continue
        write_value(file, key, value, "    ")
    end
    println(file, ")")
    # for (info_name, info_value) in pairs(kwargs)
    #     String(info_name) == "sequence" && continue
    #     if isa(info_value, NamedTuple)
    #         println(file, "$(info_name) = (")
    #         for (key, value) in pairs(info_value)
    #             write_value(file, key, value, "        ")
    #         end
    #         println(file, "    )")
    #     end
    # end
    close(file)
    @info "Config file saved"
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


## import all models/data from folder
function read_folder(
    path,
    files = nothing;
    my_filter = (file, type)->endswith(file, "$(type).jld2"),
    type = :model,
    name = nothing,
)
    if isnothing(files)
        files = []
    end
    n = 0
    for file in readdir(path)
        if my_filter(file, type)
            n+=1
            @info n, file
            push!(files, joinpath(path, file))
        end
    end
    return files
end

function read_folder!(df, path; type = :model, name = nothing)
    read_folder(path, df; type = type, name = name)
end




export save_model,
    load_model,
    load_data,
    save_parameters,
    get_path,
    data2model,
    write_config,
    print_summary,
    load_or_run,
    read_folder,
    read_folder!
