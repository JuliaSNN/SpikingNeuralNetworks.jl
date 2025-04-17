using .Plots, Statistics

## Raster plot


function raster(spiketimes::Spiketimes, t = nothing, kwargs...)
    t = isnothing(t) ? [0, maximum(vcat(spiketimes...))] : t
    X, Y = _raster(spiketimes, t)
    X, Y = _resample_spikes(X, Y)
    plt = scatter(
        X,
        Y,
        m = (1, :black),
        leg = :none,
        xaxis = ("Time (ms)", (0, Inf)),
        yaxis = ("Neuron",),
        label = "",
    )
    !isnothing(t) && plot!(xlims = t)
    plot!(plt; kwargs...)
    return plt
end

function raster(P, t = nothing; kwargs...)
    raster!(plot(), P, t; kwargs...)
end

function raster!(
    p,
    P,
    t = nothing;
    dt = 0.125ms,
    populations = nothing,
    names = nothing,
    every = 1,
    kwargs...,
)
    t = t[[1, end]]
    if isnothing(populations)
        y0 = Int32[0]
        X = Float32[]
        Y = Float32[]
        names = Vector{String}()
        P = typeof(P) <: AbstractPopulation ? [P] : [getfield(P, k) for k in keys(P)]
        for p in P
            x, y, _y0 = _raster(p, t)
            push!(names, p.name)
            append!(X, x)
            append!(Y, y .+ sum(y0))
            isempty(_y0) ? push!(y0, p.N) : (y0 = vcat(y0, _y0))
        end
    else
        @assert typeof(P) <: AbstractPopulation
        X, Y, y0 = _raster_populations(P, t; populations = populations)
    end
    names = isnothing(names) ? ["pop_$i" for i = 1:length(P)] : names

    X, Y = _resample_spikes(X, Y)
    X = X ./ s

    plt = scatter!(
        p,
        X[1:every:end],
        Y[1:every:end],
        m = (1, :black),
        leg = :none,
        xaxis = ("Time (s)", (0, Inf)),
        yaxis = ("Neuron",),
        label = "",
    )
    !isnothing(t) && plot!(xlims = t ./ s)
    plot!(yticks = (cumsum(y0)[1:end-1] .+ (y0./2)[2:end], names), yrotation = 45)
    y0 = y0[2:(end-1)]
    !isempty(y0) && hline!(plt, cumsum(y0), linecolor = :red, label = "")
    plot!(plt; kwargs...)
    return plt
end

function _raster_populations(
    p,
    t = nothing;
    populations::Vector{T},
) where {T<:AbstractVector}
    all_spiketimes = spiketimes(p)
    x, y = Float32[], Float32[]
    y0 = Int32[0]
    for pop in populations
        spiketimes_pop = all_spiketimes[pop] ## population spiketimes
        for n in eachindex(spiketimes_pop) ## neuron spiketimes
            for st in spiketimes_pop[n] ## spiketime
                if isnothing(st) || (st > t[1] && st < t[2])
                    push!(x, st)
                    push!(y, n + cumsum(y0)[end])
                end
            end
        end
        push!(y0, length(spiketimes_pop))
    end
    return x, y, y0
end

function _raster(spiketimes::Spiketimes, t = nothing)
    t, X, Y = t[[1, end]], Float32[], Float32[]
    for n in eachindex(spiketimes)
        for st in spiketimes[n]
            if isnothing(st) || (st > t[1] && st < t[2])
                push!(X, st)
                push!(Y, n)
            end
        end
    end
    return X, Y
end

function _resample_spikes(X, Y)
    if length(X) > 200_000
        s = ceil(Int, length(X) / 200_000)
        points = Vector{Int}(eachindex(X))
        points = sample(points, 200_000, replace = false)
        X = X[points]
        Y = Y[points]
        @warn "Subsampling raster plot, 1 out of $s spikes"
    end
    return X, Y
end



function _raster(p::T, interval = nothing) where {T<:Union{AbstractPopulation, AbstractStimulus}}
    !haskey(p.records, :fire) && @error "No fire record found in population $(p.name)"
    fire = p.records[:fire]
    x, y = Float32[], Float32[]
    y0 = Int32[]
    # which time to plot
    for i in eachindex(fire[:time])
        t = fire[:time][i]
        # which neurons to plot
        for n in fire[:neurons][i]
            if isnothing(interval) || (t > interval[1] && t < interval[2])
                push!(x, t)
                push!(y, n)
            end
        end
    end
    return x, y, y0
end

## Vector plot

function vecplot(p, sym; kwargs...)
    vecplot!(plot(), p, sym; kwargs...)
end

function vecplot(p, sym::Vector{Symbol}; kwargs...)
    my_plot = plot()
    for s in sym
        vecplot!(my_plot, p, s; label=string(s), kwargs...)
    end
    plot!(my_plot, ylims=:auto)
    return my_plot
end

function vecplot(P::Array, sym; kwargs...)
    plts = [vecplot(p, sym; kwargs...) for p in P]
    N = length(plts)
    plot(plts..., size = (600, 400N), layout = (N, 1))
end

function _match_r(r, r_v)
    r = isnothing(r) ? range(r_v[1], r_v[end]) : r
    r[end] > r_v[end] &&
        throw(ArgumentError("The end time is greater than the record time"))
    r[1] < r_v[1] && throw(ArgumentError("The start time is less than the record time"))
    return r
end

function vecplot!(
    my_plot,
    p,
    sym;
    neurons = nothing,
    pop_average = false,
    interval = nothing,
    r = nothing,
    sym_id = nothing,
    factor = 1.0f0,
    kwargs...,
)
    # get the record and its sampling rate
    y, r_v = interpolated_record(p, sym)
    r = isnothing(interval) ? r : interval
    r = _match_r(r, r_v)


    neurons = isnothing(neurons) ? axes(y, 1) : neurons
    neurons = isa(neurons, Int) ? [neurons] : neurons

    # check if the record is     a vector or a matrix
    if ndims(y) == 3
        isnothing(sym_id) && (throw(
            ArgumentError(
                "The record is a matrix, please specify the index ($sym_id) of the matrix to plot with `sym_id`",
            ),
        ))
        y = y[neurons, sym_id, r]
    else
        y = y[neurons, r]
    end

    ribbon = pop_average ? std(y, dims = 1) : nothing
    y = pop_average ? mean(y, dims = 1) : y

    @info "Vector plot in: $(r[1])ms to $(round(Int, r[end]))ms"
    return plot!(
        my_plot,
        r ./ 1000,
        y' .* factor,
        ribbon = ribbon,
        leg = :none,
        xaxis = ("t", extrema(r ./ 1000)),
        yaxis = (string(sym), extrema(y));
        lw = 3,
        kwargs...,
    )
end

function vecplot(P, syms::Array; kwargs...)
    plts = [vecplot(P, sym; kwargs...) for sym in syms]
    N = length(plts)
    plot(plts..., size = (600, 400N), layout = (N, 1))
end

export raster, raster!, vecplot, vecplot!

##

function raster_firing(model; path=nothing, τ=20ms, every=1)
    fr_average, r, labels = firing_rate(model.pop, interval=0s:get_time(model), τ=τ, pop_average=true)
    pr = raster(model.pop, (get_time(model)-5s):get_time(model), every=every, size=(1200, 1000))
    pf = plot(r, fr_average, labels=hcat(labels...), xlabel="Time (s)", ylabel="Firing rate (Hz)")
    presults = plot(pr, pf, layout=(2,1))
    if !isnothing(path)
        savefig(presults, path)        
    end
end


export raster, vecplot, vecplot!, vecplot, dendrite_gplot, soma_gplot, raster_firing



## Rateplot

function rateplot(p, sym)
    r = getrecord(p, sym)
    R = hcat(r...)
end

function rateplot(P::Array, sym)
    R = vcat([rateplot(p, sym) for p in P]...)
    y0 = [p.N for p in P][2:(end-1)]
    plt = heatmap(R, leg = :none)
    !isempty(y0) && hline!(plt, cumsum(y0), line = (:black, 1))
    plt
end


function if_curve(model, current; neuron = 1, dt = 0.1ms, duration = 1second)
    E = model(neuron)
    monitor!(E, [:fire])
    f = Float32[]
    for I in current
        clear_records!(E)
        E.I = [I]
        SNN.sim!([E], []; dt = dt, duration = duration)
        push!(f, activity(E))
    end
    plot(current, f)
end

# export density
# function density(p, sym)
#   X = getrecord(p, sym)
#   t = dt*(1:length(X))
#   xmin, xmax = extrema(vcat(X...))
#   edge = linspace(xmin, xmax, 100)
#   c = center(edge)
#   ρ = [fit(Histogram, x, edge).weights |> reverse |> float for x in X] |> x->hcat(x...)
#   ρ = smooth(ρ, windowsize(p), 2)
#   ρ ./= sum(ρ, 1)
#   surface(t, c, ρ, ylabel="p")
# end
# function density(P::Array, sym)
#   plts = [density(p, sym) for p in P]
#   plot(plts..., layout=(length(plts),1))
# end


# function windowsize(p)
#     A = sum.(p.records[:fire]) / length(p.N)
#     W = round(Int32, 0.5p.N / mean(A)) # filter window, unit=1
# end

# function density(p, sym)
#     X = getrecord(p, sym)
#     t = 1:length(X)
#     xmin, xmax = extrema(vcat(X...))
#     edge = linspace(xmin, xmax, 50)
#     c = center(edge)
#     ρ = [fit(Histogram, x, edge).weights |> float for x in X] |> x -> hcat(x...)
#     ρ = smooth(ρ, windowsize(p), 2)
#     ρ ./= sum(ρ, 1)
#     p = @gif for t = 1:length(X)
#         bar(c, ρ[:, t], leg = false, xlabel = string(sym), yaxis = ("p", extrema(ρ)))
#     end
#     is_windows() && run(`powershell start $(p.filename)`)
#     is_unix() && run(`xdg-open $(p.filename)`)
#     p
# end

# function activity(p)
#     A = sum.(p.records[:fire]) / length(p.N)
#     W = windowsize(p)
#     A = smooth(A, W)
# end

# function activity(P::Array)
#     A = activity.(P)
#     t = 1:length(P[1].records[:fire])
#     plot(t, A, leg = :none, xaxis = ("t",), yaxis = ("A", (0, Inf)))
# end
