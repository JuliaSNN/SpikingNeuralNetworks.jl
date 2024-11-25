using .Plots, Statistics
import Interpolations: scale, interpolate, BSpline, Linear

## Raster plot


function raster(P, t = nothing, dt = 0.1ms; populations=nothing, names=nothing, kwargs...)
    @show t
    t = t[[1,end]]
    if isnothing(populations)
        y0 = Int32[0]
        X = Float32[]
        Y = Float32[]
        names = Vector{String}()
        P = typeof(P)<: AbstractPopulation ? [P] : [getfield(P, k) for k in keys(P)]
        for p in P
            x, y, _y0= _raster(p, t) 
            push!(names, p.name)
            append!(X, x)
            append!(Y, y .+ sum(y0))
            isempty(_y0) ? push!(y0, p.N) : (y0 = vcat(y0, _y0))
        end
    else
        @assert typeof(P)<: AbstractPopulation 
        X, Y, y0 = _raster_populations(P, t; populations = populations)
    end
    names = isnothing(names) ? ["pop_$i" for i in 1:length(P)] : names

    if length(X) > 200_000 
        s = ceil(Int, length(X) / 200_000)
        points = Vector{Int}(eachindex(X))
        points = sample(points, 200_000, replace = false)
        X = X[points]
        Y = Y[points]
        @warn "Subsampling raster plot, 1 out of $s spikes"
    end
    plt = scatter(
        X,
        Y,
        m = (1, :black),
        leg = :none,
        xaxis = ("Time (ms)", (0, Inf)),
        yaxis = ("Neuron",),
    )
    !isnothing(t) && plot!(xlims = t)
    @show y0
    @show (cumsum(y0) .+ y0 ./ 2, names)
    @show (cumsum(y0)[1:end-1] .+ (y0 ./ 2)[2:end], names)
    plot!(yticks = (cumsum(y0)[1:end-1] .+ (y0 ./ 2)[2:end], names), yrotation=45)
    y0 = y0[2:(end-1)]
    !isempty(y0) && hline!(plt, cumsum(y0), linecolor = :red)
    plot!(plt; kwargs...)
    return plt
end

function _raster_populations(p, interval = nothing; populations::Vector{T} ) where T<: AbstractVector
    all_spiketimes = spiketimes(p)
    x, y = Float32[], Float32[]
    y0 = Int32[0]
    for pop in populations
        spiketimes_pop = all_spiketimes[pop] ## population spiketimes
        for n in eachindex(spiketimes_pop) ## neuron spiketimes
            for t in spiketimes_pop[n] ## spiketime
                if isnothing(interval) || (t > interval[1] && t < interval[2])
                    push!(x, t)
                    push!(y, n + cumsum(y0)[end])
                end
            end
        end
        push!(y0, length(spiketimes_pop))
    end
    return x, y, y0
end


function _raster(p::T, interval = nothing) where T<: AbstractPopulation
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

function vecplot(P::Array, sym; kwargs...)
    plts = [vecplot(p, sym; kwargs...) for p in P]
    N = length(plts)
    plot(plts..., size = (600, 400N), layout = (N, 1))
end

function vecplot!(
    my_plot,
    p,
    sym;
    neurons = nothing,
    pop_average = false,
    r::AbstractArray{T} = 0:-1,
    sym_id = nothing,
    kwargs...,
) where {T<:Real}
    # get the record and its sampling rate
    y, r_v = interpolated_record(p, sym, sym_id)

    # set the plot range
    r = isempty(r) ? r_v : r
    r[end] > r_v[end] && throw(ArgumentError("The end time is greater than the record time"))
    r[1] < r_v[1] && throw(ArgumentError("The start time is less than the record time"))
    
    # get the neurons to plot
    neurons = isnothing(neurons) ? axes(y, 1) : neurons
    neurons = isa(neurons,Int) ? [neurons] : neurons
    if pop_average
        y = mean(y, dims = 1)
        neurons = [1:1]
    end
    
    @info "Vector plot in: $(r[1])s to $(round(Int, r[end]))s"
    return plot!(
        my_plot,
        r./1000,
        y[neurons, r]',
        leg = :none,
        xaxis = ("t", extrema(r./1000)),
        yaxis = (string(sym), extrema(y));
        kwargs...,
    )
end


# function vecplot!(P::Array, sym; kwargs...)
#     plts = [vecplot(p, sym; kwargs...) for p in P]
#     my_plot = plot()
#     for p in P
#         vecplot!(my_plot, p, sym; kwargs...)
#     end
#     plot!(my_plot; kwargs...)
#     return my_plot
# end

function vecplot(P, syms::Array; kwargs...)
    plts = [vecplot(P, sym; kwargs...) for sym in syms]
    N = length(plts)
    plot(plts..., size = (600, 400N), layout = (N, 1))
end

##
function dendrite_gplot(population, target; sym_id=1, r, dt, param=:dend_syn, nmda=true, kwargs...)
    syn = getfield(population, param)
    if nmda
        @unpack mg, b, k = getfield(population, :NMDA)
    end
    r_dt =  r[2:(end-1)] |> r-> round.(Int, r ./ dt)[1:(end-1)]
    v_sym = Symbol("v_", target) 
    g_sym = Symbol("g_", target)
    indices =  haskey(population.records[:indices], g_sym) ? population.records[:indices][g_sym] : 1:population.N
    v = getvariable(population, v_sym)[indices, r_dt]
    g = getvariable(population, g_sym)[:,:, r_dt]

    @assert length(axes(g,1)) == length(axes(v,1))
    @assert length(axes(g,2)) == length(syn) "Syn size: $(length(syn)) != $(length(axes(g,2)))"
    @assert length(axes(g,3)) == length(axes(v,2))
    curr = zeros(size(g))
    for i in axes(g,3)
        for r in axes(g,2)
            @unpack gsyn, E_rev, nmda = syn[r]
            for n in axes(g,1)
                if nmda > 0.
                    curr[n,r,i] = - gsyn * g[n,r,i] * (v[n,i]-E_rev)/ (1.0f0 + (mg / b) * SNN.exp32(k * v[n,i]))
                else
                    curr[n,r,i] = - gsyn * g[n,r,i] * (v[n,i]-E_rev)
                end
            end
        end
    end
    curr .= curr ./1000

    ylims =abs.(maximum(abs.(curr[sym_id,:,:]))) |> x->(-x, x)
    plot(r_dt.*dt, curr[sym_id,1,:].+curr[sym_id,2,:], label="Glu")
    plot!(r_dt*dt, curr[sym_id,3,:].+curr[sym_id,4,:], label="GABA")
    plot!(ylims=ylims, xlabel="Time (ms)", ylabel="Syn. curr. dendrite (μA)")
    hline!([0.0], c=:black, label="") 
    plot!(;kwargs...)
end

function soma_gplot(population, target;sym_id=1, r, dt, param=:soma_syn, nmda=true, kwargs...)
    syn = getfield(population, param)
    if nmda
        @unpack mg, b, k = getfield(population, :NMDA)
    end
    r_dt =  r[2:(end-1)] |> r-> round.(Int, r ./ dt)[1:(end-1)]
    v_sym = Symbol("v_", target) 
    ge_sym = Symbol("ge_", target)
    gi_sym = Symbol("gi_", target)
    indices =  haskey(population.records[:indices], ge_sym) ? population.records[:indices][ge_sym] : 1:population.N
    v = getvariable(population, v_sym)[indices, r_dt]
    g = zeros(length(indices), 2, length(r_dt))
    g[:,1,:] = getvariable(population, ge_sym)[:,r_dt]
    g[:,2,:] = getvariable(population, gi_sym)[:,r_dt]

    @assert length(axes(g,1)) == length(axes(v,1))
    @assert length(axes(g,2)) == length(syn) "Syn size: $(length(syn)) != $(length(axes(g,2)))"
    @assert length(axes(g,3)) == length(axes(v,2))
    curr = zeros(size(g))
    for i in axes(g,3)
        for r in axes(g,2)
            @unpack gsyn, E_rev, nmda = syn[r]
            for n in axes(g,1)
                curr[n,r,i] = - gsyn * g[n,r,i] * (v[n,i]-E_rev)
            end
        end
    end
    curr .= curr ./1000


    ylims =abs.(maximum(abs.(curr[sym_id,:,:]))) |> x->(-x, x)
    plot(r_dt.*dt, curr[sym_id,1,:], label="Glu")
    plot!(r_dt*dt, curr[sym_id,2,:], label="GABA")
    plot!(ylims=ylims, xlabel="Time (ms)", ylabel="Syn. curr. soma (μA)")
    hline!([0.0], c=:black, label="") 
    plot!(;kwargs...)
end

function gegi_plot(population; r, dt, param=:soma_syn, nmda=true, kwargs...)
    syn = getfield(population, param)
    if nmda
        @unpack mg, b, k = getfield(population, :NMDA)
    end
    r_dt =  r[2:(end-1)] |> r-> round.(Int, r ./ dt)[1:(end-1)]
    indices =  haskey(population.records[:indices], sym) ? population.records[:indices][sym] : 1:population.N
    v_sym = :v
    ge_sym = :ge
    gi_sym = :gi
    v = getvariable(population, v_sym)[indices, r_dt]
    ge = getvariable(population, ge_sym)[:, r_dt]
    gi = getvariable(population, gi_sym)[:, r_dt]

    # curr = zeros(2, size(ge,2), )
    # for i in axes(g,3)
    #     for r in axes(g,2)
    #         @unpack gsyn, E_rev, nmda = syn[r]
    #         for n in axes(g,1)
    #             if nmda > 0.
    #                 curr[n,r,i] = - gsyn * g[n,r,i] * (v[n,i]-E_rev)/ (1.0f0 + (mg / b) * SNN.exp32(k * v[n,i]))
    #             else
    #                 curr[n,r,i] = - gsyn * g[n,r,i] * (v[n,i]-E_rev)
    #             end
    #         end
    #     end
    # end
    # curr .= curr ./1000

    plot(r_dt.*dt, curr[1,1,:].+curr[1,2,:], label="Glu")
    plot!(r_dt*dt, curr[1,3,:].+curr[1,4,:], label="GABA")
    plot!(ylims=(-maximum(abs.(curr)), maximum(abs.(curr))), xlabel="Time (ms)", ylabel="Syn Curr dendrite (μA)")
    hline!([0.0], c=:black, label="") 
end
##


export raster, vecplot, vecplot!, vecplot, dendrite_gplot, soma_gplot


## Matrix plot

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
    monitor(E, [:fire])
    f = Float32[]
    for I in current
        clear_records(E)
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

