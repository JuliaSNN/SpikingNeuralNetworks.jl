using Plots
# FIXME: using StatsBase

function raster(p)
    fire = p.records[:fire]
    x, y = Float32[], Float32[]
    for t = eachindex(fire)
        for n in findall(fire[t])
            push!(x, t)
            push!(y, n)
        end
    end
    x, y
end

function raster(P::Array)
    y0 = Int32[0]
    X = Float32[]; Y = Float32[]
    for p in P
        x, y = raster(p)
        append!(X, x)
        append!(Y, y .+ sum(y0))
        push!(y0, p.N)
    end
    plt = scatter(X, Y, m = (1, :black), leg = :none,
                  xaxis=("t", (0, Inf)), yaxis = ("neuron",))
    y0 = y0[2:end-1]
    !isempty(y0) && hline!(plt, cumsum(y0), linecolor = :red)
    return plt
end

function vecplot(p, sym)
    v = getrecord(p, sym)
    y = hcat(v...)'
    x = 1:length(v)
    plot(x, y, leg = :none,
    xaxis=("t", extrema(x)),
    yaxis=(string(sym), extrema(y)))
end

function vecplot(P::Array, sym)
    plts = [vecplot(p, sym) for p in P]
    N = length(plts)
    plot(plts..., size = (600, 400N), layout = (N, 1))
end

function windowsize(p)
    A = sum.(p.records[:fire]) / length(p.N)
    W = round(Int32, 0.5p.N / mean(A)) # filter window, unit=1
end

function density(p, sym)
    X = getrecord(p, sym)
    t = 1:length(X)
    xmin, xmax = extrema(vcat(X...))
    edge = linspace(xmin, xmax, 50)
    c = center(edge)
    ρ = [fit(Histogram, x, edge).weights |> float for x in X] |> x -> hcat(x...)
    ρ = smooth(ρ, windowsize(p), 2)
    ρ ./= sum(ρ, 1)
    p = @gif for t = 1:length(X)
        bar(c, ρ[:, t], leg = false, xlabel = string(sym), yaxis = ("p", extrema(ρ)))
    end
    is_windows() && run(`powershell start $(p.filename)`)
    is_unix() && run(`xdg-open $(p.filename)`)
    p
end

function rateplot(p, sym)
    r = getrecord(p, sym)
    R = hcat(r...)
end

function rateplot(P::Array, sym)
    R = vcat([rateplot(p, sym) for p in P]...)
    y0 = [p.N for p in P][2:end-1]
    plt = heatmap(flipdim(R, 1), leg = :none)
    !isempty(y0) && hline!(plt, cumsum(y0), line = (:black, 1))
    plt
end

function activity(p)
    A = sum.(p.records[:fire]) / length(p.N)
    W = windowsize(p)
    A = smooth(A, W)
end

function activity(P::Array)
    A = activity.(P)
    t = 1:length(P[1].records[:fire])
    plot(t, A, leg=:none, xaxis=("t",), yaxis=("A", (0, Inf)))
end

function if_curve(model, current; neuron = 1, dt = 0.1ms, duration = 1second)
    E = model(neuron)
    monitor(E, [:fire])
    f = Float32[]
    for I = current
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
