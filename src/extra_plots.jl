# using StatsBase
## Dendrite and soma plot
"""
    dendrite_gplot(population, target; sym_id=1, r, dt, param=:dend_syn, nmda=true, kwargs...)

    Plot the synaptic current in the dendrite of a population of neurons. 
    The function uses the synaptic conductance and the membrane potential to calculate the synaptic current.
    
    Parameters
    ----------
    population : AbstractPopulation
        The population of neurons to plot
    target : Symbol
        The target of the plot, either `:d` for single dendrite or `:d1/:d2` 
    neuron : Int
        The neuron to plot
    r : Array{Int}
        The time range to plot
    nmda : Bool
        If true, the NMDA conductance is used to calculate the synaptic current
    kwargs... : Any

"""
function dendrite_gplot(population, target; neuron=1, r, param=:dend_syn, nmda=true, kwargs...)
    syn = getfield(population, param)
    if nmda
        @unpack mg, b, k = getfield(population, :NMDA)
    end
    # r_dt =  r[2:(end-1)] |> r-> round.(Int, r ./ dt)[1:(end-1)]
    v_sym = Symbol("v_", target) 
    g_sym = Symbol("g_", target)
    indices =  haskey(population.records[:indices], g_sym) ? population.records[:indices][g_sym] : 1:population.N
    v, r_v= interpolated_record(population, v_sym)
    g, r_v = interpolated_record(population, g_sym)
    r = _match_r(r, r_v)
    v = Float32.(v[indices, r])
    g = Float32.(g[:, :, r])

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
    @info size(curr), size(r)

    ylims =abs.(maximum(abs.(curr[neuron,:,:]))) |> x->(-x, x)
    plot(r, curr[neuron,1,:].+curr[neuron,2,:], label="Glu")
    plot!(r, curr[neuron,3,:].+curr[neuron,4,:], label="GABA")
    plot!(ylims=ylims, xlabel="Time (ms)", ylabel="Syn. curr. dendrite (μA)")
    hline!([0.0], c=:black, label="") 
    plot!(;kwargs...)
end

"""
    soma_gplot( population, target; neuron=1, r, dt, param=:soma_syn, nmda=true, ax=plot(), kwargs...)

    Plot the synaptic current in the soma of a population of neurons.
    The function uses the synaptic conductance and the membrane potential to calculate the synaptic current.
    
    Parameters
    ----------
    population : AbstractPopulation
        The population of neurons to plot
    neuron : Int
        The neuron to plot
    r : Array{Int}:
        The time range to plot
    param : Symbol
        The parameter to use for the synaptic conductance
    ax : Plots.Plot
        Plot over the current axis 
"""
function soma_gplot( population; neuron=1, r, param=:soma_syn, ax=plot(), kwargs...)
    syn = getfield(population, param)
    v_sym = :v_s
    ge_sym = :ge_s
    gi_sym = :gi_s
    indices =  haskey(population.records[:indices], ge_sym) ? population.records[:indices][ge_sym] : 1:population.N
    v, r_v= interpolated_record(population, v_sym)
    ge, r_v = interpolated_record(population, ge_sym)
    gi, r_v = interpolated_record(population, gi_sym)

    r = _match_r(r, r_v)
    v = Float32.(v[indices, r])
    ge = Float32.(ge[:,  r])
    gi = Float32.(gi[:,  r])

    @assert length(axes(ge,1)) == length(axes(v,1))
    @assert length(axes(ge,2)) == length(axes(v,2))
    curr = zeros(size(ge,1), 2, size(ge,2))
    r = _match_r(r, r_v)
    for i in axes(ge,2)
        for n in axes(ge,1)
            @unpack gsyn, E_rev, nmda = syn[1]
            curr[n,1,i] = - gsyn * ge[n,i] * (v[n,i]-E_rev)
            @unpack gsyn, E_rev, nmda = syn[2]
            curr[n,2,i] = - gsyn * gi[n,i] * (v[n,i]-E_rev)
        end
    end
    curr .= curr ./1000

    plot!(ax, r, curr[neuron,1,:], label="Glu soma")
    plot!(r, curr[neuron,2,:], label="GABA soma")
    plot!(ylims=:auto, xlabel="Time (ms)", ylabel="Syn. curr. (μA)")
    hline!([0.0], c=:black, label="") 
    plot!(;kwargs...)
end
"""
    plot_activity(network, Trange)

Plot the activity of a spiking neural network with one dendritic excitatory population and two inhibitory populations. The function plots the firing rate of the populations, the membrane potential of the neurons, the synaptic conductance in the dendrite, the synaptic current in the dendrite, and the raster plot of the excitatory population.

Arguments:
- `network`: The spiking neural network object.
- `Trange`: The time range for plotting.

Returns:
- Nothing.

Example:
"""
function plot_activity(network, Trange; conductance=false)
    frE, interval  = SNN.firing_rate(network.pop.E,  interval = Trange, τ=10ms)
    frI1, interval = SNN.firing_rate(network.pop.I1, interval = Trange, τ=10ms)
    frI2, interval = SNN.firing_rate(network.pop.I2, interval = Trange, τ=10ms)
    pr = plot(xlabel = "Time (ms)", ylabel = "Firing rate (Hz)")
    # plot!(Trange, mean(frE[:,Trange], dims=1)[1,:], label = "E", c = :black)
    # plot!(Trange, mean(frI1[:,Trange], dims=1)[1,:], label = "I1", c = :red)
    # plot!( Trange,mean(frI2[:,Trange], dims=1)[1,:], label = "I2", c = :green)
    plot!(margin = 5Plots.mm, xlabel="")
    pv = nothing
    try
        pv =SNN.vecplot(network.pop.E, :v_d, sym_id=1, r = Trange,  pop_average = true, label="dendrite")
    catch
        pv =SNN.vecplot(network.pop.E, :v_d1, r = Trange,  pop_average = true, label="dendrite")
    end
    SNN.vecplot!(pv, network.pop.E, :v_s, r = Trange, pop_average = true, label="soma")
    plot!(ylims=:auto, margin = 5Plots.mm, ylabel = "Membrane potential (mV)", legend=true, xlabel="")
    rplot = SNN.raster(network.pop, Trange, size=(900,500), margin=5Plots.mm, xlabel="")

    p5 = histogram(average_firing_rate(network.pop.E), c=:black, lc=:black, label="Firing rate (Hz)")
    ## Conductance
    if conductance 
        dgplot = dendrite_gplot(network.pop.E, :d, r=Trange, dt=0.125, margin=5Plots.mm, xlabel="")
        soma_gplot(network.pop.E, r=Trange, margin=5Plots.mm, xlabel="", ax=dgplot)
        layout = @layout  [ 
                    c{0.25h}
                    e{0.25h}
                    a{0.25h}
                    d{0.25h}]
        return plot(pr, rplot,pv,  dgplot, layout=layout, size=(900, 1200), topmargn=0Plots.mm, bottommargin=0Plots.mm, bgcolorlegend=:transparent, fgcolorlegend=:transparent)
    else
        layout = @layout  [ 
            c{0.3h}
            e{0.4h}
            d{1.0h} e{0.4w}]
        return plot(pr, rplot,pv, p5, layout=layout, size=(900, 1200), topmargn=0Plots.mm, bottommargin=0Plots.mm, bgcolorlegend=:transparent, fgcolorlegend=:transparent)
    end
end

"""
    plot_weights(network)

Plot the synaptic weights of:
    - inhibitory to excitatory neurons
    - correlation of synaptic weights between inhibitory and excitatory neurons
    - distribution of firing rates of the network

# Arguments
- `network`: The spiking neural network object.

# Returns
- `plot`: The plot object.

"""
function plot_weights(network)
    W = network.syn.I1_to_E.W
    h_I1E = histogram(W, bins=minimum(W):maximum(W)/200:maximum(W)+1, title = "Synaptic weights from I1 to E",  xlabel="Synaptic weight", ylabel="Number of synapses", yticks=:none, c=:black)
    W = network.syn.I2_to_E.W
    h_I2E = histogram(W, bins=minimum(W):maximum(W)/200:maximum(W)+1, title = "Synaptic weights from I2 to E", xlabel="Synaptic weight", ylabel="Number of synapses", yticks=:none, c=:black)
    sc_w = scatter(network.syn.I2_to_E.W, network.syn.I1_to_E.W,  xlabel="Synaptic weight from I2 to E", ylabel="Synaptic weight from I1 to E", alpha=0.01, c=:black)
    frE= SNN.average_firing_rate(network.pop.E, interval = Trange)
    sc_fr=histogram(frE, c=:black, label="E", xlabel="Firing rate (Hz)",bins=-0.5:0.2:12, ylabel="Number of neurons")
    layout = @layout  [ 
                grid(2,2)
                ]
    return plot(h_I2E, h_I1E, sc_w, sc_fr, layout=layout, size=(800, 600), legend=false, margin=5Plots.mm)
end

export soma_gplot, dendrite_gplot, plot_activity, plot_weights
## 

## 
"""
    stp_plot(model, interval, assemblies)

    Plot the activity of a spiking neural network with short-term plasticity. The function plots the membrane potential, the firing rate, the synaptic weights, and the raster plot of the excitatory population.
"""
function stp_plot(model, interval, assemblies, stimuli=[]; every=10)
    @unpack pop, syn = model
    ρ, r_t= SNN.interpolated_record(syn.EE, :ρ)
    w,r_t= SNN.interpolated_record(syn.EE, :W)
    weff = ρ.*w ./ maximum(w)
    in_assembly = 1:length(indices(syn.EE, assemblies[1].cells, assemblies[1].cells))
    out_assembly = length(in_assembly)+1:size(weff,1)
    p12 = SNN.raster(pop, interval, yrotation=90, every=10)
    p11 = plot(SNN.vecplot( pop.I, :v, r=interval, neurons=1, pop_average=true, label="Excitatory", ylabel="", xlabel=""),
                SNN.vecplot(pop.E, :v, r=interval, neurons=1, pop_average=true, label="Inhibitory", ylabel="Membrane potential (mV)"), layout=(2,1), topmargin=0Plots.mm, bottommargin=0Plots.mm)
    p1 = plot(p11, p12, layout=(1,2), size=(800,400), margin=5Plots.mm, legend=:topleft)
    fr, interval = SNN.firing_rate(pop, interval=interval)
    interval
    p2 = plot(interval./1000, mean(fr[1], dims=1)', label="Excitatory", lw=3)
    plot!(interval./1000, mean(fr[2], dims=1)', label="Inhibitory", lw=3)
    for a in assemblies
        plot!(interval./1000, mean(fr[1][a.cells, interval], dims=1)', label="Assembly $(a.id)", lw=3)
    end
    # plot!(interval./1000, mean(fr[1][assemblies[1].cells, interval], dims=1)', label="Assembly", lw=3)
    plot!(ylabel="Firing rate (Hz)")
    p3 = plot(r_t./1000, mean(weff[out_assembly,:], dims=1)', c=:black, lw=4, ylims=:auto, label="w_{base}", ls=:dash)
    p3 = plot!(r_t./1000, mean(weff[in_assembly,:], dims=1)', c=:black, lw=4, ylims=:auto, label="w_{eff}")
    SNN.vecplot!(p3, syn.EE, :u, r=interval, dt=0.125, pop_average=true, ls=:dash, ribbon=false, c=:blue, label="")
    SNN.vecplot!(p3, syn.EE, :x, r=interval, dt=0.125, pop_average=true, ls=:dash, ribbon=false, c=:red, label="")
    interval
    SNN.vecplot!(p3, syn.EE, :u,  r=interval, dt=0.125, neurons=assemblies[1].cells, pop_average=true, label="u", c=:blue)
    SNN.vecplot!(p3, syn.EE, :x, r=interval, dt=0.125, neurons=assemblies[1].cells, pop_average=true, label="x", c=:red)
    plot!(p3, ylims=(0,1), legend=:topleft, ylabel="STP")
    p23 = plot(p2,p3)
    in_ass = [a.cells for a in assemblies]
    push!(in_ass, (pop.E.N - length(assemblies[1].cells) : pop.E.N))
    p = plot()
    rectangle(_start, _end) = Shape([_start, _start, _end, _end],
                                    [0, length(vcat(in_ass...)),length(vcat(in_ass...)), 0 ])
    for stim in stimuli
        plot!(rectangle(stim[1], stim[end]), c=:grey, opacity=.5, label="", lc=:transparent)
    end
    p4 = SNN.raster!(p, pop.E, interval, yrotation=90, populations=in_ass, every=every, names=["Assembly 1", "Assembly 2"])
    plot_network = plot!(p1, p23, p4, layout=(3,1), size=(1300,900), margin=5Plots.mm, legend=:topleft)
    return plot_network
end

"""
    plot_average_word_activity(sym, word, model, seq; target=:d, before=100ms, after=300ms, zscore=true)

    Plot the value of the `sym` variable for the cells associated to the `word` stimulus. 
    `cells = getcells(model.stim, seq.symbols.words[w], target)`

    Arguments:
    - `sym`: The variable to plot.
    - `word`: The word stimulus.
    - `model`: The spiking neural network model.
    - `seq`: The sequence of stimuli.
    - `target`: The target compartment (default=:d).
    - `before`: The time before the stimulus (default=100ms).
    - `after`: The time after the stimulus (default=300ms).
    - `zscore`: Whether to z-score the activity (default=true).
"""
function plot_average_word_activity(sym, word, model, seq; target=:d, before=100ms, after=300ms, zscore=true)
    membrane, r_v = SNN.interpolated_record(model.pop.E, sym)
    myintervals = sign_intervals(word, seq)
    Trange = -before:1ms:diff(myintervals[1])[1]+after
    activity = zeros(length(seq.symbols.words),size(Trange,1))
    for w in eachindex(seq.symbols.words)
        cells = getcells(model.stim, seq.symbols.words[w], :d)
        ave_fr = mean(membrane[cells, :])
        std_fr = std(membrane[cells, :])
        n = 0
        for myinterval in myintervals
            _range = myinterval[1]-before:1ms:myinterval[2]+after
            _range[end] > r_v[end] && continue
            v = mean(membrane[cells, _range], dims=1)[1,:]
            activity[w, :] += zscore ? (v .- ave_fr)./std_fr : v
            n+=1
        end
        activity[w, :] ./= n
    end
    plot(Trange, activity[:,:]', label=hcat(string.(seq.symbols.words)...), xlabel="Time (ms)", ylabel="Membrane potential (mV)", title="")
    vline!([0, diff(myintervals[1])[1]], c=:black, ls=:dash, label="")
    word_id = findfirst(seq.symbols.words .== word)
    plot!(Trange, activity[word_id,:], c=:black, label=string(word), lw=5, )
end

export plot_average_word_activity

function get_updown_hist(model)
	hist = [fit(Histogram,getvariable(model.pop.E, :v_s)[x, :], -75:1:-30).weights for x in 1:1000]
	hist = hcat(hist...).+0.001
	zscored = fit(ZScoreTransform, hist.+0.001, dims=1) |> x->StatsBase.transform(x,hist)
	pca_transform = MultivariateStats.fit(PCA,zscored, maxoutdim=2)
	h = predict(pca_transform, zscored)'
	scatter(h[:,1], h[:,2], c=:blue, ms=2, subplot=1)
	pca1 = sort(1:1000, by = x->h[x,1])
	neurons = [pca1[50], pca1[500], pca1[950]]
	scatter!(h[neurons,1], h[neurons,2], c=:red, ms=5, subplot=1)
	histogram!(getvariable(model.pop.E, :v_s)[neurons[1], :], bins=-75:1:-30, normed=true, label="", 
			inset = (1, bbox(0,0.8,0.3,0.2)), subplot=2, frame=:none, c=:black)
	histogram!(getvariable(model.pop.E, :v_s)[neurons[2], :], bins=-75:1:-30, normed=true, label="", 
			inset = (1, bbox(0.35,0.8,0.3,0.2)), subplot=3, frame=:none, c=:black)
	histogram!(getvariable(model.pop.E, :v_s)[neurons[3], :], bins=-75:1:-30, normed=true, label="", 
			inset = (1, bbox(0.7,0.8,0.3,0.2)), subplot=4, frame=:none, c=:black)
	histogram!(getvariable(model.pop.E, :v_d)[neurons[1],1, :], bins=-75:1:-10, normed=true, label="", 
			inset = (1, bbox(0,0.,0.3,0.2)), subplot=5, frame=:none, c=:black)
	histogram!(getvariable(model.pop.E, :v_d)[neurons[2],1, :], bins=-75:1:-10, normed=true, label="", 
			inset = (1, bbox(0.35,0.,0.3,0.2)), subplot=6, frame=:none, c=:black)
	histogram!(getvariable(model.pop.E, :v_d)[neurons[3],1, :], bins=-75:1:-10, normed=true, label="", 
			inset = (1, bbox(0.7,0.,0.3,0.2)), subplot=7, frame=:none, c=:black)

	return plot!(ylims=(-4, 4), xlims=(-5, 5), xlabel="PC1", ylabel="PC2", legend=false, subplot=1, background=:white)
end

export get_updown_hist


export stp_plot, plot_weights, plot_activity, dendrite_gplot, soma_gplot

# ## conductance plot
# function gegi_plot(population; r, dt, param=:soma_syn, nmda=true, kwargs...)
#     syn = getfield(population, param)
#     if nmda
#         @unpack mg, b, k = getfield(population, :NMDA)
#     end
#     r_dt =  r[2:(end-1)] |> r-> round.(Int, r ./ dt)[1:(end-1)]
#     indices =  haskey(population.records[:indices], sym) ? population.records[:indices][sym] : 1:population.N
#     v_sym = :v
#     ge_sym = :ge
#     gi_sym = :gi
#     v = getvariable(population, v_sym)[indices, r_dt]
#     ge = getvariable(population, ge_sym)[:, r_dt]
#     gi = getvariable(population, gi_sym)[:, r_dt]

#     # curr = zeros(2, size(ge,2), )
#     # for i in axes(g,3)
#     #     for r in axes(g,2)
#     #         @unpack gsyn, E_rev, nmda = syn[r]
#     #         for n in axes(g,1)
#     #             if nmda > 0.
#     #                 curr[n,r,i] = - gsyn * g[n,r,i] * (v[n,i]-E_rev)/ (1.0f0 + (mg / b) * SNN.exp32(k * v[n,i]))
#     #             else
#     #                 curr[n,r,i] = - gsyn * g[n,r,i] * (v[n,i]-E_rev)
#     #             end
#     #         end
#     #     end
#     # end
#     # curr .= curr ./1000

#     plot(r_dt.*dt, curr[1,1,:].+curr[1,2,:], label="Glu")
#     plot!(r_dt*dt, curr[1,3,:].+curr[1,4,:], label="GABA")
#     plot!(ylims=(-maximum(abs.(curr)), maximum(abs.(curr))), xlabel="Time (ms)", ylabel="Syn Curr dendrite (μA)")
#     hline!([0.0], c=:black, label="") 
# end
# 