# Define each of the network recurrent assemblies
function attractor_model(config;)
    function get_subnet(name, config)
        N = config.N
        # Create dendrites for each neuron
        E = SNN.AdExNeuron(N = N, param = config.adex, name = "Exc_$name")
        # Define interneurons 
        I = SNN.IF(;
            N = N ÷ 4,
            param = SNN.IFParameter(τm = 20ms, El = -50mV),
            name = "Inh_$name",
        )
        # Define synaptic interactions between neurons and interneurons
        E_to_I = SNN.SpikingSynapse(E, I, :he; config.E_to_I..., name = "E_to_I_$name")
        E_to_E = SNN.SpikingSynapse(E, E, :he; config.E_to_E..., name = "E_to_E_$name")
        I_to_I = SNN.SpikingSynapse(I, I, :hi; config.I_to_I..., name = "I_to_I_$name")
        I_to_E = SNN.SpikingSynapse(
            I,
            E,
            :hi;
            config.I_to_E...,
            param = config.istdp,
            name = "I_to_E_$name",
        )
        norm = SNN.SynapseNormalization(E, [E_to_E], param = SNN.AdditiveNorm(τ = 10ms))

        # Store neurons and synapses into a dictionary
        pop = SNN.@symdict E I
        syn = SNN.@symdict I_to_E E_to_I E_to_E norm I_to_I
        noise = SNN.PoissonStimulus(E, :he, param = config.noise, neurons = :ALL)
        # Return the network as a tuple
        SNN.monitor([E, I], [:fire])
        SNN.monitor(I_to_E, [:W], sr = 10Hz)
        SNN.merge_models(pop, syn, noise = noise, silent = true)
    end

    n_assemblies = config.n_assemblies
    subnets = Dict(Symbol("sub_$n") => get_subnet(n, config) for n = 1:n_assemblies)
    syns = Dict{Symbol,Any}()
    for i in eachindex(subnets)
        for j in eachindex(subnets)
            i == j && continue
            symbolEI = Symbol(string("$(i)E_to_$(j)I_lateral"))
            synapseEI = SNN.SpikingSynapse(
                subnets[i].pop.E,
                subnets[j].pop.I,
                :he;
                config.lateral_EI...,
            )
            push!(syns, symbolEI => synapseEI)
            symbolEE = Symbol(string("$(i)E_to_$(j)E_lateral"))
            synapseEE = SNN.SpikingSynapse(
                subnets[i].pop.E,
                subnets[j].pop.E,
                :he;
                config.lateral_EE...,
            )
            push!(syns, symbolEE => synapseEE)
        end
    end
    SNN.merge_models(subnets, syns)
end

function add_stimulus(network, population, interval)
    @unpack ext_stim = config
    trig_param = PoissonStimulusInterval(rate = fill(ext_stim, 400), intervals = [interval])
    name = Symbol("stim_$(population)_$(randstring(2))")
    trigger = Dict{Symbol,Any}(
        name => SNN.PoissonStimulus(
            getfield(network.pop, population),
            :he,
            param = trig_param,
            neurons = :ALL,
            name = string(name),
        ),
    )
    SNN.merge_models(network, trigger, silent = true)
end

function test_istdp(config)
    model = attractor_model(config)
    mytime = train!(model = model, duration = config.warmup, pbar = true, dt = 0.125)
    clear_records(model)
    mytime = Time()
    for (pop, interval) in config.intervals
        model = add_stimulus(model, pop, interval)
    end
    train!(
        model = model,
        duration = config.duration,
        pbar = true,
        dt = 0.125,
        time = mytime,
    )
    return model
end

#
function iSTDP_activity(model, istdp, config; interval = 1s:20ms:10s)
    i_to_e = SNN.filter_items(model.syn, condition = p->occursin("I_to_E", p.name))
    w_i = map(eachindex(i_to_e)) do i
        w, r_t = record(i_to_e[i], :W, interpolate = true)
        mean(w, dims = 1)[1, :]
    end |> collect

    _, r_t = record(i_to_e[1], :W, interpolate = true)
    p1 = plot(
        r_t ./ 1000,
        w_i,
        xlabel = "Time (s)",
        ylabel = "Synaptic weight",
        legend = :topleft,
        title = "I to E synapse",
        labels = ["pop 1" "pop 2" "pop 3" "pop 4"],
        lw = 4,
    )


    Epop = SNN.filter_items(model.pop, condition = p->occursin("E", p.name))
    fr, r, names =
        firing_rate(model.pop, interval = 0s:10ms:10s, τ = 50ms, pop_average = true)
    c = [:blue :blue :red :red]
    ls = [:solid :dash :solid :dash]
    p2 = plot(
        r ./ 1000,
        fr[[1, 3]],
        xlabel = "Time (s)",
        ylabel = "Firing rate (Hz)",
        legend = :topleft,
        title = "Firing rate of the exc. pop",
        lw = 4,
        labels = hcat(names[[1, 3]]...),
    )#, yscale=:log10, ylims=(0.1,50))
    vline!(
        [x[2] for x in config.intervals] ./ 1000,
        color = :black,
        linestyle = :dash,
        label = "",
    )
    p3 = plot(
        r ./ 1000,
        fr[[2, 4]],
        xlabel = "Time (s)",
        ylabel = "Firing rate (Hz)",
        legend = :topleft,
        title = "Firing rate of the inh. pop",
        lw = 4,
        labels = hcat(names[[2, 4]]...),
    )#, yscale=:log10, ylims=(0.1,50))
    vline!(
        [x[2] for x in config.intervals] ./ 1000,
        color = :black,
        linestyle = :dash,
        label = "",
    )
    # p3 = SNN.stdp_kernel(istdp, fill=false)
    p4 = SNN.raster(model.pop, interval, every = 5)

    # layout = @layout [ grid(3,1), grid(2,1) ]
    return plot(p1, p2, p4, p3, layout = (2, 2), size = (800, 800), margin = 5Plots.mm)
    # plot(p4, p2, layout=(2,1), size=(800,800), margin=5Plots.mm)
    # return p1, p2, p3, p4
end
