## AdEx neuron with fixed external current connections with multiple receptors
E_uni = SNN.AdExParameter(; El = -50mV)
E_het = SNN.heterogeneous(E_uni, 800; τm = Normal(10.0f0, 2.0f0), b = Normal(60.0f0, 4.0f0))
E = SNN.Population(E_het, synapse = SNN.DoubleExpSynapse(); N = 800, name = "Excitatory")

I = SNN.Population(
    SNN.IFParameter(),
    synapse = SNN.SingleExpSynapse();
    N = 200,
    name = "Inhibitory",
    spike = SNN.PostSpike(),
)
EE = SNN.SpikingSynapse(E, E, :he; conn = (μ = 2, p = 0.02))
EI = SNN.SpikingSynapse(E, I, :ge; conn = (μ = 30, p = 0.02))
IE = SNN.SpikingSynapse(I, E, :hi; conn = (μ = 50, p = 0.02))
II = SNN.SpikingSynapse(I, I, :gi; conn = (μ = 10, p = 0.02))
model = SNN.compose(; E, I, EE, EI, IE, II)

SNN.monitor!(E, [(:ge, 1:1), (:gi, 1:1)], variables = :synvars)
SNN.monitor!(E, (:v, 1:3))

SNN.monitor!(model.pop, [:fire])


model.pop.E.records[:start_time]
SNN.sim!(model = model; duration = 4second)

# default(palette = :okabe_ito)
## Plot
p1 = plot(
    [
        SNN.vecplot(
            E,
            :v,
            neurons = n,
            r = 3s:2ms:4s,
            add_spikes = true,
            lw = 2,
            xlabel = "",
            c = :black,
        ) for n = 1:3
    ]...,
    layout = (3, 1),
    leftmargin = 10Plots.mm,
    rightmargin = 10Plots.mm,
    frame = :none,
    ylims = (-60, 20),
    size = (800, 400),
)
plot!(
    p1,
    subplot = 3,
    xticks = (3:0.2:4, 3:0.2:4),
    xlabel = "Time (s)",
    xaxis = true,
    frame = :axes,
    grid = false,
    yaxis = false,
    ylabel = "",
)
plot!(
    p1,
    ylabel = "Membrane potential (mV)",
    subplot = 2,
    yaxis = true,
    xaxis = false,
    frame = :axes,
    grid = false,
)
plot!(p1, subplot = 1, topmargin = 5Plots.mm)

p = plot(
    SNN.raster(model.pop, [3.4s, 4s], yrotation = 90),
    SNN.vecplot(
        E,
        [:synvars_ge, :synvars_gi],
        neurons = 1,
        r = 3.8s:4s;
        legend = true,
        xlabel = "Time (s)",
        ylabel = "Conductance (nS)",
        palette = :okabe_ito,
    ),
    p1,
    SNN.firing_rate(model.pop.E, 0:4s, time_average = true)[1] |>
    x->histogram(
        x,
        ylabel = "Neurons",
        xlabel = "Firing rate (Hz)",
        label = "",
        c = :black,
    ),
    size = (900, 600),
    fg_legend = :transparent,
    legendfontsize = 12,
)

p = plot!(p, size = (800, 600))
savefig(p, joinpath(SNN.DOCS_ASSETS_PATH, "AdEx_net.png"))
