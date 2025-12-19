
## AdEx neuron with fixed external current connections with multiple receptors
neuron_param = SNN.IFParameter(; 
                    τm = 20ms,
                    R  = 100MΩ,
                    Vt = -50mV,
                    Vr = -60mV,
                    El = -60mV,)

spike = SNN.PostSpike(τabs=5ms, up=0ms, At = 0ms, AP_membrane=0mV,)
synapse = SNN.SingleExpSynapse(τe = 5ms, τi = 10ms, E_e = 0mV, E_i = -80mV)

E = SNN.Population(;param = neuron_param, synapse, spike, N = 8000, name = "Excitatory")
I = SNN.Population(;param = neuron_param, synapse, spike, N = 2000, name = "Inhibitory")
E.I .= 150pA
I.I .= 150pA
EE = SNN.SpikingSynapse(E, E, :glu; conn = (μ = 60*0.27/10, p = 0.02, rule=:Bernoulli), delay_dist=Normal(0.8ms, 0))
EI = SNN.SpikingSynapse(E, I, :glu; conn = (μ = 60*0.27/10, p = 0.02, rule=:Bernoulli), delay_dist=Normal(0.8ms, 0))
II = SNN.SpikingSynapse(I, I, :gaba; conn = (μ = 20*4.5/10, p = 0.02, rule=:Bernoulli), delay_dist=Normal(0.8ms, 0))
IE = SNN.SpikingSynapse(I, E, :gaba; conn = (μ = 20*4.5/10, p = 0.02, rule=:Bernoulli), delay_dist=Normal(0.8ms, 0))
model = SNN.compose(; E, I, EE, EI, IE, II)

E.v .= neuron_param.Vr .+ rand(Float32, size(E.v)) .* (neuron_param.Vt - neuron_param.Vr)
SNN.monitor!(model.pop, [:fire], )
# SNN.monitor!(E, [:v])
# SNN.monitor!(I, [:ge, :gi], variables = :synvars)
start = time()
@profview SNN.sim!(model = model; duration = 1second, dt = 0.125, pbar = true)
stop = time()
println("Simulation time: $(stop - start) seconds")
E.I .= 0pA
I.I .= 0pA
SNN.sim!(model = model; duration = 5second, dt = 0.125, pbar = true)
##
fr, r =SNN.firing_rate(model.pop, 0:20ms:10s, pop_average = true) 
plot(r,
    fr[1],
    xlabel = "Time (s)",
    ylabel = "Firing rate (Hz)",
    title = "Firing rate of Excitatory neurons",
    label = "Excitatory",
    c = :blue,
)
##
plot(
    histogram(
        SNN.ISI_CV2(model.pop.E, interval = 0:60s),
        xlabel = "CV2",
        ylabel = "Number of neurons",
        title = "ISI CV2 distribution of Excitatory neurons",
        label = "",
        c = :black,
    ),
    begin 
        fr = SNN.firing_rate(model.pop, 1ms:10ms:SNN.get_time(model), time_average=true)
        histogram(fr[1][1])
    end
    ,
    layout = (2,1),
    size=(800,800),
)
# SNN.raster(model.pop, 0:1s)

##
neurons = SNN.spiketimes(model.pop.E)[1:250] 
SNN.raster(neurons, 20s:20.4s; order=collect(1:200), figsize = (900, 300), title = "Raster plot of Excitatory neurons")

##

SNN.vecplot(
    E,
    :v,
    neurons = 1:1,
    r = 59s:60s,
    add_spikes = true,
    lw = 2,
    xlabel = "Time (s)",
    ylabel = "Membrane potential (mV)",
    title = "Membrane potential of Excitatory neurons",
    # size = (800, 400),
)

##
SNN.vecplot(
    I,
    [:ge, :gi],
    variables = :synvars,
    neurons = 1,
    r = 4s:5s;
    legend = true,
    xlabel = "Time (s)",
    ylabel = "Conductance (nS)",
    title = "Synaptic conductances of Inhibitory neuron",
    # size = (800, 400),
)



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
    ylims = :auto,
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
        [:ge, :gi],
        variables = :synvars,
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

p