using Distributions
SNN.SNNPlots.default(palette = :okabe_ito)

if_neuron =(  
    param = SNN.IFParameter(R = 0.5GΩ, Vt = -50mV, ΔT = 2mV, El = -70mV, τm = 20ms, Vr = -55mV),
    spike = SNN.PostSpike(),
    synapse = SNN.SingleExpSynapse(),
    )


# Create the IF neuron with tonic firing parameters
E = SNN.Population(;N = 1, if_neuron...)
SNN.monitor!(E, [:v, :fire, :w, :I], sr = 2kHz)

# Create a withe noise input current 
current_param = SNN.CurrentNoise(E.N; I_base = 30pA, I_dist = Normal(00pA, 100pA))
current_stim = SNN.Stimulus(current_param, E, :I)
model = SNN.compose(; E = E, I = current_stim)
SNN.clear_records!(model)
SNN.sim!(; model, duration = 2000ms)

p = plot(
    vecplot(
        E,
        :v,
        add_spikes = true,
        ylabel = "Membrane potential (mV)",
        ylims = (-80, 10),
        c = :black,
    ),
    vecplot(E, :I, ylabel = "External current (pA)", lw = 0.4, alpha = 0.9),
    layout = (2, 1),
    xlabel = "Time (s)",
    leftmargin = 10Plots.mm,
)

savefig(
    p,
    joinpath(ASSET_PATH, "noise_current.png"),
)