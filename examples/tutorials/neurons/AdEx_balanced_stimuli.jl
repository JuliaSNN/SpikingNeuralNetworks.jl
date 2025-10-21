neuron_parameter =(
    param = SNN.AdExParameter(
        R=0.5GΩ,
        Vt = -50mV,
        ΔT = 2mV,
        El = -70mV,
        τm = 20ms,
        Vr = -55mV,),
    synapse = SNN.CurrentSynapse(2ms, 0.2f0, -75mV, 0mV),
    spike = SNN.PostSpike(τabs= 5ms)
)

# Create the IF neuron
E = SNN.Population(;neuron_parameter..., N = 1)

# Create an excitatory and inhibitory spike trains

# Define the Poisson stimulus parameters 
poisson_exc = SNN.PoissonLayer(
    rate=1Hz,    # Mean firing rate (Hz) 
    N = 1000, # Neurons in the Poisson Layer
)
poisson_inh = SNN.PoissonLayer(
    rate = 10Hz,       # Mean firing rate (Hz)
    N = 1000,  # Neurons in the Poisson Layer
)
conn = (
    p = 1,
    μ = 5nS
)

# Create the Poisson layers for excitatory and inhibitory inputs
stim_exc = SNN.Stimulus(poisson_exc, E, :glu, name = "Exc Noise"; conn)
stim_inh = SNN.Stimulus(poisson_inh, E, :gaba, name = "Inh Noise"; conn)

# Create the model and run the simulation
model = SNN.compose(; E = E, stim_exc, stim_inh)
SNN.monitor!(E, [:v, :fire, :w, :glu, :gaba], sr = 2kHz)
SNN.monitor!(E, [:ge, :gi], sr = 2kHz, variables= :synvars)
SNN.monitor!(model.stim, [:fire])
SNN.sim!(; model, duration = 1000ms)

# Plot the results
# gplot is a special function the plots the synaptic currents

plot(SNN.record(model.pop.E, :glu)(1,0s:1s))


# model.pop.E.synvars_ge

Plots.default(palette = :okabe_ito)
p = plot(
    SNN.raster(model.stim),
    SNN.gplot(
        E,
        v_sym = :v,
        ge_sym = :synvars_ge,
        gi_sym = :synvars_gi,
        Ee_rev = 0mV,
        Ei_rev = -75mV,
        r = 0ms:2.5ms:1000ms,
        ylabel = "Synaptic current (μA)",
    ),
    SNN.vecplot(
        E,
        :v,
        add_spikes = true,
        ylabel = "Membrane potential (mV)",
        ylims = (-180, 10),
        c = :black,
    ),
    layout = (3, 1),
    fgcolorlegend = :transparent,
    size = (800, 900),
    xlabel = "Time (s)",
    leftmargin = 10Plots.mm,
)
##

savefig(p, joinpath(SNN.DOCS_ASSETS_PATH, "balanced_stimuli.png"))

p
