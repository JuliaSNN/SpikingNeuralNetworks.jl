
Zerlaut2019_network = (
    # Number of neurons in each population
    Npop = (E=800, I=200),

    spatial = (
        type = :gaussian,
        σs = (;E =(200um,200um),
                I =(100um,100um) ), 
        ϵ =.4f0,
        grid_size = Float32.([1mm, 1mm]),  # Size of the grid
    ),

    turnover = (
        rate = 20Hz,
        fraction = 0.1f0,
        τpre = 1s,
        τpost = 1s,
        μ = 2.0f0
    ),

    # Parameters for excitatory neurons
    exc = SNN.IFParameter(
                τm = 200pF / 10nS,  # Membrane time constant
                El = -70mV,         # Leak reversal potential
                Vt = -50.0mV,       # Spike threshold
                Vr = -70.0f0mV,     # Reset potential
                R  = 1/10nS,        # Membrane resistance
                # a = 4nS,
                # b = 80pA
                ),

    # Parameters for inhibitory neurons
    inh = SNN.IFParameter(
                τm = 200pF / 10nS,  # Membrane time constant
                El = -70mV,         # Leak reversal potential
                Vt = -53.0mV,       # Spike threshold
                Vr = -70.0f0mV,     # Reset potential
                R  = 1/10nS,        # Membrane resistance
                ),

    spike_exc = SNN.PostSpike(τabs = 2ms),         # Absolute refractory period
    spike_inh = SNN.PostSpike(τabs = 1ms),         # Absolute refractory period 

    # Synaptic properties
    synapse_exc = SNN.SingleExpSynapse(
                τi=5ms,             # Inhibitory synaptic time constant
                τe=5ms,             # Excitatory synaptic time constant
                E_i = -80mV,        # Inhibitory reversal potential
                E_e = 0mV           # Excitatory reversal potential
            ),

    synapse_inh = SNN.SingleExpSynapse(
                τi=5ms,             # Inhibitory synaptic time constant
                τe=5ms,             # Excitatory synaptic time constant
                E_i = -80mV,        # Inhibitory reversal potential
                E_e = 0mV           # Excitatory reversal potential
            ),


    # Connection probabilities and synaptic weights
    connections = (
        E_to_E = (p = 0.05, μ = 2nS,  rule=:Fixed),  # Excitatory to excitatory
        E_to_I = (p = 0.05, μ = 2nS,  rule=:Fixed),  # Excitatory to inhibitory
        I_to_E = (p = 0.05, μ = 10nS, rule=:Fixed), # Inhibitory to excitatory
        I_to_I = (p = 0.05, μ = 10nS, rule=:Fixed), # Inhibitory to inhibitory
        ),

    # Parameters for external Poisson input
    afferents = (
        layer = SNN.PoissonLayer(rate=10Hz, N=100), # Poisson input layer
        conn = (p = 0.1f0, μ = 4.0nS), # Connection probability and weight
        ),
)

# %% [markdown]
# ## Network Construction

#
# Define a function to create the network based on the configuration parameters.

# %%
# Function to create the network
function network(config)
    @unpack afferents, connections, Npop, spike_exc, spike_inh, exc, inh = config
    @unpack synapse_exc, synapse_inh = config

    # Create neuron populations
    E = SNN.Population(exc; synapse=synapse_exc, spike=spike_exc, N=Npop.E, name="E")  # Excitatory population
    I = SNN.Population(inh; synapse=synapse_inh, spike=spike_inh, N=Npop.I, name="I")  # Inhibitory population

    # Create external Poisson input
    @unpack layer = afferents
    afferentE = SNN.Stimulus(layer, E, :glu, conn=afferents.conn, name="noiseE")  # Excitatory input
    afferentI = SNN.Stimulus(layer, I, :glu, conn=afferents.conn, name="noiseI")  # Inhibitory input

    
    SynTurn = SNN.ActivityDependentTurnover(;config.turnover...)
    points =(SNN.place_populations(config.Npop, config.spatial.grid_size))
    L, W, P = SNN.compute_connections(:E, :E, points; spatial=config.spatial, conn=config.connections.E_to_E)
    # Create recurrent connections
    synapses = (
        E_to_E = SNN.SpikingSynapse(E, E, :glu, conn = W, name="E_to_E"),
        E_to_I = SNN.SpikingSynapse(E, I, :glu, conn = connections.E_to_I, name="E_to_I"),
        I_to_E = SNN.SpikingSynapse(I, E, :gaba, conn = connections.I_to_E, name="I_to_E"),
        I_to_I = SNN.SpikingSynapse(I, I, :gaba, conn = connections.I_to_I, name="I_to_I"),
    )
    Exc_turnover = SNN.MetaPlasticity(SynTurn, synapses.E_to_E; p=P, name="EE_Turnover")

    # Compose the model
    model = SNN.compose(; E,I, afferentE, afferentI, synapses..., Exc_turnover, name="Balanced network")

    # Set up monitoring
    SNN.monitor!(model.pop, [:fire])  # Monitor spikes
    SNN.monitor!(model.stim, [:fire])  # Monitor input spikes

    return model, points
end

model, points = network(Zerlaut2019_network)


SNN.monitor!(model.pop, [:fire])
SNN.monitor!(model.syn, [:W], sr=10Hz)
SNN.monitor!(model.syn, [:p_rewire], sr=200Hz)
SNN.monitor!(model.syn, [:p_values], sr=10Hz)
SNN.monitor!(model.syn, [:pre], sr=200Hz)
SNN.monitor!(model.syn, [:post], sr=200Hz)


SNN.train!(model, duration = 100second, pbar=true)
# SNN.raster(model.pop, [10s, 19s], yrotation = 90)

ms=1
fr, r, labels = SNN.firing_rate(model.pop, 0:100ms:20s, pop_average=true)
SNN.firing_rate(model.pop.E, 0:20ms:20s, time_average=true)[1] |> x->histogram(x, ylabel="Neurons", xlabel="Firing rate (Hz)", label="", c=:black, bins=0.:0.1:100)

ms = SNN.firing_rate(model.pop.E, 0:20ms:20s, time_average=true)[1]

exc_x, exc_y = hcat(points.E...)[1,:], hcat(points.E...)[2,:]
scatter(exc_x, exc_y, marker_z=ms, clims=(0, 3), c=:viridis, xlabel="x (mm)", ylabel="y (mm)", title="Average firing rate (Hz)", colorbar_title="Hz")
##

p1= plot(r, fr, xlabel="Time (s)", ylabel="Average firing rate (Hz)", lw=3)
p2 =SNN.raster(model.pop, [98s, 99s], yrotation = 90)

fr, r = SNN.firing_rate(model.pop.E, 0:20ms:20s, τ=50ms)
spatial_avg, xs, ys = SNN.spatial_activity(points.E, fr(1:800,r), N=100, L=0.3mm, )

heatmap(xs, ys, spatial_avg[:,:,1]; c=:viridis, xlabel="x (mm)", ylabel="y (mm)", title="Spatial firing rate (Hz)")

plot(spatial_avg[:])


th, r = SNN.record(model.syn.Exc_turnover, :p_rewire, range=true)
plot(r,th(1, r), xlabel="Time (s)", ylabel="p_rewire")
##

mean(CC, dims=1)[1,:] |> plot

CC = SNN.record(model.syn.Exc_turnover, :p_values, interpolate=false)
bins = range(extrema(CC)..., 100)[2:end-1]
anim = @animate for i in axes(CC, 2)
    histogram(CC[:,i]; bins,  xlabel="Correlation", ylabel="Counts", label="", c=:blue,
    title="Time: $(i/10)s", xlim=(minimum(bins), maximum(bins)), ylim=(0, 200))
end
gif(anim, "turnover.gif", fps = 10)

zz = SNN.matrix(model.syn.E_to_E, CC, 0s:100ms:9s)
zz[:,:,end] .=0
exc_p = hcat(points.E...)[1,:] |> sortperm
anim = @animate for i in axes(zz,3)
    # SNN.plasticity!(turnover, tt)
    heatmap(zz[exc_p,exc_p,i], c = :greys, clim=(0, 1))
end
gif(anim, "turnover.gif", fps = 3)


##
W, r = SNN.record(model.syn.E_to_E, :W, range=true)
zz = SNN.matrix(model.syn.E_to_E, W, 0s:100ms:9s)
zz[:,:,end] .=0
exc_p = hcat(points.E...)[1,:] |> sortperm
anim = @animate for i in axes(zz,3)
    # SNN.plasticity!(turnover, tt)
    heatmap(zz[exc_p,exc_p,i], c = :greys, clim=(0, 1))
end
gif(anim, "turnover.gif", fps = 3)
##

p = plot!(p, size = (800, 600))
savefig(p, joinpath(SNN.DOCS_ASSETS_PATH, "AdEx_net.png"))
