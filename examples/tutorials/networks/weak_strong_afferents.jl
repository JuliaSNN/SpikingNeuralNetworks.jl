using DrWatson
using Plots
using UnPack
using SpikingNeuralNetworks
using Statistics
SNN.@load_units;


Zerlaut2019_network = (
    Npop = (E=4000, I=1000),
    exc = SNN.IFSinExpParameter(
                τm = 200pF / 10nS, 
                El = -70mV, 
                Vt = -50.0mV, 
                Vr = -70.0f0mV,
                R  = 1/10nS, 
                τabs = 2ms,       
                τi=5ms,
                τe=5ms,
                E_i = -80mV,
                E_e = 0mV,
                ),

    inh = SNN.IFSinExpParameter(
                τm = 200pF / 10nS, 
                El = -70mV, 
                Vt = -53.0mV, 
                Vr = -70.0f0mV,
                R  = 1/10nS, 
                τabs = 2ms,       
                τi=5ms,
                τe=5ms,
                E_i = -80mV,
                E_e = 0mV,
                ),

    connections = (
        E_to_E = (p = 0.05, μ = 2nS),
        E_to_I = (p = 0.05, μ = 2nS),
        I_to_E = (p = 0.05, μ = 10nS),
        I_to_I = (p = 0.05, μ = 10nS),
        ),
    
    afferents = (
        N = 100,
        p = 0.1f0,
        rate = 20Hz,
        μ = 4.0,
        ), 
)

function network(config)
    @unpack afferents, connections, Npop = config
    E = SNN.IF(N=Npop.E, param=config.exc, name="E")
    I = SNN.IF(N=Npop.I, param=config.inh, name="I")

    AfferentParam = SNN.PoissonLayerParameter(afferents.rate; afferents...)
    afferentE = SNN.PoissonLayer(E, :ge, param=AfferentParam, name="noiseE")
    afferentI = SNN.PoissonLayer(I, :ge, param=AfferentParam, name="noiseI")

    synapses = (
        E_to_E = SNN.SpikingSynapse(E, E, :ge, p=connections.E_to_E.p, μ=connections.E_to_E.μ, name="E_to_E", dist=:fixed),
        E_to_I = SNN.SpikingSynapse(E, I, :ge, p=connections.E_to_I.p, μ=connections.E_to_I.μ, name="E_to_I", dist=:fixed),
        I_to_E = SNN.SpikingSynapse(I, E, :gi, p=connections.I_to_E.p, μ=connections.I_to_E.μ, name="I_to_E", dist=:fixed),
        I_to_I = SNN.SpikingSynapse(I, I, :gi, p=connections.I_to_I.p, μ=connections.I_to_I.μ, name="I_to_I", dist=:fixed),
    )
    model = SNN.compose(;E,I, afferentE, afferentI, synapses..., silent=true, name="Balanced network") 
    SNN.monitor!(model.pop, [:fire])
    SNN.monitor!(model.stim, [:fire])
    # monitor!(model.pop, [:v], sr=200Hz)
    return SNN.compose(;model..., silent=true)
end

config = SNN.@update Zerlaut2019_network begin
    afferents.rate = 10Hz
end 
model = network(config)


# ##
# plots = map() do input_rate
#     model = network(config)
#     sim!(;model, duration=10_000ms,  pbar=true)
#     pr= raster(model.pop, every=40)

#     # Firing rate of the network with a fixed afferent rate
#     frE, r = firing_rate(model.pop.E, interval=3s:10s, pop_average=true)
#     frI, r = firing_rate(model.pop.I, interval=3s:10s, pop_average=true)
#     pf = plot(r, [frE, frI], labels=["E" "I"],
#         xlabel="Time (s)", ylabel="Firing rate (Hz)", 
#         title="Afferent rate: $input_rate Hz",
#         size=(600, 400), lw=2)

#     # Plot the raster plot of the network
#     plot(pf, pr, layout=(2, 1))
# end

# plot(plots..., layout=(1,2), size=(1200, 600), xlabel="Time (s)", leftmargin=10Plots.mm)
# ##


νs =  exp.(range(log(0.1),log(30), 15))
plots = map(νs) do input_rate
    config = SNN.@update Zerlaut2019_network begin
        afferents.rate = input_rate* Hz
    end 
    model = network(config)
    SNN.sim!(;model, duration=10_000ms,  pbar=true)

    # Firing rate of the network with a fixed afferent rate
    frE, r = SNN.firing_rate(model.pop.E, interval=3s:10s, pop_average=true)
    @show mean(frE)
end

plot(νs, plots, xscale=:log10,  
    xlabel="Afferent rate (Hz)", ylabel="Firing rate (Hz)", 
    labels=["E" "I"], lw=2, size=(600, 400))

##
νs2 =  exp.(range(log(30),log(150), 15))
plots2 = map(νs2) do input_rate
    config = SNN.@update Zerlaut2019_network begin
        afferents.rate = input_rate* Hz
    end 
    model = network(config)
    SNN.sim!(;model, duration=10_000ms,  pbar=true)

    # Firing rate of the network with a fixed afferent rate
    frE, r = SNN.firing_rate(model.pop.E, interval=3s:10s, pop_average=true)
    @show mean(frE)
end

ν = vcat(νs, νs2)
fr = vcat(plots, plots2)
plot(ν, fr, xscale=:log10,  
    xlabel="Afferent rate (Hz)", ylabel="Firing rate (Hz)", 
    labels=["E" "I"], lw=2, size=(600, 400))