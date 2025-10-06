##
using ThreadTools
using Plots
using UnPack
using Statistics
using SpikingNeuralNetworks
SNN.@load_units

import SpikingNeuralNetworks: IFSinExpParameter, IF, PoissonLayer, Stimulus, SpikingSynapse, compose, monitor!, sim!, firing_rate, @update, SingleExpSynapse, IFParameter, Population

Zerlaut2019_network = (
    Npop = (E=4000, I=1000),

    exc = IFParameter(
                τm = 200pF / 10nS, 
                El = -70mV, 
                Vt = -50.0mV, 
                Vr = -70.0f0mV,
                R  = 1/10nS, 
                τabs = 2ms,       
                ),

    inh = IFParameter(
                τm = 200pF / 10nS, 
                El = -70mV, 
                Vt = -53.0mV, 
                Vr = -70.0f0mV,
                R  = 1/10nS, 
                τabs = 2ms,       
                ),

    synapse = SingleExpSynapse(
                τi=5ms,
                τe=5ms,
                E_i = -80mV,
                E_e = 0mV
            ),

    connections = (
        E_to_E = (p = 0.05, μ = 2nS, rule=:Fixed),
        E_to_I = (p = 0.05, μ = 2nS, rule=:Fixed),
        I_to_E = (p = 0.05, μ = 10nS, rule=:Fixed),
        I_to_I = (p = 0.05, μ = 10nS, rule=:Fixed),
        ),
    
    afferents = (
        N = 100, 
        rate = 20Hz,
        conn = (p = 0.1f0,μ = 4.0),
        ), 
)


function soma_network(config)
    @unpack afferents, connections, Npop = config
    E = Population(config.exc, config.synapse, N=Npop.E, name="E")
    I = Population(config.inh, config.synapse, N=Npop.I, name="I")

    AfferentParam = PoissonLayer(rate=afferents.rate, N=afferents.N)
    afferentE = Stimulus(AfferentParam, E, :ge, conn=afferents.conn, name="noiseE")
    afferentI = Stimulus(AfferentParam, I, :ge, conn=afferents.conn, name="noiseI")

    synapses = (
        E_to_E = SpikingSynapse(E, E, :ge, conn = connections.E_to_E, name="E_to_E"),
        E_to_I = SpikingSynapse(E, I, :ge, conn = connections.E_to_I, name="E_to_I"),
        I_to_E = SpikingSynapse(I, E, :gi, conn = connections.I_to_E, name="I_to_E"),
        I_to_I = SpikingSynapse(I, I, :gi, conn = connections.I_to_I, name="I_to_I"),
    )
    model = compose(;E,I, afferentE, afferentI, synapses..., silent=true, name="Balanced network") 
    monitor!(model.pop, [:fire])
    monitor!(model.stim, [:fire])
    return compose(;model..., silent=true)
end


#

νa =  exp.(range(log(1), log(40), 20))
f_rate = map(νa) do x
    frs = tmap(1:5) do _
        config = @update Zerlaut2019_network begin
            afferents.rate = x*Hz
        end 
        model = soma_network(config)
        sim!(;model, duration=10_000ms,  pbar=false)
        fr= firing_rate(model.pop.E, interval=3s:10s, pop_average=true, time_average=true)[1]
    end
    f = mean(frs)
    @info "rate: $x Hz = $(mean(f))"
    frs
end

ff_rate = [filter(x -> x < 80, mean.(fr)) for fr in f_rate]
scatter(νa, mean.(ff_rate), ribbon=std.(ff_rate), scale=:log10, xlims=(0.9,20), ylims=(0.0000001,80))#, xscale=:log, yscale=:log)
plot!(νa, mean.(ff_rate), ribbon=std.(ff_rate), scale=:log10, xlims=(0.9, 20), ylims=(0.0000001,80), lw=5, xticks=([1,5,10, 20], [1,5,10,20]))#, xscale=:log, yscale=:log)
plot!(xlabel="Afferent rate (Hz)", ylabel="Firing rate (Hz)",  legend=false, size=(400,400), xlims=(1,40))




##


# ##