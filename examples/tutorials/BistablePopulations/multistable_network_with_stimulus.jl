using Revise
using SpikingNeuralNetworks
using DrWatson
SNN.@load_units;
using SNNUtils
using Plots
using Statistics
##

# Define each of the network recurrent assemblies
function define_network(N, name, istdp)
    # Number of neurons in the network
    N = N
    # Create dendrites for each neuron
    E = SNN.AdEx(N = N, param = SNN.AdExParameter(Vr = -55mV, At = 1mV, b=0, a=0), name="Exc_$name")
    # Define interneurons 
    I = SNN.IF(; N = N ÷ 4, param = SNN.IFParameter(τm = 20ms, El = -50mV), name="Inh_$name")
    # Define synaptic interactions between neurons and interneurons
    E_to_I = SNN.SpikingSynapse(E, I, :he, p = 0.2, μ = 1.0, name="E_to_I_$name")
    E_to_E = SNN.SpikingSynapse(E, E, :he, p = 0.2, μ = 0.5, name="E_to_E_$name")
    I_to_I = SNN.SpikingSynapse(I, I, :hi, p = 0.2, μ = 1.0, name="I_to_I_$name")
    I_to_E = SNN.SpikingSynapse(
        I,
        E,
        :hi,
        p = 0.2,
        μ = 25,
        param = istdp,
        name="I_to_E_$name",
    )
    norm = SNN.SynapseNormalization(E, [E_to_E], param = SNN.AdditiveNorm(τ = 10ms))

    # Store neurons and synapses into a dictionary
    pop = SNN.@symdict E I
    syn = SNN.@symdict I_to_E E_to_I E_to_E norm I_to_I
    noise = SNN.PoissonStimulus(E, :he, param=4.5kHz, neurons=:ALL)
    # Return the network as a tuple
    SNN.monitor([E, I], [:fire])
    SNN.monitor(I_to_E, [:W], sr=10Hz)
    SNN.merge_models(pop, syn, noise=noise, silent=true)
end

n_assemblies = 4
istdp = SNN.iSTDPParameterTime(τy = 20ms, η = 0.5) 
subnets = Dict(Symbol("sub_$n") => define_network(400, n, istdp) for n = 1:n_assemblies)
syns = Dict{Symbol,Any}()
for i in eachindex(subnets)
    for j in eachindex(subnets)
        i == j && continue
        symbol = Symbol(string("$(i)E_to_$(j)I_lateral"))
        synapse = SNN.SpikingSynapse(
                subnets[i].pop.E,
                subnets[j].pop.I,
                :he,
                p = 0.2,
                μ = 2.0,
            )
        push!( syns, symbol => synapse)
    end
end



# Merge the models and run the simulation, the merge_models function will return a model object (syn=..., pop=...); the function has strong type checking, see the documentation.
network = SNN.merge_models(subnets, syns, silent=true)
trig_param1 = PoissonStimulusInterval(fill(8.5kHz, 400), [[8s, 9s]])
trig_param2 = PoissonStimulusInterval(fill(8.5kHz, 400), [[12s, 13s]])
trigger = Dict{Symbol,Any}(
    :first => SNN.PoissonStimulus(network.pop.sub_1_E, :he, param=trig_param1, neurons=:ALL, name="First stim"),
    :second => SNN.PoissonStimulus(network.pop.sub_2_E, :he, param=trig_param2, neurons=:ALL, name="Second stim")
)
network = SNN.merge_models(network, trigger=trigger)

# Define a time object to keep track of the simulation time, the time object will be passed to the train! function, otherwise the simulation will not create one on the fly.
# train!(model = network, duration = 5000ms, pbar = true, dt = 0.125)
SNN.clear_records(network.pop)
SNN.clear_records(network.syn)
train!(model = network, duration = 15000ms, pbar = true, dt = 0.125)

# Plot the raster plot of the network
SNN.raster(network.pop, [0s, 15s], every=5)
##
i_to_e = SNN.filter_items(network.syn, condition=p->occursin("I_to_E", p.name))
w_i = map(eachindex(i_to_e)) do i
    w, r_t = record(i_to_e[i], :W, interpolate=true)
    mean(w, dims=1)[1,:]
end |> collect

_, r_t= record(i_to_e[1], :W, interpolate=true)
p1 = plot(r_t./1000, w_i, xlabel="Time (s)", ylabel="Synaptic weight", legend=:topleft, title="Synaptic weight of I to E synapse", labels=["pop 1" "pop 2" "pop 3" "pop 4"], lw=4)

Epop = SNN.filter_items(network.pop, condition=p->occursin("E", p.name))
rates, interval = SNN.firing_rate(Epop, interval = 0s:20ms:15s, interpolate=false)
rates = mean.(rates)
p2 = plot(interval, rates, xlabel="Time (s)", ylabel="Firing rate (Hz)", legend=:topleft, title="Firing rate of the excitatory population", lw=4)

plot(p1, p2, layout=(2,1), size=(800,800))

##

# define the time interval for the analysis
# select only excitatory populations
# get the spiketimes of the excitatory populations and the indices of each population
exc_populations = SNN.filter_items(network.pop, condition=p->occursin("Exc", p.name))
exc_spiketimes = SNN.spiketimes(network.pop)
# exc_indices = SNN.population_indices(exc_populations)
# calculate the firing rate of each excitatory population
rates, intervals = SNN.firing_rate(exc_populations, interval = interval,  τ = 50, interpolate=false)
rates = mean.(rates)

# Plot the firing rate of each assembly and the correlation matrix
p1 = plot()
for i in eachindex(rates)
    plot!(
        interval,
        rates[i],
        label = "Assembly $i",
        xlabel = "Time (ms)",
        ylabel = "Firing rate (Hz)",
        xlims = (2_000, 15_000),
        legend = :topleft,
    )
end
plot!()


##

cor_mat = zeros(length(rates), length(rates))
for i in eachindex(rates)
    for j in eachindex(rates)
        cor_mat[i, j] = cor(rates[i], rates[j])
    end
end
p2 = heatmap(
    cor_mat,
    c = :bluesreds,
    clims = (-1, 1),
    xlabel = "Assembly",
    ylabel = "Assembly",
    title = "Correlation matrix",
    xticks = 1:3,
    yticks = 1:3,
)
plot(p1, p2, layout = (2, 1), size = (600, 800), margin = 5Plots.mm)
