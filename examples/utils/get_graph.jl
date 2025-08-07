using DrWatson
using Revise
using SpikingNeuralNetworks
SNN.@load_units;
using SNNUtils

using GLMakie
using SpikingNeuralNetworks.Graphs
using SpikingNeuralNetworks.MetaGraphs

# Define the network
network = let
    # Number of neurons in the network
    N = 1000
    # Create dendrites for each neuron
    E = SNN.AdEx(N = N, param = SNN.AdExParameter(Vr = -60mV))
    # Define interneurons 
    I = SNN.IF(; N = N ÷ 4, param = SNN.IFParameter(τm = 20ms, El = -50mV))
    # Define synaptic interactions between neurons and interneurons
    E_to_I = SNN.SpikingSynapse(E, I, :ge, p = 0.2, μ = 3.0)
    E_to_E = SNN.SpikingSynapse(E, E, :ge, p = 0.2, μ = 0.5)#, param = SNN.vSTDPParameter())
    I_to_I = SNN.SpikingSynapse(I, I, :gi, p = 0.2, μ = 4.0)
    I_to_E = SNN.SpikingSynapse(
        I,
        E,
        :gi,
        p = 0.2,
        μ = 1,
        param = SNN.iSTDPParameterRate(r = 4Hz),
    )
    norm = SNN.SynapseNormalization(E, [E_to_E], param = SNN.AdditiveNorm(τ = 30ms))

    # Store neurons and synapses into a dictionary
    pop = SNN.@symdict E I
    syn = SNN.@symdict I_to_E E_to_I E_to_E norm I_to_I
    (pop = pop, syn = syn)
end


# Create background for the network simulation
noise = SNN.PoissonStimulus(network.pop[:E], :ge, param = 2.8kHz, neurons = :ALL)
noise2 = SNN.PoissonStimulus(network.pop[:E], :gi, param = 3kHz, neurons = :ALL)
old_model = SNN.compose(network, noise = noise, noise2 = noise2)

using SpikingNeuralNetworks: MetaGraphs
my_graph = SNN.graph(old_model)
##


##
set_theme!(theme_light())
f, ax, p = graphplot(
    my_graph,
    edge_width = [0.1 for i = 1:ne(my_graph)],
    node_size = [30 for i = 1:nv(my_graph)],
    arrow_shift = 0.90,
    nlabels = [get_prop(my_graph, v, :name) for v in vertices(my_graph)],
    nlabels_distance = 20,
)



# f, ax, p = graphplot(my_graph, n_labels=names, nlabels_fontsize=12,node_size = 30, edge_width = .1, arrow_shift=.90)
# hidedecorations!(ax)
# names = [get_prop(my_graph, v, :name) for v in vertices(my_graph)]
deregister_interaction!(ax, :rectanglezoom)
register_interaction!(ax, :nhover, NodeHoverHighlight(p))
register_interaction!(ax, :ehover, EdgeHoverHighlight(p))
register_interaction!(ax, :ndrag, NodeDrag(p))
register_interaction!(ax, :edrag, EdgeDrag(p))
