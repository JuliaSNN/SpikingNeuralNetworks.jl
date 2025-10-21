## Example connectivity
pre = SNN.Identity(N=3)
post = SNN.Identity(N=5)

w  = zeros(Float32, 5, 3)
w[3,1] = 1.0
w[2,2] = 1.0
w[5,3] = 1

conn = SNN.SpikingSynapse(pre, post, :g; conn=w)
SNN.matrix(conn) |> x->heatmap(x, yflip=true)

## Brief recap on sparse synapses.
# in SpikingNeuralNetworks.jl connections are maintained in the SparseMatrixCSC format, which means that they are indicized by column (pre-synaptic neurons).
# For each pre-synaptic j, we have a indices of the non-zero connections in colptr[j]:colptr[j+1]-1 (or postsynaptic_idxs(conn, j))
SNN.postsynaptic_idxs(conn, 1)

# and we can add a connection from pre-synaptic neuron j to post-synaptic neuron i with
SNN.connect!(conn, 1, 2, 0.5)
SNN.matrix(conn) |> x->heatmap(x, yflip=true)

## Now let's modify the connectivity. First we swap post-synaptic cells, so that the detailed out-degree is preserved.

# Let's say we want to swap post-synaptic neurons 1 and 2
conn.I[2] = 1

# this means that the second connection (which was from pre-synaptic neuron 1 to post-synaptic neuron 2) is now from pre-synaptic neuron 1 to post-synaptic neuron 1
SNN.matrix(conn) |> x->heatmap(x, yflip=true)

# the postsynaptic indices of pre-synaptic neuron are correct because they depend on colptr 
SNN.postsynaptic(conn,1)

# but now the presynaptic indices are not correct, because they depend on I and J
SNN.presynaptic(conn,1)

# To fix this, we need to update the sparse matrix structure
SNN.update_sparse_matrix!(conn)
@assert SNN.presynaptic(conn,1) == [1]

## let s instead swap pre-synaptic neurons 1 and 2
conn.J[1] = 3
SNN.matrix(conn) |> x->heatmap(x, yflip=true)

# now the presynaptic indices of post-synaptic neuron are correct because they depend on rowptr
SNN.presynaptic(conn,1)
# but now the postsynaptic indices are not correct, because they depend on colptr
SNN.postsynaptic(conn,3)
# To fix this, we need to update the sparse matrix structure
SNN.update_sparse_matrix!(conn)

## Let's now do synaptic turnover in a network
E_uni = SNN.AdExParameter(; El = -50mV)
E = SNN.Population(E_uni, synapse=SNN.DoubleExpSynapse(); N = 1000, name="Excitatory")

I = SNN.Population(SNN.IFParameter(), synapse = SNN.SingleExpSynapse(); N = 200, name="Inhibitory", spike=SNN.PostSpike() )
EE = SNN.SpikingSynapse(E, E, :he; conn=(μ = 2, p = 0.2, rule=:FixedOut))
EI = SNN.SpikingSynapse(E, I, :ge; conn=(μ = 30, p = 0.02))
IE = SNN.SpikingSynapse(I, E, :hi; conn=(μ = 50, p = 0.02))
II = SNN.SpikingSynapse(I, I, :gi; conn=(μ = 10, p = 0.02))
model = SNN.compose(;  E, I, EE, EI, IE, II)


#

synaptic_turnover!(EE; p_rewire=0.0015)
# for _ in 1:29
#     synaptic_turnover!(EE; p_rewire=0.05)
# end
anim = @animate for _ in 1:50
    synaptic_turnover!(EE; p_rewire=0.015)
    SNN.matrix(EE) |> x->heatmap(x, yflip=true)
end
gif(anim, "tmp.gif", fps=10)
##


EE.colptr = []


SNN.postsynaptic(EE)[800]

tt


tt = rand(Uniform(0,1),length(EE.W))
β = 0.01
findall(tt .< β)


##
# default(palette = :okabe_ito)
## Plot
xxp1 = plot([SNN.vecplot(E, :v, neurons = n, r = 3s:2ms:4s, add_spikes = true, lw = 2, xlabel = "", c=:black) for n in 1:3]..., layout=(3,1), leftmargin=10Plots.mm, rightmargin=10Plots.mm, frame=:none, ylims=(-60,20), size=(800,400))
plot!(p1, subplot=3,  xticks=(3:0.2:4, 3:0.2:4), xlabel = "Time (s)", xaxis=true, frame=:axes, grid=false, yaxis=false, ylabel="")
plot!(p1, ylabel="Membrane potential (mV)", subplot=2, yaxis=true, xaxis=false, frame=:axes, grid=false)
plot!(p1, subplot=1, topmargin=5Plots.mm)

p = plot(
    SNN.raster(model.pop,
        [3.4s, 4s], yrotation=90),
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
    size = (900, 600), fg_legend=:transparent, legendfontsize=12
)

p = plot!(p, size=(800,600))
savefig(p, joinpath(SNN.DOCS_ASSETS_PATH, "AdEx_net.png"))