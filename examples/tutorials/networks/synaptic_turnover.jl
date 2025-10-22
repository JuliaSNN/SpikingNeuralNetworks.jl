## AdEx neuron with fixed external current connections with multiple receptors
E_uni = SNN.AdExParameter(; El = -50mV)
E_het = SNN.heterogeneous(E_uni, 800; τm = Normal(10f0, 2f0), b= Normal(60f0,4f0))
E = SNN.Population(E_het, synapse=SNN.DoubleExpSynapse(); N = 800, name="Excitatory")

I = SNN.Population(SNN.IFParameter(), synapse = SNN.SingleExpSynapse(); N = 200, name="Inhibitory", spike=SNN.PostSpike() )
EE = SNN.SpikingSynapse(E, E, :he; conn=(μ = 2, p = 0.02))
EI = SNN.SpikingSynapse(E, I, :ge; conn=(μ = 30, p = 0.02))
IE = SNN.SpikingSynapse(I, E, :hi; conn=(μ = 50, p = 0.02))
II = SNN.SpikingSynapse(I, I, :gi; conn=(μ = 10, p = 0.02))
model = SNN.compose(;  E, I, EE, EI, IE, II)

SNN.monitor!(E, [(:ge,1:1), (:gi,1:1), ], variables=:synvars)
SNN.monitor!(E, (:v, 1:3))

SNN.monitor!(model.pop, [:fire])
SNN.sim!(model = model; duration = 4second)

## Synaptic turnover

pre_tt = randn(size(EE.fireJ))
post_tt = randn(size(EE.fireI))

old = Int[]
new = Int[]
@unpack colptr, I, J, index, W, fireJ = EE
for j in eachindex(fireJ)
    for s = colptr[j]:(colptr[j+1]-1)
        @show I[s], j
        if pre_tt[J[index[s]]] + post_tt[I[s]] .< -2.5
            I[s] = rand(1:model.pop.E.N)
            # push!(old, s)
            # push!(new, rand(1:model.pop.E.N))
        end
    end
end



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