# AdEx/IF network whose four connection classes have heterogeneous Tsodyks-Markram short-term
# plasticity, sampled from experimental fits of mouse cortex (SNNUtils `sample_stp_campagnola`).
using SpikingNeuralNetworks, SNNUtils
using CairoMakie, Distributions, Statistics, Random
SNN.@load_units
Random.seed!(1234)

# per-presynaptic-neuron STP parameters of one connection class
function stp_het(N, class)
    p = SNNUtils.sample_stp_campagnola(N, class; weights = false)
    return SNN.MarkramSTPParameterHet(τD = Float32.(p.τD), τF = Float32.(p.τF), U = Float32.(p.U))
end

E_het = SNN.make_heterogeneous(SNN.AdExParameter(; El = -50mV), 1600;
                               τm = Normal(10.0f0, 2.0f0), b = Normal(60.0f0, 4.0f0))
E = SNN.Population(E_het, synapse = SNN.DoubleExpSynapse(); N = 1600, name = "Excitatory")
I = SNN.Population(SNN.IFParameter(), synapse = SNN.SingleExpSynapse(); N = 400,
                   name = "Inhibitory", spike = SNN.PostSpike())
EE = SNN.SpikingSynapse(E, E, :he; conn = (μ = 1, p = 0.02), STPParam = stp_het(E.N, :exc_exc))
EI = SNN.SpikingSynapse(E, I, :ge; conn = (μ = 30, p = 0.02), STPParam = stp_het(E.N, :exc_inh))
IE = SNN.SpikingSynapse(I, E, :hi; conn = (μ = 50, p = 0.02), STPParam = stp_het(I.N, :inh_exc))
II = SNN.SpikingSynapse(I, I, :gi; conn = (μ = 10, p = 0.02), STPParam = stp_het(I.N, :inh_inh))
model = SNN.compose(; E, I, EE, EI, IE, II)

SNN.monitor!(model.pop, [:fire])
SNN.monitor!(EE, [:ρ], sr = 20Hz)                      # efficacy of every EE synapse
SNN.train!(model = model; duration = 3second)          # STP runs only under train!

## Plot: efficacy of the first synapse of 100 presynaptic neurons, and a raster
ρ, r = SNN.record(EE, :ρ, range = true)
first_syn = EE.colptr[1:100]
fig = Figure(size = (700, 600))
ax1 = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Efficacy ρ")
series!(ax1, r ./ 1000, ρ(first_syn, r), color = :viridis)
ax2 = Axis(fig[2, 1], ylabel = "Neuron")
SNN.raster!(ax2, model.pop, 2s:3s)
save(joinpath(mktempdir(), "STP_het.png"), fig)
fig
