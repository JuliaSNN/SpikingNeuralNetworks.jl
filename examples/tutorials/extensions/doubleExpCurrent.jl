using SpikingNeuralNetworks
using CairoMakie
SNN.@load_units

# A new synapse model for the generalized integrate-and-fire populations: current-based synapses with
# double-exponential kinetics. (SNNModels ships the same model as `DoubleExpCurrentSynapse`; here it is
# rebuilt under another name to show the three methods a synapse model needs.)
@eval SNN.SNNModels begin
    """
        MyDoubleExpCurrentSynapse(; τre = 1ms, τde = 6ms, τri = 0.5ms, τdi = 2ms)

    Current-based synapse with double-exponential kinetics; `ge`, `gi` are currents (pA).
    """
    MyDoubleExpCurrentSynapse

    @snn_kw struct MyDoubleExpCurrentSynapse{FT = Float32} <: AbstractSynapseParameter
        τre::FT = 1ms
        τde::FT = 6ms
        τri::FT = 0.5ms
        τdi::FT = 2ms
    end

    @snn_kw struct MyDoubleExpCurrentVars{VFT = Vector{Float32}} <: AbstractSynapseVariable
        N::Int = 100
        ge::VFT = zeros(eltype(VFT), N)
        gi::VFT = zeros(eltype(VFT), N)
        he::VFT = zeros(eltype(VFT), N)
        hi::VFT = zeros(eltype(VFT), N)
    end

    # 1. state of the synapses of N neurons, in the precision FT of the population
    synaptic_variables(synapse::MyDoubleExpCurrentSynapse, N::Int, ::Type{FT}) where {FT<:AbstractFloat} =
        MyDoubleExpCurrentVars(; N, VFT = Vector{FT})

    # 2. one step: add the input delivered to the receptor buffers, integrate, empty the buffers
    function update_synapses!(p::AbstractGeneralizedIF, synapse::MyDoubleExpCurrentSynapse,
                              receptors::NamedTuple, synvars::MyDoubleExpCurrentVars, dt::AbstractFloat)
        @unpack N, ge, gi, he, hi = synvars
        @unpack τde, τre, τdi, τri = synapse
        @unpack gaba, glu = receptors
        @inbounds for i ∈ 1:N
            he[i] += glu[i]
            hi[i] += gaba[i]
            ge[i] += dt * (-ge[i] / τde + he[i])
            he[i] += dt * (-he[i] / τre)
            gi[i] += dt * (-gi[i] / τdi + hi[i])
            hi[i] += dt * (-hi[i] / τri)
        end
        fill!(glu, zero(eltype(glu)))
        fill!(gaba, zero(eltype(gaba)))
    end

    # 3. the synaptic current seen by the membrane equation
    function synaptic_current!(p::AbstractPopulation, synapse::MyDoubleExpCurrentSynapse,
                               synvars::MyDoubleExpCurrentVars, v::AbstractVector, syncurr::AbstractVector)
        @unpack ge, gi = synvars
        @inbounds for i ∈ eachindex(syncurr)
            syncurr[i] = -(ge[i] - gi[i])
        end
    end
end

E = SNN.Population(SNN.AdExParameter(; El = -70mV); synapse = SNN.SNNModels.MyDoubleExpCurrentSynapse(),
                   N = 800, name = "Excitatory")
extE = SNN.Stimulus(SNN.PoissonLayer(5Hz; N = 100), E, :glu, conn = (ρ = 0.3, μ = 1))
extI = SNN.Stimulus(SNN.PoissonLayer(1Hz; N = 100), E, :gaba, conn = (ρ = 0.7, μ = 1))
model = SNN.compose(; E, extE, extI, silent = true)

SNN.monitor!(model.pop, [:fire, :v])
SNN.sim!(model, 1s)
fig = Figure(size = (600, 500))
SNN.raster!(Axis(fig[1, 1]), model.pop)
SNN.vecplot!(Axis(fig[2, 1], ylabel = "V (mV)"), model.pop.E, :v, neurons = 1, add_spikes = true)
save(joinpath(mktempdir(), "doubleExpCurrent.png"), fig)
