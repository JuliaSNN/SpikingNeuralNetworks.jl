@eval SNN.SNNModels begin
    abstract type AbstractDoubleExpCurrentParameter <: AbstractSynapseParameter end

    """
        DoubleExpCurrentSynapse{FT} <: AbstractDoubleExpCurrentParameter

    A synaptic parameter type that models double exponential current synaptic dynamics.

    # Fields
    - `τre::FT`: Rise time constant for excitatory synapses (default: 1ms)
    - `τde::FT`: Decay time constant for excitatory synapses (default: 6ms)
    - `τri::FT`: Rise time constant for inhibitory synapses (default: 0.5ms)
    - `τdi::FT`: Decay time constant for inhibitory synapses (default: 2ms)

    # Type Parameters
    - `FT`: Floating point type (default: `Float32`)

    This type implements double exponential current synaptic dynamics, where synaptic currents are calculated using separate rise and decay time constants for both excitatory and inhibitory synapses.
    """
    DoubleExpCurrentSynapse

    @snn_kw struct DoubleExpCurrentSynapse{FT = Float32} <: AbstractDoubleExpCurrentParameter
        τre::FT = 1ms # Rise time for excitatory synapses
        τde::FT = 6ms # Decay time for excitatory synapses
        τri::FT = 0.5ms # Rise time for inhibitory synapses
        τdi::FT = 2ms # Decay time for inhibitory synapses
    end

    """
        DoubleExpCurrentSynapseVars{VFT} <: AbstractSynapseVariable
    A synaptic variable type that stores the state variables for double exponential current synaptic dynamics.
    # Fields
    - `N::Int`: Number of synapses
    - `ge::VFT`: Vector of excitatory conductances
    - `gi::VFT`: Vector of inhibitory conductances
    - `he::VFT`: Vector of auxiliary variables for excitatory synapses
    - `hi::VFT`: Vector of auxiliary variables for inhibitory synapses
    """
    DoubleExpCurrentSynapseVars
    @snn_kw struct DoubleExpCurrentSynapseVars{VFT = Vector{Float32}} <: AbstractSynapseVariable
        N::Int = 100
        ge::VFT = zeros(Float32, N)
        gi::VFT = zeros(Float32, N)
        he::VFT = zeros(Float32, N)
        hi::VFT = zeros(Float32, N)
    end

    function synaptic_variables(synapse::DoubleExpCurrentSynapse, N::Int)
        return DoubleExpCurrentSynapseVars(;
            N = N,
            ge = zeros(Float32, N),
            gi = zeros(Float32, N),
            he = zeros(Float32, N),
            hi = zeros(Float32, N),
        )
    end

    function update_synapses!(
        p::P,
        synapse::T,
        receptors::RECT,
        synvars::DoubleExpCurrentSynapseVars,
        dt::Float32,
    ) where {P<:AbstractGeneralizedIF,T<:AbstractDoubleExpCurrentParameter,RECT<:NamedTuple}
        @unpack N, ge, gi, he, hi = synvars
        @unpack τde, τre, τdi, τri = synapse
        @unpack gaba, glu = receptors
        @inbounds @simd for i ∈ 1:N
            he[i] += glu[i]
            hi[i] += gaba[i]
            ge[i] += dt * (-ge[i] / τde + he[i])
            he[i] += dt * (-he[i] / τre)
            gi[i] += dt * (-gi[i] / τdi + hi[i])
            hi[i] += dt * (-hi[i] / τri)
        end
        fill!(glu, 0.0f0)
        fill!(gaba, 0.0f0)
    end


    @inline function synaptic_current!(
        p::T,
        synapse::DoubleExpCurrentSynapse,
        synvars::DoubleExpCurrentSynapseVars,
        v::VT1, # membrane potential
        syncurr::VT2, # synaptic current
    ) where {T<:AbstractPopulation,VT1<:AbstractVector,VT2<:AbstractVector}
        @unpack N = p
        @unpack ge, gi = synvars
        @inbounds @simd for i ∈ 1:N
            syncurr[i] = -(ge[i] - gi[i] )
        end
    end

export DoubleExpCurrentSynapse
end

##
# SNNModels.DoubleExpCurrentSynapse()

E = SNN.Population(SNN.AdExParameter(; El = -70mV), synapse = SNN.DoubleExpCurrentSynapse(); N = 800, name = "Excitatory")
extE = SNN.Stimulus(SNN.PoissonLayer(5.0f0; N = 100, name = "PoissonInput"), E, :glu, conn=(ρ=0.3, μ=1))
extI = SNN.Stimulus(SNN.PoissonLayer(1.0f0; N = 100, name = "PoissonInput"), E, :gaba, conn=(ρ=0.7, μ=1))
model = SNN.compose(;E, extE, extI)

SNN.monitor!(model.pop, [:fire, :v])
SNN.sim!(model, 1s, pbar=true)
SNN.raster(model.pop)
SNN.vecplot(model.pop.E, :v)