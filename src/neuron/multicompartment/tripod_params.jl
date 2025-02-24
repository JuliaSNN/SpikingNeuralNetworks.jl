
"""
	AdExSoma

An implementation of the Adaptive Exponential Integrate-and-Fire (AdEx) model, adapted for a Tripod neuron.

# Fields
- `C::FT = 281pF`: Membrane capacitance.
- `gl::FT = 40nS`: Leak conductance.
- `R::FT = nS / gl * GΩ`: Total membrane resistance.
- `τm::FT = C / gl`: Membrane time constant.
- `Er::FT = -70.6mV`: Resting potential.
- `Vr::FT = -55.6mV`: Reset potential.
- `Vt::FT = -50.4mV`: Rheobase threshold.
- `ΔT::FT = 2mV`: Slope factor.
- `τw::FT = 144ms`: Adaptation current time constant.
- `a::FT = 4nS`: Subthreshold adaptation conductance.
- `b::FT = 80.5pA`: Spike-triggered adaptation increment.
- `AP_membrane::FT = 10.0f0mV`: After-potential membrane parameter .
- `BAP::FT = 1.0f0mV`: Backpropagating action potential parameter.
- `up::IT = 1ms`, `τabs::IT = 2ms`: Parameters related to spikes.

The types `FT` and `IT` represent Float32 and Int64 respectively.
"""
AdExSoma

@snn_kw struct AdExSoma{FT = Float32,IT = Int64} <: AbstractAdExParameter
    #Membrane parameters
    C::FT = 281pF           # (pF) membrane timescale
    gl::FT = 40nS                # (nS) gl is the leaking conductance,opposite of Rm
    R::FT = nS / gl * GΩ               # (GΩ) total membrane resistance
    τm::FT = C / gl                # (ms) C / gl
    Er::FT = -70.6mV          # (mV) resting potential
    # AdEx model
    Vr::FT = -55.6mV     # (mV) Reset potential of membrane
    Vt::FT = -50.4mV          # (mv) Rheobase threshold
    ΔT::FT = 2mV            # (mV) Threshold sharpness
    # Adaptation parameters
    τw::FT = 144ms          #ms adaptation current relaxing time
    a::FT = 4nS            #nS adaptation current to membrane
    b::FT = 80.5pA         #pA adaptation current increase due to spike
    # After spike timescales and membrane
    AP_membrane::FT = 10.0f0mV
    BAP::FT = 1.0f0mV
    up::FT = 1ms
    τabs::FT = 2ms
end

## Synapses Tripod neuron

MilesGabaSoma =
    GABAergic(Receptor(E_rev = -70.0, τr = 0.1, τd = 15.0, g0 = 0.38), Receptor())
DuarteGluSoma = Glutamatergic(
    Receptor(E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.73),
    ReceptorVoltage(E_rev = 0.0, nmda = 0.0f0),
)
EyalGluDend = Glutamatergic(
    Receptor(E_rev = 0.0, τr = 0.26, τd = 2.0, g0 = 0.73),
    ReceptorVoltage(E_rev = 0.0, τr = 8, τd = 35.0, g0 = 1.31, nmda = 1.0f0),
)
MilesGabaDend = GABAergic(
    Receptor(E_rev = -70.0, τr = 4.8, τd = 29.0, g0 = 0.27),
    Receptor(E_rev = -90.0, τr = 30, τd = 400.0, g0 = 0.006), # τd = 100.0
)

TripodSomaSynapse = Synapse(DuarteGluSoma, MilesGabaSoma)
TripodDendSynapse = Synapse(EyalGluDend, MilesGabaDend)

export AdExSoma, TripodSomaSynapse, TripodDendSynapse
