struct TripletRule
    A⁺₂::Float32
    A⁺₃::Float32
    A⁻₂::Float32
    A⁻₃::Float32
    τˣ::Float32
    τʸ::Float32
    τ⁺::Float32
    τ⁻::Float32
    # τˣ⁻::Float32
    # τʸ⁻::Float32
    # τ⁺⁻::Float32
    # τ⁻⁻::Float32
end

# Gutig 2003
struct NLTAH
    τ::Float32
    λ::Float32
    μ::Float32
end


function vogels_istdp()
    ## Inhibition
    tauy = 20.0 #decay of inhibitory rate trace (ms)
    eta = 1.0   #istdp learning rate    (pF⋅ms) eta*rate = weights
    r0 = 0.005   #target rate (khz)
    alpha = 2 * r0 * tauy #rate trace threshold for istdp sign (kHz) (so the 2 has a unit)
    jeimin = 48.7 #minimum ei strength (pF)
    jeimax = 243 #maximum ei strength   (pF)

    return ISTDP(tauy, eta, r0, alpha, jeimin, jeimax)
end

# Clopath 2010
@with_kw struct STDP
    #voltage based stdp
    a⁻::Float32 = 0.0f0    #ltd strength (pF/mV) # a*(V-θ) = weight
    a⁺::Float32 = 0.0f0    #ltp strength (pF/mV)
    θ⁻::Float32 = -90.0f0 #ltd voltage threshold (mV)
    θ⁺::Float32 = 0.0f0 #ltp voltage threshold (mV)
    τs::Float32 = 20 # homeostatic scaling timescale
    τu::Float32 = 1.0f0  #timescale for u variable   (1/ms)
    τv::Float32 = 1.0f0  #timescale for v variable   (1/ms)
    τx::Float32 = 1.0f0  #timescale for x variable   (1/ms)
    τ1::Float32 = 1.0f0
    ϵ::Float32 = 1.0f0  # filter for delayed membrane potential.
    j⁻::Float32 = 0.0f0 # minimum weight
    j⁺::Float32 = 100.0f0 # maximum weight
    τu⁻::Float32 = 1 / τu  #timescale for u variable   (1/ms)
    τv⁻::Float32 = 1 / τv  #timescale for v variable   (1/ms)
    τx⁻::Float32 = 1 / τx  #timescale for x variable   (1/ms)
    τ1⁻::Float32 = 1 / τ1

end





#Vogel 2011
#inhibitory stdp
@with_kw struct ISTDP
    ## sISP
    η::Float32 = 0.2
    r0::Float32 = 0.01
    vd::Float32 = -70
    τd::Float64 = 5 #decay of dendritic potential (ms)
    τy::Float32 = 20 #decay of inhibitory rate trace (ms)
    α::Float32 = 2 * r0 * τy
    j⁻::Float32 = 2.78f0  # minimum weight
    j⁺::Float32 = 243.0f0 # maximum weight
    # ## vISP
    # ηv::Float32=10e-3 ## learning rate
    # θv::Float32=-65 ## threshold for voltage
    # αv::Float32=2*10e-4 ## depression parameter
    # τv::Float32=5 ## decay of inhibitory rate trace (ms)
    # τs::Float32=200ms ## decay of inhibitory rate trace (ms)
end
vISP = ISTDP
sISP = ISTDP

function lkd_stdp()
    return STDP(
        a⁻ = 8.0f-4pF / mV,  #ltd strength
        a⁺ = 14.0f-4pF / mV, #ltp strength
        θ⁻ = -70.0f0mV,  #ltd voltage threshold
        θ⁺ = -49.0f0mV,  #ltp voltage threshold
        τu = 10.0f0ms,  #timescale for u variable
        τv = 7.0f0ms,  #timescale for v variable
        τx = 15.0f0ms,  #timescale for x variable
        τ1 = 5ms,    # filter for delayed voltage
        j⁻ = 1.7f8pF,  #minimum ee strength
        j⁺ = 21.0f4pF,   #maximum ee strength
    )
end

function clopath_vstdp_visualcortex()
    return STDP(
        a⁻ = 14.0f-3pF / mV,  #ltd strength
        a⁺ = 8.0f-3pF / mV, #ltp strength
        θ⁻ = -70.6mV,  #ltd voltage threshold
        θ⁺ = -25.3mV,  #ltp voltage threshold
        τu = 10.0ms,  #timescale for u variable
        τv = 7.0ms,  #timescale for v variable
        τx = 15.0ms,  #timescale for x variable
        ϵ = 1ms,    # filter for delayed voltage
        j⁻ = 1.78pF,  #minimum ee strength
        j⁺ = 21.4pF,   #maximum ee strength
    )

end

function bono_vstdp()
    return STDP(
        a⁻ = 4.0f-4pF / mV,  #ltd strength
        a⁺ = 14.0f-4pF / mV, #ltp strength
        θ⁻ = -59.0mV,  #ltd voltage threshold
        θ⁺ = -20.0mV,  #ltp voltage threshold
        τu = 15.0ms,  #timescale for u variable
        τv = 45.0ms,  #timescale for v variable
        τx = 20.0ms,  #timescale for x variable
        τ1 = 5ms,    # filter for delayed voltage
        j⁻ = 1.78pF,  #minimum ee strength
        j⁺ = 21.4pF,   #maximum ee strength
    )
end



function pfister_visualcortex(alltoall::Bool = true, full::Bool = true)
    if alltoall
        if full
            return TripletRule(5e-10, 6.2e-3, 7e-3, 2.3e-4, 101.0, 125.0, 16.8, 33.7)
        else
            return TripletRule(0.0, 6.5e-3, 7.1e-3, 0.0, -1.0, 125.0, 16.8, 33.7)
        end
    else
        if full
            return TripletRule(8.8e-11, 5.3e-2, 6.6e-3, 3.1e-3, 714.0, 40.0, 16.8, 33.7)
        else
            return TripletRule(0.0, 5.2e-2, 8.e-3, 0.0, -1.0, 40.0, 16.8, 33.7)
        end
    end
end


lkd_stdp = STDP(
    a⁻ = 8.0f-5,  #ltd strength
    a⁺ = 14.0f-5, #ltp strength
    θ⁻ = -70.0f0,  #ltd voltage threshold
    θ⁺ = -49.0f0,  #ltp voltage threshold
    τu = 10.0f0,  #timescale for u variable
    τv = 7.0f0,  #timescale for v variable
    τx = 15.0f0,  #timescale for x variable
    τ1 = 5,    # filter for delayed voltage
    j⁻ = 1.78f0,  #minimum ee strength
    j⁺ = 21.0f0,   #maximum ee strength
)

duarte_stdp = STDP(
    a⁻ = 8.0f-5,  #ltd strength
    a⁺ = 14.0f-5, #ltp strength
    θ⁻ = -70.0f0,  #ltd voltage threshold
    θ⁺ = -49.0f0,  #ltp voltage threshold
    τu = 10.0f0,  #timescale for u variable
    τv = 7.0f0,  #timescale for v variable
    τx = 15.0f0,  #timescale for x variable
    τ1 = 5,    # filter for delayed voltage
    j⁻ = 0.05f0,  #minimum ee strength
    j⁺ = 10.0f0,   #maximum ee strength
)