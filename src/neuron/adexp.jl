

#SpikingNeuralNetworks.ADEXParameter(; a=6.050246708405076, b=7.308480222357973,
# cm=803.1019662706587, v_rest=-63.22881649139353, tau_m=19.73777028610565,
#tau_w=351.0551915202058, v_thresh=-39.232165554444265, delta_T=6.37124632135508, v_reset=-59.18792270568965, spike_delta=16.33506432689027)

@snn_kw struct ADEXParameter{FT=Float32}
    a::FT = 4.0
    b::FT = 0.0805
    cm::FT = 0.281
    v0::FT = -70.6
    τ_m::FT = 9.3667
    τ_w::FT = 144.0
    θ::FT = -50.4
    delta_T::FT = 2.0
    v_reset::FT = -70.6
    spike_delta::FT = 30
end

#(; a=6.050246708405076, b=7.308480222357973, cm=803.1019662706587,
# v_rest=-63.22881649139353, tau_m=19.73777028610565,
#  tau_w=351.0551915202058, v_thresh=-39.232165554444265, delta_T=6.37124632135508,
#   v_reset=-59.18792270568965, spike_delta=16.33506432689027)

@snn_kw mutable struct AD{VFT=Vector{Float32},VBT=Vector{Bool}}
    param::ADEXParameter = ADEXParameter()
    N::Int32 = 1
    cnt::Int32 = 1
    v::VFT = fill(param.v0, N)
    w::VFT = zeros(N)
    fire::VBT = zeros(Bool, N)
    I::VFT = zeros(N)
    sized::Int32 = 1
    spike_raster::Vector{Int32} = zeros(N)
    records::Dict = Dict()
end

function integrate!(p::AD, param::ADEXParameter, dt::Float32)
    @unpack N, cnt, v, w, fire, I,spike_raster,sized = p
    @unpack a,b,cm,v0,τ_m,τ_w,θ,delta_T,v_reset,spike_delta = param
    @inbounds for i = 1:N
        if spike_raster[cnt] == 1 || fire[i]
          v[i] = v_reset*0.01
          w[i] += b
        end
        dv  = (((v0-v[i]) +
                delta_T*exp((v[i] - θ)/delta_T))/τ_m +
                (I[i] - w[i])/cm) *dt
        v[i] += dv*0.01
        w[i] += dt * (a*(v[i] - v0) - w[i])/τ_w * dt
        fire[i] = v[i] > θ
        if v[i]>θ*0.01
            fire[i] = 1
            v[i] = spike_delta
            spike_raster[cnt] = 1

        else
            spike_raster[cnt] = 0
        end
    end
    p.cnt+=1
    
end
"""
Julia SNN Implementation of AdExp Neuron.
[Adaptive_exponential_integrate and fire neuron](http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model)
Dr. Wulfram Gerstner
Romain Brette, Ecole Normale Supérieure, Paris, France
"""
AD
