
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
@snn_kw mutable struct AD{VFT=Vector{Float32},VBT=Vector{Bool}}
    param::ADEXParameter = ADEXParameter(a,
                                        b,
                                        cm,
                                        v_rest,
                                        tau_m,
                                        tau_w,
                                        v_thresh,
                                        delta_T,
                                        v_reset,
                                        spike_height)
    #param::ADEXParameter = ADEXParameter()
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
          v[i] = v_reset
          w[i] += b
        end
        dv  = (((v0-v[i]) +
                delta_T*exp((v[i] - θ)/delta_T))/τ_m +
                (I[i] - w[i])/cm) *dt
        v[i] += dv
        w[i] += dt * (a*(v[i] - v0) - w[i])/τ_w * dt
        fire[i] = v[i] > θ
        if v[i]>θ
            fire[i] = 1
            v[i] = spike_delta
            spike_raster[cnt] = 1

        else
            spike_raster[cnt] = 0
        end
    end
    p.cnt+=1
    
end
