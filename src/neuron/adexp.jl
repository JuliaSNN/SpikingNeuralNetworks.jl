@snn_kw struct ADEXParameter{FT=Float32}
    a::FT = 4.0
    b::FT = 0.0805
    cm::FT = 0.281
    v_rest::FT = -70.6
    tau_m::FT = 9.3667
    tau_w::FT = 144.0
    v_thresh::FT = -50.4
    delta_T::FT = 2.0
    v_spike::FT = -40.0
    v_reset::FT = -70.6
    spike_delta::FT = 30
end
@snn_kw mutable struct AD{VFT=Vector{Float32},VBT=Vector{Bool}}
    param::ADEXParameter = ADEXParameter()

    N::Int32 = 1
    cnt::Int32 = 1
    v::VFT = fill(param.v_rest, N)
    w::VFT = zeros(N)
    fire::VBT = zeros(Bool, N)
    I::VFT = zeros(N)
    sized::Int32 = 1
    spike_raster::Vector{Int32} = zeros(N)
    records::Dict = Dict()
end

function integrate!(p::AD, param::ADEXParameter, dt::Float32)
    @unpack N, cnt, v, w, fire, I,spike_raster,sized = p
    @unpack a,b,cm,v_rest,tau_m,tau_w,v_thresh,delta_T,v_spike,v_reset,spike_delta = param
    @inbounds for i = 1:N
        if spike_raster[cnt] == 1 || fire[i]
          v[i] = v_reset
          w[i] += b
        end
        dv  = (((v_rest-v[i]) +
                delta_T*exp((v[i] - v_thresh)/delta_T))/tau_m +
                (I[i] - w[i])/cm) *dt
        v[i] += dv
        w[i] += dt * (a*(v[i] - v_rest) - w[i])/tau_w * dt
        fire[i] = v[i] > v_thresh
        if v[i]>v_thresh
            fire[i] = 1
            v[i] = spike_delta
            spike_raster[cnt] = 1

        else
            spike_raster[cnt] = 0
        end
    end
    cnt+=1
end
