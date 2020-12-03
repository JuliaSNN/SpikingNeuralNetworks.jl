@with_kw struct ADEXParameter
    a::SNNFloat = 4.0
    b::SNNFloat = 0.0805
    cm::SNNFloat = 0.281
    v_rest::SNNFloat = -70.6
    tau_m::SNNFloat = 9.3667
    tau_w::SNNFloat = 144.0
    v_thresh::SNNFloat = -50.4
    delta_T::SNNFloat = 2.0
    v_spike::SNNFloat = -40.0
    v_reset::SNNFloat = -70.6
    spike_delta::SNNFloat = 30
end

@with_kw mutable struct AD
    param::ADEXParameter = ADEXParameter(a,
                                        b,
                                        cm,
                                        v_rest,
                                        tau_m,
                                        tau_w,
                                        v_thresh,
                                        delta_T,
                                        v_spike,
                                        v_reset,
                                        spike_delta)
    N::SNNInt = 1
    cnt::SNNInt = 2
    v::Vector{SNNFloat} = fill(param.v_rest, N)
    w::Vector{SNNFloat} = zeros(N)
    fire::Vector{Bool} = zeros(Bool, N)
    I::Vector{SNNFloat} = zeros(N)
    spike_raster::Vector{SNNInt} = zeros(N)
    records::Dict = Dict()
end

function integrate!(p::AD, param::ADEXParameter, dt::SNNFloat)
    @unpack N, cnt, v, w, fire, I,spike_raster = p
    @unpack a,b,cm,v_rest,tau_m,tau_w,v_thresh,delta_T,v_spike,v_reset,spike_delta = param
    if spike_raster[cnt-1] == 1 || fire[1]
      v[1] = v_reset
      w[1] += b
    end
    dv  = (((v_rest-v[1]) +
            delta_T*exp((v[1] - v_thresh)/delta_T))/tau_m +
            (I[1] - w[1])/cm) *dt
    v[1] += dv
    w[1] += dt * (a*(v[1] - v_rest) - w[1])/tau_w * dt


    fire[1] = v[1] > v_thresh

    if v[1]>v_thresh
        fire[1] = 1 # v[1] > vPeak
        v[1] = spike_delta
        spike_raster[cnt] = 1

    else
        spike_raster[cnt] = 0
    end

    cnt+=1
end
