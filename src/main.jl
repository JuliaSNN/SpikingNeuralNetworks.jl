#"dt", "simulation_duration", "delay", "stimulus_duration"



function sim!(P, C, dt)
    for p in P
        integrate!(p, p.param, Float32(dt))
        record!(p)
    end
    for c in C
        forward!(c, c.param)
        record!(c)
    end
end
function sim!(P, dt)
    for p in P
        integrate!(p, p.param, Float32(dt))
        record!(p)
    end
end

#=
function sim!(P, C; dt = 0.1ms, duration = 10ms)
    sized = duration/dt
    for p in P
         if hasproperty(p, :spike_raster)
              p.spike_raster::Vector{Int32} = zeros(trunc(Int, sized))
         end
         for t = 0ms:dt:duration
              integrate!(p, p.param, Float32(dt))
              record!(p)
         end
    end
end
=#
function sim!(P; dt = 0.25ms, simulation_duration = 1300ms, delay = 300ms,stimulus_duration=1000ms)
    temp = deepcopy(P[1].I)
    size = simulation_duration/dt
    cnt1 = 0
	if hasproperty(P[1], :spike_raster )
		P[1].spike_raster::Vector{Int32} = zeros(trunc(Int, size))

	end
    for t = 0ms:dt:simulation_duration
        cnt1+=1
        if cnt1 < delay/dt
           P[1].I[1] = 0.0
        end
        if cnt1 > (delay/dt + stimulus_duration/dt)
	       P[1].I[1] = 0.0
        end
        if (delay/dt) < cnt1 < (stimulus_duration/dt)
           P[1].I[1] = maximum(temp[1])
        end
        sim!(P, dt)
    end
end


function train!(P, C, dt, t = 0)
    for p in P
        integrate!(p, p.param, Float32(dt))
        record!(p)
    end
    for c in C
        forward!(c, c.param)
        plasticity!(c, c.param, Float32(dt), Float32(t))
        record!(c)
    end
end

function train!(P, C; dt = 0.1ms, duration = 10ms)
    for t = 0ms:dt:(duration - dt)
        train!(P, C, Float32(dt), Float32(t))
    end
end
