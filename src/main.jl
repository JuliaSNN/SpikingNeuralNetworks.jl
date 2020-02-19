function sim!(P, C, dt)
    integrate!(P[1], P[1].param, SNNFloat(dt))
    record!(P[1])
end

function sim!(P, C; dt = 0.1ms, duration = 10ms)
    @show("the top function is called a lot by this one")
    @show(P)
    @show(P[1].param)
    @show(P[1].I)

    cnt = 0
    for t = 0ms:dt:duration
       cnt+=1
    end
    @show(cnt)
    size = duration/dt
    @show(size)

    cnt1 = 0
    for t = 0ms:dt:duration
        cnt1+=1
        if cnt1> 3*size/4 # if cnt1 > delay
           P[1].I[1] = 0.0 
        end 
        if cnt1< 1*size/4 # if cnt1 < duration
           P[1].I[1] = 0.0 
        end 
  
       sim!(P, C, dt)
    end
end

function train!(P, C, dt, t = 0)
    for p in P
        integrate!(p, p.param, SNNFloat(dt))
        record!(p)
    end
    for c in C
        forward!(c, c.param)
        plasticity!(c, c.param, SNNFloat(dt), SNNFloat(t))
        record!(c)
    end
end

function train!(P, C; dt = 0.1ms, duration = 10ms)
    for t = 0ms:dt:duration
        train!(P, C, SNNFloat(dt), SNNFloat(t))
    end
end
