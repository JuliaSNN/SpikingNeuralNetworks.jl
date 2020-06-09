function vars(X)
    map(p->p=>getfield(X,p), filter(x->x != :param, fieldnames(typeof(X))))
end
condition(X, u) = false
affect!(X, u) = nothing
function all_condition(u,t,integrator)
    u = u.x
    p = integrator.p[2]
    trigger = false
    for i in 1:length(u.x)
        for j in 1:length(u.x[i].x)
            _u = u.x[i].x[j]
            _p = p[i][j]
            trigger |= condition(_p, _u)
        end
    end
    return trigger
end
function all_affect!(integrator)
    u = integrator.u.x
    p = integrator.p[2]
    for i in 1:length(u.x)
        for j in 1:length(u.x[i].x)
            _u = u.x[i].x[j]
            _p = p[i][j]
            affect!(_p, _u)
        end
    end
end

function prepare(N, S; tspan, train=true)
    uN = ArrayPartition(map(x->ArrayPartition(last.(vars(x))...), N)...)
    uS = ArrayPartition(map(x->ArrayPartition(last.(vars(x))...), S)...)
    u = Remapper(ArrayPartition(uN, uS))
    _u = similar(u)

    p = (N, S)

    return ODEProblem(_sim!, u, tspan, (train, p))
end
function restore!(u, p)
    u = u.x # unwrap Remapper
    uN, uS = u.x
    train, p = p
    pN, pS = p
    for idx in 1:length(uN.x)
        _pN = pN[idx]
        v = first.(vars(_pN))
        for vidx in 1:length(v)
            setfield!(_pN, v[vidx], uN.x[idx].x[vidx])
        end
    end
    for idx in 1:length(uS.x)
        _pS = pS[idx]
        v = first.(vars(_pS))
        for vidx in 1:length(v)
            setfield!(_pS, v[vidx], uS.x[idx].x[vidx])
        end
    end
end
function _sim!(du, u, p, t)
    du, u = du.x, u.x # unwrap Remapper
    dN, dS = du.x
    N, S = u.x
    train, p = p
    pN, pS = p
    for idx in 1:length(N.x)
        integrate!(dN.x[idx], N.x[idx], pN[idx], t)
    end
    for idx in 1:length(S.x)
        dS.x[idx] .= 0 # Hack to prevent NaN when train == false
        forward!(dS.x[idx], S.x[idx], pS[idx], t)
        train && plasticity!(dS.x[idx], S.x[idx], pS[idx], t)
    end
end
function sim!(N, S; tspan, alg=Tsit5(), reltol=1e-3, abstol=1e-3, kwargs...)
    prob = prepare(N, S; tspan=tspan, kwargs...)
    cb = DiscreteCallback(all_condition, all_affect!)
    sol = solve(prob, alg; reltol=reltol, abstol=abstol, callback=cb)
    restore!(last(sol.u), prob.p)
    (prob, sol)
end

## old API

# FIXME: dt and duration
sim!(N, S, dt; kwargs...) = sim!(N, S; tspan=(0ms, dt), kwargs...)
train!(N, S, dt, t=0ms; kwargs...) =
    sim!(N, S; tspan=(t, dt), train=true, kwargs...)
train!(P, C; dt=0.1ms, duration=10ms) =
    sim!(N, S; tspan=(0ms, duration), train=true, kwargs...)
