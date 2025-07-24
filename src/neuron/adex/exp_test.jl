using Plots


mg = 1.0f0
b = 3.36   # voltage dependence of nmda channels
k = -0.077    # Eyal 2018

function nmda_(x, func, k=k)
    1/( 1.0f0 + (mg / b) * func(Float32(k * (x))))
end

function exp32(x::R) where {R<:Real}
    x = ifelse(x < -10.0f0, -32.0f0, x)
    x = 1.0f0 + x / 32.0f0
    x *= x
    x *= x
    x *= x
    x *= x
    x *= x
    return x
end

function exp64(x::R) where {R<:Real}
    x = ifelse(x < -10.0f0, -64.0f0, x)
    x = 1.0f0 + x / 64.0f0
    x *= x
    x *= x
    x *= x
    x *= x
    x *= x
    x *= x
    return x
end

function exp256(x::R) where {R<:Real}
    x = ifelse(x < -10.0f0, -256.0f0, x)
    x = 1.0f0 + x / 256.0f0
    x *= x
    x *= x
    x *= x
    x *= x
    x *= x
    x *= x
    x *= x
    x *= x
    return x
end



xr = -70:0.1:0
plot(xr, nmda_.(xr, exp256), label= "exp256")
plot!(xr,nmda_.(xr, exp64), label= "exp64")
plot!(xr,nmda_.(xr, exp32), label= "exp32")
plot!(xr,nmda_.(xr,  exp), label= "exp")
##

xr = -70:0.1:0
plot(xr, nmda_.(xr, exp256, k), label= "exp256")
plot!(xr,nmda_.(xr,  exp256, -0.074), label= "exp256")
plot!(xr,nmda_.(xr, exp32), label= "exp32")
##

plot.(exp256)
plot!.(exp32)