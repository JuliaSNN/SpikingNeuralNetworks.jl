@with_kw struct IZParameter_more
    a::SNNFloat = 0.01
    b::SNNFloat = 0.2
    c::SNNFloat = -65
    d::SNNFloat = 2
    C::SNNFloat = 0.9
    vr::SNNFloat = -65
    k::SNNFloat = 1.6
    vPeak::SNNFloat = 20
    vt::SNNFloat = -50
end

@with_kw mutable struct IZ_more
    param::IZParameter_more = IZParameter_more(a, b, c, d, C, vr, k, vPeak, vt)
    N::SNNInt = 1
    v::Vector{SNNFloat} = fill(param.vr, N)
    u::Vector{SNNFloat} = param.b * v
    fire::Vector{Bool} = zeros(Bool, N)
    I::Vector{SNNFloat} = zeros(N)
    records::Dict = Dict()
end


function integrate!(p::IZ_more, param::IZParameter_more, dt::SNNFloat)
    @unpack N, v, u, fire, I = p
    @unpack a, b, c, d, C, vr, k, vPeak, vt = param

    v[1] = v[1] + 0.25 * (k * (v[1] - vr) * (v[1] - vt) - u[1] + I[1]) / C
    u[1] = u[1] + 0.25*a*(b*(v[1]-vr)-u[1]) # Calculate recovery variable
    fire[1] = v[1] > vPeak

    if v[1]>=vPeak
        v[1]=c
        u[1]=u[1]+d  # reset u, except for FS cells
    end
    v[1] = ifelse(fire[1], c, v[1])
    u[1] = ifelse(fire[1],d, u[1])
end

function integrate_four!(p::IZ_more, param::IZParameter_more, dt::SNNFloat)
    @unpack N, v, u, fire, I = p
    @unpack a, b, c, d, C, vr, k, vPeak, vt = param

    v[1] = v[1] + 0.25 * (k * (v[1] - vr) * (v[1] - vt) - u[1] + I[1]) / C
    u[1] = u[1] + 0.25*a*(b*(v[1]-vr)-u[1]) # Calculate recovery variable
    fire[1] = v[1] > vPeak

    if v[1]>=vPeak- 0.1*u[1]
        v[i] = vPeak - 0.1*u[1]
        v[i+1] = c + 0.04*u[1]; # Reset voltage
        if (u[i]+d)<670
            u[1] = u[1]+d; # Reset recovery variable
        else
            u[1] = 670;
        end
    end
    #v[1] = ifelse(fire[1], c, v[1])
    #u[1] = ifelse(fire[1],d, u[1])
end



function integrate_five!(p::IZ_more, param::IZParameter_more, dt::SNNFloat)
    @unpack N, v, u, fire, I = p
    @unpack a, b, c, d, C, vr, k, vPeak, vt = param


    v[1] = v[1] + tau * (k * (v[1] - vr) * (v[1] - vt) - u[1] + I) / C

    #u[i+1]=u[i]+tau*a*(b*(v[i]-vr)-u[i]); # Calculate recovery variable
    if v[1] < d
        u[1] = u[1] + tau*a*(0-u[1])
    else
        u[1] = u[1] + tau*a*((0.025*(v[1]-d)^3)-u[1])
    end
    if v[1]>=vPeak
        v[1]=vPeak;
        v[1]=c;
    end
    fire[1] = v[1] > vPeak

end
function integrate_one_six!(p::IZ_more, param::IZParameter_more, dt::SNNFloat)
    @unpack N, v, u, fire, I = p
    @unpack a, b, c, d, C, vr, k, vPeak, vt = param

    v[1] = v[1] + tau * (k * (v[1] - vr) * (v[1] - vt) - u[1] + I) / C
    u[1] = u[1]+tau*a*(b*(v[1]-vr)-u[1]);
    if v[1] > -65
        b=0;
    else
        b=15;
    end
    if v[1] > (vPeak + 0.1*u[1])
        v[1] = vPeak + 0.1*u[1];
        v[1] = c-0.1*u[i+1]; # Reset voltage
        u[1]=u[1]+d;
    end
    v[1] = ifelse(fire[1], c, v[1])
    u[1] = ifelse(fire[1],d, u[1])
end
function integrate_one_seven!(p::IZ_more, param::IZParameter_more, dt::SNNFloat)
    @unpack N, v, u, fire, I = p
    @unpack a, b, c, d, C, vr, k, vPeak, vt = param

    v[1] = v[1] + tau * (k * (v[1] - vr) * (v[1] - vt) - u[1] + I) / C


    if v[1] > -65
        b=2;
    else
        b=10;
    end
    u[1]=u[1]+tau*a*(b*(v[1]-vr)-u[1]);
    if v[1]>=vPeak
        v[1]=vPeak;
        v[1]=c;
        u[1]=u[1]+d;  # reset u, except for FS cells
    end
    v[1] = ifelse(fire[1], c, v[1])
    u[1] = ifelse(fire[1],d, u[1])
end

#=
function integrate!(p::IZ_more, param::IZParameter_more, dt::SNNFloat)
    @unpack N, v, u, fire, I = p
    @unpack a, b, c, d, C, vr, k, vt, vPeak = param
    #tau = 0.25f0; #dt
    @inbounds for i = 1:N
        println(i," i is 1?")
        #v[i] += 0.5f0dt * (0.04f0v[i]^2 + 5f0v[i] + 140f0 - u[i] + I[i])
        v[i] += v[i] + 0.25f0 * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I[i]) / C
        u[i] += u[i] + 0.25f0*a*(b*(v[i]-vr)-u[i]) # Calculate recovery variable
        #v[i+1] = v[i] + 0.25f0 * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I[i]) / C
        #u[i+1] = u[i] + 0.25f0*a*(b*(v[i]-vr)-u[i]); # Calculate recovery variable
        println(v[i],u[i],C," ",I[i]," ",i)

        #fire[i] = v[i] > vPeak
        if v[i+1]>=vPeak
            v[i]=vPeak
            v[i+1]=c
            u[i+1]=u[i+1]+d  # reset u, except for FS cells
        end
    end
    @inbounds for i = 1:N
        fire[i] = v[i] > vPeak
        v[i] = ifelse(fire[i], c, v[i])
        u[i] += ifelse(fire[i], d, 0f0)
    end
end
=#
