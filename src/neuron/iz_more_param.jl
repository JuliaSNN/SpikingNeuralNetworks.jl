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
    cellnumber::SNNInt = 1
end

@with_kw mutable struct IZ_more
    param::IZParameter_more = IZParameter_more(a, b, c, d, C, vr, k, vPeak, vt,cellnumber)
    N::SNNInt = 1
    v::Vector{SNNFloat} = fill(param.vr, N)
    u::Vector{SNNFloat} = param.b * v
    fire::Vector{Bool} = zeros(Bool, N)
    I::Vector{SNNFloat} = zeros(N)
    records::Dict = Dict()
end

function integrate!(p::IZ_more, param::IZParameter_more, dt::SNNFloat)
    @unpack a, b, c, d, C, vr, k, vPeak, vt,cellnumber = param
    if cellnumber>=1 && cellnumber<=3
        integrate_one_three!(p, param, SNNFloat(dt))
    elseif cellnumber == 4
        integrate_four!(p,param, SNNFloat(dt))
    elseif cellnumber == 5
        integrate_five!(p, p.param, SNNFloat(dt))
    elseif cellnumber == 6
        integrate_six!(p, p.param, SNNFloat(dt))
    elseif cellnumber == 7
        integrate_seven!(p, p.param, SNNFloat(dt))
    end
end
function integrate_one_three!(p::IZ_more, param::IZParameter_more, dt::SNNFloat)
    @unpack N, v, u, fire, I = p
    @unpack a, b, c, d, C, vr, k, vPeak, vt, cellnumber = param
    @inbounds for i = 1:N
        v[i] = v[i] + 0.25 * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I[i]) / C
        u[i] = u[i] + 0.25*a*(b*(v[i]-vr)-u[i]) # Calculate recovery variable
        fire[i] = v[i] > vPeak

        if v[i]>=vPeak
            v[i]=c
            u[i]=u[i]+d  # reset u, except for FS cells
        end
        v[i] = ifelse(fire[i], c, v[i])
        u[i] = ifelse(fire[i],d, u[i])
    end
end

function integrate_four!(p::IZ_more, param::IZParameter_more, dt::SNNFloat)
    @unpack N, v, u, fire, I = p
    @unpack a, b, c, d, C, vr, k, vPeak, vt, cellnumber = param
    @inbounds for i = 1:N

        v[i] = v[i] + 0.25 * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I[i]) / C
        u[i] = u[i] + 0.25*a*(b*(v[i]-vr)-u[i]) # Calculate recovery variable
        fire[i] = v[i] > vPeak

        if v[i]>=vPeak- 0.1*u[i]
            v[i] = vPeak - 0.1*u[i]
            v[i] = c + 0.04*u[i]; # Reset voltage
            if (u[i]+d)<670
                u[i] = u[i]+d; # Reset recovery variable
            else
                u[i] = 670;
            end
        end
    end
    #v[i] = ifelse(fire[i], c, v[i])
    #u[i] = ifelse(fire[i],d, u[i])
end



function integrate_five!(p::IZ_more, param::IZParameter_more, dt::SNNFloat)
    @unpack N, v, u, fire, I = p
    @unpack a, b, c, d, C, vr, k, vPeak, vt, cellnumber = param
    tau::SNNFloat = 0.25
    @inbounds for i = 1:N
        v[i] = v[i] + 0.25 * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I[i]) / C

        #v[i] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I) / C

        #u[i+1]=u[i]+tau*a*(b*(v[i]-vr)-u[i]); # Calculate recovery variable
        if v[i] < d
            u[i] = u[i] + tau*a*(0-u[i])
        else
            u[i] = u[i] + tau*a*((0.025*(v[i]-d)^3)-u[i])
        end
        if v[i]>=vPeak
            v[i]=vPeak;
            v[i]=c;
        end
        fire[i] = v[i] > vPeak
    end
end
function integrate_six!(p::IZ_more, param::IZParameter_more, dt::SNNFloat)
    @unpack N, v, u, fire, I = p
    @unpack a, b, c, d, C, vr, k, vPeak, vt, cellnumber = param
    tau::SNNFloat = 0.25
    @inbounds for i = 1:N

        v[i] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I[i]) / C
        if v[i] > -65
            b=0;
        else
            b=15;
        end
        u[i] = u[i]+tau*a*(b*(v[i]-vr)-u[i]);

        if v[i] > (vPeak + 0.1*u[i])
            v[i] = vPeak + 0.1*u[i];
            v[i] = c-0.1*u[i]; # Reset voltage
            u[i]=u[i]+d;
        end
        v[i] = ifelse(fire[i], c, v[i])
        u[i] = ifelse(fire[i],d, u[i])
    end
end
function integrate_seven!(p::IZ_more, param::IZParameter_more, dt::SNNFloat)
    @unpack N, v, u, fire, I = p
    @unpack a, b, c, d, C, vr, k, vPeak, vt, cellnumber = param
    tau::SNNFloat = 0.25
    @inbounds for i = 1:N

        v[i] = v[i] + 0.25 * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I[i]) / C


        if v[i] > -65
            b=2;
        else
            b=10;
        end
        u[i]=u[i]+0.25*a*(b*(v[i]-vr)-u[i]);
        if v[i]>=vPeak
            v[i]=vPeak;
            v[i]=c;
            u[i]=u[i]+d;  # reset u, except for FS cells
        end
        v[i] = ifelse(fire[i], c, v[i])
        u[i] = ifelse(fire[i],d, u[i])
    end
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
