@doc """
Gerstner, W., Kempter, R., van Hemmen, J. L., & Wagner, H. (1996). A neuronal learning rule for sub-millisecond temporal coding. Nature, 383(6595), 76–78. https://doi.org/10.1038/383076a0
"""
STDPGerstner

@snn_kw struct STDPGerstner{FT = Float32} <: STDPParameter
    A_post::FT = 10e-2pA / mV         # LTD learning rate (inhibitory synapses)
    A_pre::FT = 10e-2pA / (mV * mV)  # LTP learning rate (inhibitory synapses)
    τpre::FT = 20ms                   # Time constant for pre-synaptic spike trace
    τpost::FT = 20ms                  # Time constant for post-synaptic spike trace
    Wmax::FT = 30.0pF                 # Max weight
    Wmin::FT = 0.0pF                  # Min weight (negative for inhibition)
end

@doc """
    STDPMexicanHat{FT = Float32}
    
    The STDP is defined such that integral of the kernel is zero. The STDP kernel is defined as:

    `` A x * exp(-x/sqrt(2)) ``

    where   ``A`` is the learning rate for post and pre-synaptic spikes, respectively, and ``x`` is the difference between the post and pre-synaptic traces.
"""
STDPMexicanHat

@snn_kw struct STDPMexicanHat{FT = Float32} <: STDPParameter
    A::FT = 10e-2pA / mV    # LTD learning rate (inhibitory synapses)
    τ::FT = 20ms                    # Time constant for pre-synaptic spike trace
    Wmax::FT = 30.0pF                # Max weight
    Wmin::FT = 0.0pF               # Min weight (negative for inhibition)
end

# STDP Variables Structure
@snn_kw struct STDPVariables{VFT = Vector{Float32},IT = Int} <: LTPVariables
    Npost::IT                      # Number of post-synaptic neurons
    Npre::IT                       # Number of pre-synaptic neurons
    tpre::VFT = zeros(Npre)           # Pre-synaptic spike trace
    tpost::VFT = zeros(Npost)          # Post-synaptic spike trace
    active::Vector{Bool} = [true]
end

# Function to initialize plasticity variables
function plasticityvariables(param::T, Npre, Npost) where {T<:STDPParameter}
    return STDPVariables(Npre = Npre, Npost = Npost)
end


# Function to implement STDP update rule
function plasticity!(
    c::PT,
    param::STDPGerstner,
    plasticity::STDPVariables,
    dt::Float32,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack rowptr, colptr, I, J, index, W, fireJ, fireI, g, index = c
    @unpack tpre, tpost = plasticity
    @unpack A_pre, A_post, τpre, τpost, Wmax, Wmin = param


    # Update weights based on pre-post spike timing
    @inbounds @fastmath begin
        for i = 1:(length(rowptr)-1) # loop over post-synaptic neurons
            @simd for st = rowptr[i]:(rowptr[i+1]-1)
                s = index[st]
                if fireJ[J[s]]
                    W[s] += tpost[i]  # pre-post
                end
            end
        end

        # Update weights based on pre-post spike timing
        for j = 1:(length(colptr)-1) # loop over pre-synaptic neurons
            @simd for s = colptr[j]:(colptr[j+1]-1)
                if fireI[I[s]]
                    W[s] += tpre[j]  # pre-post
                end
            end
        end
        @turbo for i in eachindex(fireI)
            tpost[i] += dt * (-tpost[i]) / τpost
        end
        @simd for i in findall(fireI)
            tpost[i] += A_post
        end

        @turbo for j in eachindex(fireJ)
            tpre[j] += dt * (-tpre[j]) / τpre
        end
        @simd for j in findall(fireJ)
            tpre[j] += A_pre
        end

    end
    # Clamp weights to the specified bounds
    @turbo for i in eachindex(W)
        @inbounds W[i] = clamp(W[i], Wmin, Wmax)
    end
end


MexicanHat(x::Float32) = (1 - x) * exp(-x / sqrt(2)) |> x -> isnan(x) ? 0 : x
function plasticity!(
    c::PT,
    param::STDPMexicanHat,
    plasticity::STDPVariables,
    dt::Float32,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack rowptr, colptr, I, J, index, W, fireJ, fireI, g, index = c
    @unpack tpre, tpost = plasticity
    @unpack A, τ, Wmax, Wmin = param


    # Update weights based on pre-post spike timing
    @inbounds @fastmath begin

        @turbo for i in eachindex(fireI)
            tpost[i] += dt * (-tpost[i]) / τ
        end
        @simd for i in findall(fireI)
            tpost[i] += 1
        end

        @turbo for j in eachindex(fireJ)
            tpre[j] += dt * (-tpre[j]) / τ
        end
        @simd for j in findall(fireJ)
            tpre[j] += 1
        end


        for i = 1:(length(rowptr)-1)
            @simd for st = rowptr[i]:(rowptr[i+1]-1)
                s = index[st]
                if fireJ[J[s]] && abs(tpost[i] * tpre[J[s]]) > 0.0f0
                    W[s] += A * MexicanHat((log(tpre[J[s]] / tpost[i]))^2)
                end
            end
        end

        # Update weights based on pre-post spike timing
        for j = 1:(length(colptr)-1)
            @simd for s = colptr[j]:(colptr[j+1]-1)
                if fireI[I[s]] && abs(tpost[I[s]] * tpre[j]) > 0.0f0
                    W[s] += A * MexicanHat(log(tpre[j] / tpost[I[s]])^2)
                end
            end
        end
    end
    # Clamp weights to the specified bounds
    @turbo for i in eachindex(W)
        @inbounds W[i] = clamp(W[i], Wmin, Wmax)
    end
end

# Export the relevant functions and structs
export STDPVariables, plasticityvariables, plasticity!, STDPMexicanHat, STDPGerstner
