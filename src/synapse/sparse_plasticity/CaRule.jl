# Define the struct to hold synapse parameters for both Exponential and Mexican Hat STDP
# STDP Parameters Structure
using Documenter
abstract type CaAbstractParameter <: SpikingSynapseParameter end

# τdρ  dt 1⁄4 − ρð1 − ρÞðρ⋆ − ρÞ þ γpð1 − ρÞΘ cðtÞ − θp  − γdρΘ1⁄2cðtÞ − θd þ NoiseðtÞ:
##################################################################################################
# STDP Parameters Structure

@snn_kw struct CaPlasticityParameter{FT = Float32} <: STDPAbstractParameter
    A_post::FT = 10e-2pA / mV         # LTD learning rate (inhibitory synapses)
    A_pre::FT = 10e-2pA / (mV * mV)  # LTP learning rate (inhibitory synapses)
    τpre::FT = 20ms                   # Time constant for pre-synaptic spike trace
    τpost::FT = 20ms                  # Time constant for post-synaptic spike trace
    Wmax::FT = 30.0pF                 # Max weight
    Wmin::FT = 0.0pF                  # Min weight (negative for inhibition)
end

# STDP Variables Structure
@snn_kw struct STDPVariables{VFT = Vector{Float32},IT = Int} <: PlasticityVariables
    Npost::IT                      # Number of post-synaptic neurons
    Npre::IT                       # Number of pre-synaptic neurons
    tpre::VFT = zeros(Npre)           # Pre-synaptic spike trace
    tpost::VFT = zeros(Npost)          # Post-synaptic spike trace
end

# Function to initialize plasticity variables
function plasticityvariables(param::T, Npre, Npost) where {T<:STDPAbstractParameter}
    return STDPVariables(Npre = Npre, Npost = Npost)
end

function plasticity!(
    c::PT,
    param::mySTDP,
    dt::Float32,
    T::Time,
) where {PT<:AbstractSparseSynapse,mySTDP<:STDPAbstractParameter}
    plasticity!(c, param, c.plasticity, dt, T)
end

# Function to implement STDP update rule
function plasticity!(
    c::PT,
    param::STDPParameter,
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
export STDPParameter, STDPVariables, plasticityvariables, plasticity!, STDPMexicanHat
