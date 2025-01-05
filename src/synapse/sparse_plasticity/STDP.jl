# Define the struct to hold synapse parameters for both Exponential and Mexican Hat STDP
# STDP Parameters Structure
@snn_kw struct STDPParameter{FT = Float32} <: SpikingSynapseParameter
    A_post::FT = 10e-2pA / mV    # LTD learning rate (inhibitory synapses)
    A_pre::FT =  10e-2pA / (mV * mV)  # LTP learning rate (inhibitory synapses)
    τpre::FT = 20ms                    # Time constant for pre-synaptic spike trace
    τpost::FT = 20ms                     # Time constant for post-synaptic spike trace
    Wmax::FT = 30.0pF                # Max weight
    Wmin::FT = 0.0pF               # Min weight (negative for inhibition)
end

# STDP Variables Structure
@snn_kw struct STDPVariables{VFT = Vector{Float32}, IT = Int} <: PlasticityVariables
    Npost::IT                      # Number of post-synaptic neurons
    Npre::IT                       # Number of pre-synaptic neurons
    tpre::VFT = zeros(Npre)           # Pre-synaptic spike trace
    tpost::VFT = zeros(Npost)          # Post-synaptic spike trace
end

# Function to initialize plasticity variables
function plasticityvariables(param::T, Npre, Npost) where T <: STDPParameter
    return STDPVariables(Npre = Npre, Npost = Npost)
end

function plasticity!(c::PT, param::STDPParameter, dt::Float32, T::Time) where PT <: AbstractSparseSynapse
    plasticity!(c, param, c.plasticity, dt, T)
end

# Function to implement STDP update rule
function plasticity!(
    c::PT,
    param::STDPParameter,
    plasticity::STDPVariables,
    dt::Float32,
    T::Time
) where PT <: AbstractSparseSynapse
    @unpack rowptr, colptr, I, J, index, W, fireJ, fireI, g, index = c
    @unpack tpre, tpost = plasticity
    @unpack A_pre, A_post, τpre, τpost, Wmax, Wmin = param


    # Update weights based on pre-post spike timing
    @fastmath for j in eachindex(fireJ)
        for st = rowptr[j]:(rowptr[j+1]-1)
            s = index[st]
            if fireJ[J[s]]
                W[s] += tpost[I[s]]  # pre-post
                # @info "pre: $(W[s]) $tpost $(J[s])"
            end
        end
    end

    # Update weights based on pre-post spike timing
    @fastmath for i in 1:length(colptr)-1
        for s = colptr[i]:(colptr[i+1]-1)
            if fireI[I[s]]
                W[s] += tpre[J[s]]  # pre-post
                # @info "post: $(W[s]) $tpre $(I[s])"
            end
        end
    end

    @fastmath for i in eachindex(fireI)
        tpost[i] += dt * (-tpost[i]) / τpost
        if fireI[i]
            tpost[i] += A_post
        end
    end
    @fastmath for j in eachindex(fireJ)
        tpre[j] += dt * (-tpre[j]) / τpre
        if fireJ[j]
            tpre[j] += A_pre
        end
    end
    # Clamp weights to the specified bounds
    @turbo for i in eachindex(W)
        @inbounds W[i] = clamp(W[i], Wmin, Wmax)
    end
end


# Export the relevant functions and structs
export STDPParameter, STDPVariables, plasticityvariables, plasticity!

# @snn_kw struct STDPParameter{FT = Float32} <: SpikingSynapseParameter
#     τpre::FT = 20ms
#     τpost::FT = 20ms
#     Wmax::FT = 0.01
#     ΔApre::FT = 0.01 * Wmax
#     ΔApost::FT = -ΔApre * τpre / τpost * 1.05
# end

# @snn_kw struct STDPVariables{VFT = Vector{Float32},IT = Int} <: PlasticityVariables
#     ## Plasticity variables
#     Npost::IT
#     Npre::IT
#     tpre::VFT = zeros(Npre) # presynaptic spiking time
#     tpost::VFT = zeros(Npost) # postsynaptic spiking time
#     Apre::VFT = zeros(Npre) # presynaptic trace
#     Apost::VFT = zeros(Npost) # postsynaptic trace
# end

# function plasticityvariables(param::T, Npre, Npost) where T <: STDPParameter
#     return STDPVariables(Npre = Npre, Npost = Npost)
# end

# ## It's broken   !!

# function plasticity!(c::PT, param::STDPParameter, dt::Float32, T::Time) where PT <: AbstractSparseSynapse
#     plasticity!(c, param, c.plasticity, dt, T)
# end

# function plasticity!(
#     c::AbstractSparseSynapse,
#     param::STDPParameter,
#     plasticity::STDPVariables,
#     dt::Float32,
#     T::Time,
# )
#     @unpack rowptr, colptr, I, J, index, W, fireI, fireJ, g = c
#     @unpack τpre, τpost, Wmax, ΔApre, ΔApost = plasticity

#     @inbounds for j = 1:(length(colptr)-1)
#         if fireJ[j]
#             for s = colptr[j]:(colptr[j+1]-1)
#                 Apre[s] *= exp32(-(t - tpre[s]) / τpre)
#                 Apost[s] *= exp32(-(t - tpost[s]) / τpost)
#                 Apre[s] += ΔApre
#                 tpre[s] = t
#                 W[s] = clamp(W[s] + Apost[s], 0.0f0, Wmax)
#             end
#         end
#     end
#     @inbounds for i = 1:(length(rowptr)-1)
#         if fireI[i]
#             for st = rowptr[i]:(rowptr[i+1]-1)
#                 s = index[st]
#                 Apre[s] *= exp32(-(t - tpre[s]) / τpre)
#                 Apost[s] *= exp32(-(t - tpost[s]) / τpost)
#                 Apost[s] += ΔApost
#                 tpost[s] = t
#                 W[s] = clamp(W[s] + Apre[s], 0.0f0, Wmax)
#             end
#         end
#     end
# end

# export STDPParameter, STDPVariables, plasticityvariables, plasticity!
