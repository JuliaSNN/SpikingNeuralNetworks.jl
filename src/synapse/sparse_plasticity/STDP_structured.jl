# Define the struct to hold synapse parameters for both Exponential and Mexican Hat STDP
# STDP Parameters Structure
abstract type STDPStructuredParameter <: STDPParameter end

@doc """
    STDPSymmetric{FT = Float32}

    Symmetric STDP rules described in:
    `Structured stabilization in recurrent neural circuits through inhibitory synaptic plasticity` 
    by Festa, D., Cusseddu, C, and Gjorgjieva, J. (2024).
    
    The STDP is defined such that integral of the kernel is zero. The STDP kernel is defined as:

    `` (\frac{A_{post}}{1/\tau_{post}} * exp(-t/\tau_{post} - \frac{A_{pre}}{\tau_pre} * exp(-t/\tau_{pre}) ``

    where ``A_{post}`` and ``A_{pre}`` are the learning rates for post and pre-synaptic spikes, respectively, and ``\tau_{post}`` and ``\tau_{pre}`` are the time constants for post and pre-synaptic traces, respectively.
"""
STDPSymmetric
@snn_kw struct STDPSymmetric{FT = Float32} <: STDPStructuredParameter
    A_x::FT = 3e-2    # LTP learning rate (inhibitory synapses)
    A_y::FT = 3e-2    # LTD learning rate (inhibitory synapses)
    τ_x::FT = 50ms       # Time constant for pre-synaptic spike trace
    τ_y::FT = 500ms      # Time constant for post-synaptic spike trace
    αpre::FT = 0.0pF
    αpost::FT = 0.0pF
    Wmax::FT = 30.0pF   # Max weight
    Wmin::FT = 0.0pF    # Min weight (negative for inhibition)
end

@snn_kw struct STDPAntiSymmetric{FT = Float32} <: STDPStructuredParameter
    A_y::FT = 3e-2     # LTD learning rate (inhibitory synapses)
    A_x::FT = 3e-2    # LTP learning rate (inhibitory synapses)
    τ_x::FT = 50ms       # Time constant for pre-synaptic spike trace
    τ_y::FT = 50ms      # Time constant for post-synaptic spike trace
    αpre::FT = 0.0pF
    αpost::FT = 0.0pF
    Wmax::FT = 30.0pF   # Max weight
    Wmin::FT = 0.0pF    # Min weight (negative for inhibition)
end


@snn_kw struct STDPStructuredVariables{VFT = Vector{Float32},IT = Int} <:
               PlasticityVariables
    Npost::IT                      # Number of post-synaptic neurons
    Npre::IT                       # Number of pre-synaptic neurons
    to_x::VFT = zeros(Npost)        # Pre-synaptic spike trace
    to_y::VFT = zeros(Npost)        # Pre-synaptic spike trace
    tr_x::VFT = zeros(Npre)         # Post-synaptic spike trace
    tr_y::VFT = zeros(Npre)         # Post-synaptic spike trace
end

function plasticityvariables(
    param::T,
    Npre,
    Npost,
) where {T<:STDPStructuredParameter}
    return STDPStructuredVariables(Npre = Npre, Npost = Npost)
end


# SymmetricSTDP and AntiSymmetricSTDP
function plasticity!(
    c::PT,
    param::STDPAntiSymmetric,
    plasticity::STDPStructuredVariables,
    dt::Float32,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack rowptr, colptr, I, J, index, W, fireJ, fireI, g, index = c
    @unpack tr_x, to_y = plasticity
    @unpack A_x, A_y, τ_x, τ_y, Wmax, Wmin, αpre, αpost = param
    # Update weights based on pre-post spike timing
    @inbounds @fastmath begin

        for j = 1:(length(colptr)-1) # loop over post-synaptic neurons
            if fireJ[j]
                @turbo for s = colptr[j]:(colptr[j+1]-1)
                    i = I[s]
                    # @info "Pre synaptic firing, time: $(get_time(T)) trace: $(to_y[i])"
                    W[s] += αpre - A_y / τ_y * to_y[i]  # pre spike
                end
            end
        end

        for i = 1:(length(rowptr)-1) # loop over post-synaptic neurons
            if fireI[i]
                @turbo for st = rowptr[i]:(rowptr[i+1]-1)
                    j = J[index[st]]
                    s = index[st]
                    # @info "Post synaptic firing, time: $(get_time(T)) trace: $(tr_x[j])"
                    W[s] += αpost + A_x / τ_x * tr_x[j]  # post spike
                end
            end
        end

        # Update traces based on pre synpatic firing
        @turbo for i in eachindex(fireI)
            to_y[i] += dt * (-to_y[i]) / τ_y
        end
        @simd for i in findall(fireI)
            to_y[i] += 1
        end

        # Update traces based on pre synpatic firing
        @turbo for j in eachindex(fireJ)
            tr_x[j] += dt * (-tr_x[j]) / τ_x
        end
        @simd for j in findall(fireJ)
            tr_x[j] += 1
        end

    end
    # Clamp weights to the specified bounds
    @turbo for i in eachindex(W)
        @inbounds W[i] = clamp(W[i], Wmin, Wmax)
    end
end

function plasticity!(
    c::PT,
    param::STDPSymmetric,
    plasticity::STDPStructuredVariables,
    dt::Float32,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack rowptr, colptr, I, J, index, W, fireJ, fireI, g, index = c
    @unpack to_x, tr_x, to_y, tr_y = plasticity
    @unpack A_x, A_y, τ_x, τ_y, Wmax, Wmin, αpre, αpost = param


    # Update weights based on pre-post spike timing
    @inbounds @fastmath begin
        # for i in 1:length(rowptr)-1 # loop over post-synaptic neurons
        for j = 1:(length(colptr)-1) # loop over post-synaptic neurons
            if fireJ[j]
                @turbo for s = colptr[j]:(colptr[j+1]-1)
                    i = I[s]
                    W[s] += αpre + (A_x / 2τ_x * to_x[i] - A_y / 2τ_y * to_y[i])  # pre spike
                end
            end
        end

        # Update weights based on pre-post spike timing
        # for j in 1:length(colptr)-1 # loop over pre-synaptic neurons
        for i = 1:(length(rowptr)-1) # loop over post-synaptic neurons
            if fireI[i]
                @turbo for st = rowptr[i]:(rowptr[i+1]-1)
                    j = J[index[st]]
                    s = index[st]
                    W[s] += αpost + (A_x / 2τ_x * tr_x[j] - A_y / 2τ_y * tr_y[j])  # post spike
                end
            end
        end
        @turbo for i in eachindex(fireI)
            to_x[i] += dt * (-to_x[i]) / τ_x
            to_y[i] += dt * (-to_y[i]) / τ_y
        end
        @simd for i in findall(fireI)
            to_x[i] += 1
            to_y[i] += 1
        end

        @turbo for j in eachindex(fireJ)
            tr_x[j] += dt * (-tr_x[j]) / τ_x
            tr_y[j] += dt * (-tr_y[j]) / τ_y
        end

        @simd for j in findall(fireJ)
            tr_x[j] += 1
            tr_y[j] += 1
        end

    end
    # Clamp weights to the specified bounds
    @turbo for i in eachindex(W)
        @inbounds W[i] = clamp(W[i], Wmin, Wmax)
    end
end
# Function to implement STDP update rule

export STDPSymmetric, STDPAntiSymmetric
