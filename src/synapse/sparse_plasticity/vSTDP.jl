"""
    vSTDPParameter{FT = Float32} <: SpikingSynapseParameter

Contains parameters for the voltage-dependent Spike-Timing Dependent Plasticity (vSTDP) model.

# Fields
- `A_LTD::FT`: Long Term Depression learning rate.
- `A_LTP::FT`: Long Term Potentiation learning rate.
- `θ_LTD::FT`: LTD threshold.
- `θ_LTP::FT`: LTP threshold.
- `τu::FT`: Time constant for the pre-synaptic spike trace.
- `τv::FT`: Time constant for the post-synaptic membrane trace.
- `τx::FT`: Time constant for the variable `x`.
- `Wmax::FT`: Maximum synaptic weight.
- `Wmin::FT`: Minimum synaptic weight.
- `active::Vector{Bool}`: Flag to activate or deactivate the plasticity rule.
"""
vSTDPParameter

@snn_kw struct vSTDPParameter{FT = Float32} <: LTPParameter
    A_LTD::FT = 8 * 10e-5pA / mV
    A_LTP::FT = 14 * 10e-5pA / (mV * mV)
    θ_LTD::FT = -70mV
    θ_LTP::FT = -49mV
    τu::FT = 20ms
    τv::FT = 7ms
    τx::FT = 15ms
    Wmax::FT = 30.0pF
    Wmin::FT = 0.1pF
    active::Vector{Bool} = [true]
end

@snn_kw struct vSTDPVariables{VFT = Vector{Float32},IT = Int} <: LTPVariables
    ## Plasticity variables
    Npost::IT
    Npre::IT
    u::VFT = zeros(Npost) # presynaptic spiking time
    v::VFT = zeros(Npost) # postsynaptic spiking time
    x::VFT = zeros(Npost) # postsynaptic spiking time
    active::Vector{Bool} = [true]
end

function plasticityvariables(param::T, Npre, Npost) where {T<:vSTDPParameter}
    return vSTDPVariables(Npre = Npre, Npost = Npost)
end

"""
    plasticity!(c::AbstractSparseSynapse, param::vSTDPParameter, dt::Float32)

Perform update of synapses using plasticity rules based on the Spike Timing Dependent Plasticity (STDP) model.
This function updates pre-synaptic spike traces and post-synaptic membrane traces, and modifies synaptic weights using vSTDP rules.

# Arguments
- `c::AbstractSparseSynapse`: The spiking synapse to be updated.
- `param::vSTDPParameter`: Contains STDP parameters including A_LTD, A_LTP, θ_LTD, θ_LTP, τu, τv, τx, Wmax, Wmin.
    - `A_LTD`: Long Term Depression learning rate.
    - `A_LTP`: Long Term Potentiation learning rate.
    - `θ_LTD`: LTD threshold.
    - `θ_LTP`: LTP threshold.
    - `τu, τv, τx`: Time constants for different variables in STDP.
    - `Wmax, Wmin`: Maximum and minimum synaptic weight.
- `dt::Float32`: Time step for simulation.

In addition to these, the function uses normalization where the operator can be multiplicative or additive as defined by `c.normalize.param.operator`.
The `operator` is applied when updating the synaptic weights. The frequency of normalization is controlled by `τ`, 
where if `τ > 0.0f0` then normalization will occur at intervals approximately equal to `τ`.

After all updates, the synaptic weights are clamped between `Wmin` and `Wmax`.

"""

function plasticity!(
    c::PT,
    param::vSTDPParameter,
    plasticity::vSTDPVariables,
    dt::Float32,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack rowptr, colptr, I, J, index, W, v_post, fireJ, g, index = c
    @unpack u, v, x = plasticity
    @unpack A_LTD, A_LTP, θ_LTD, θ_LTP, τu, τv, τx, Wmax, Wmin = param
    # R(x::Float32) = x < 0.0f0 ? 0.0f0 : x

    # update pre-synaptic spike trace
   @fastmath @inbounds begin 
        @turbo for j in eachindex(fireJ) # Iterate over all columns, j: presynaptic neuron
            x[j] += dt * (-x[j] + fireJ[j]) / τx
        end

        Is = 1:(length(rowptr)-1)
        @turbo for i in eachindex(Is) # Iterate over postsynaptic neurons
            u[i] += dt * (-u[i] + v_post[i]) / τu # postsynaptic neuron
            v[i] += dt * (-v[i] + v_post[i]) / τv # postsynaptic neuron
        end
        # @simd for s = colptr[j]:(colptr[j+1]-1) 
        Threads.@threads :static for j in eachindex(fireJ) # Iterate over presynaptic neurons
            if fireJ[j]
                @turbo for s = colptr[j]:(colptr[j+1]-1)
                    W[s] += -A_LTD * clamp(u[I[s]] - θ_LTD, 0.0f0, Inf)
                end
            end
            @turbo for s = colptr[j]:(colptr[j+1]-1)
                W[s] +=
                    A_LTP *
                    x[j] *
                    clamp(v[I[s]] - θ_LTD, 0.0f0, Inf) *
                    clamp(v_post[I[s]] - θ_LTP, 0.0f0, Inf)
            end
        end

        @turbo for i in eachindex(W)
            W[i] = clamp(W[i], Wmin, Wmax)
        end
    end
end

export vSTDPParameter, vSTDPVariables, plasticityvariables, plasticity!


# @inbounds @fastmath @simd for i in eachindex(fireI) # Iterate over postsynaptic neurons
#     u[i] += dt * (-u[i] + v_post[i]) / τu # postsynaptic neuron
#     v[i] += dt * (-v[i] + v_post[i]) / τv # postsynaptic neuron
# end

# @inbounds @fastmath  for i in eachindex(Is) # Iterate over postsynaptic neurons
#     ltd_v = (v[i] - θ_LTD)
#     ltp = (v_post[i] - θ_LTP)
#     @simd for s = rowptr[i]:(rowptr[i+1]-1)
#         j = J[index[s]]
#         if fireJ[j] && (u[i] - θ_LTD) > 0.0f0
#             W[index[s]] += -A_LTD * (u[i] - θ_LTD)
#         end
#         if ltp > 0.0f0 && ltd_v > 0.0f0
#             W[index[s]] += A_LTP * x[j] * ltp * ltd_v
#         end
#     end
# end
