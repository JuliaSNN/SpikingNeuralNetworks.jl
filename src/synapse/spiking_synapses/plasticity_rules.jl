function plasticity!(c::SpikingSynapse, param::no_STDPParameter, dt::Float32) end

"""
    plasticity!(c::SpikingSynapse, param::iSTDPParameterRate, dt::Float32)

Performs the synaptic plasticity calculation based on the inihibitory spike-timing dependent plasticity (iSTDP) model from Vogels (2011). 
The function updates synaptic weights `W` of each synapse in the network according to the firing status of pre and post-synaptic neurons.
This is an in-place operation that modifies the input `SpikingSynapse` object `c`.

# Arguments
- `c::SpikingSynapse`: The current spiking synapse object which contains data structures to represent the synapse network.
- `param::iSTDPParameterRate`: Parameters needed for the iSTDP model, including learning rate `η`, target rate `r`, STDP time constant `τy`, maximal and minimal synaptic weight (`Wmax` and `Wmin`).
- `dt::Float32`: The time step for the numerical integration.

# Algorithm
- For each pre-synaptic neuron, if it fires, it reduces the synaptic weight by an amount proportional to the difference between the target rate and the actual rate and increases the inhibitory term, 
  otherwise the inhibitory term decays exponentially over time with a time constant `τy`.
- For each post-synaptic neuron, if it fires, it increases the synaptic weight by an amount proportional to the pre-synaptic trace and increases the excitatory term,
  otherwise the excitatory term decays exponentially over time with a time constant `τy`.
- The synaptic weights are bounded by `Wmin` and `Wmax`.
"""
##
function plasticity!(c::SpikingSynapse, param::iSTDPParameterRate, dt::Float32)
    @unpack rowptr, colptr, index, I, J, W, tpost, tpre, fireI, fireJ, g = c
    @unpack η, r, τy, Wmax, Wmin = param

    # @inbounds 
    # if pre-synaptic inhibitory neuron fires
    for j in eachindex(fireJ) # presynaptic indices j
        tpre[j] += dt * (-tpre[j]) / τy
        if fireJ[j] # presynaptic neuron
            tpre[j] += 1
            for st = colptr[j]:(colptr[j+1]-1)
                W[st] = clamp(W[st] + η * (tpost[I[st]] - 2 * r * Hz * τy), Wmin, Wmax)
            end
        end
    end
    # if post-synaptic excitatory neuron fires
    # @inbounds 
    for i in eachindex(fireI) # postsynaptic indices i
        tpost[i] += dt * (-tpost[i]) / τy
        if fireI[i] # postsynaptic neuron
            tpost[i] += 1
            for st = rowptr[i]:(rowptr[i+1]-1) ## 
                st = index[st]
                W[st] = clamp(W[st] + η * tpre[J[st]], Wmin, Wmax)
            end
        end
    end
end

"""
    plasticity!(c::SpikingSynapse, param::iSTDPParameterRate, dt::Float32)

Performs the synaptic plasticity calculation based on the inihibitory spike-timing dependent plasticity (iSTDP) model from Vogels (2011) adapted to control the membrane potential. 
The function updates synaptic weights `W` of each synapse in the network according to the firing status of pre and post-synaptic neurons.
This is an in-place operation that modifies the input `SpikingSynapse` object `c`.

# Arguments
- `c::SpikingSynapse`: The current spiking synapse object which contains data structures to represent the synapse network.
- `param::iSTDPParameterVoltage`: Parameters needed for the iSTDP model, including learning rate `η`, target membrane potenital `v0`, STDP time constant `τy`, maximal and minimal synaptic weight (`Wmax` and `Wmin`).
- `dt::Float32`: The time step for the numerical integration.

# Algorithm
- For each pre-synaptic neuron, if it fires, it reduces the synaptic weight by an amount proportional to the difference between the target membrane potential and the actual one. 
- For each pre-synaptic neuron, if it fires, increases the inhibitory term, otherwise the inhibitory term decays exponentially over time with a time constant `τy`.
- For each post-synaptic neuron, if it fires, it increases the synaptic weight by an amount proportional to the pre-synaptic trace and increases the excitatory term, otherwise the excitatory term decays exponentially over time with a time constant `τy`.
- The synaptic weights are bounded by `Wmin` and `Wmax`.
"""
function plasticity!(c::SpikingSynapse, param::iSTDPParameterPotential, dt::Float32)
    @unpack rowptr, colptr, index, I, J, W, v_post, tpost, tpre, fireI, fireJ, g = c
    @unpack η, v0, τy, Wmax, Wmin = param

    # @inbounds 
    # if pre-synaptic inhibitory neuron fires
    @fastmath @inbounds for j in eachindex(fireJ) # presynaptic indices j
        tpre[j] += dt * (-tpre[j]) / τy
        if fireJ[j] # presynaptic neuron
            tpre[j] += 1
            for st = colptr[j]:(colptr[j+1]-1)
                W[st] = clamp(W[st] + η * (tpost[I[st]] - v0), Wmin, Wmax)
            end
        end
    end
    # if post-synaptic excitatory neuron fires
    @fastmath @inbounds for i in eachindex(fireI) # postsynaptic indices i
        # trace of the membrane potential
        tpost[i] += dt * -(tpost[i] - v_post[i]) / τy
        if fireI[i] # postsynaptic neuron
            for st = rowptr[i]:(rowptr[i+1]-1) ## 
                st = index[st]
                W[st] = clamp(W[st] + η * tpre[J[st]], Wmin, Wmax)
            end
        end
    end
end



#
# function plasticity!(
#     c::SpikingSynapse,
#     param::STDPParameter,
#     dt::Float32,
# )
#     @unpack rowptr, colptr, I, J, index, W, tpre, tpost, Apre, Apost, fireI, fireJ, g = c
#     @unpack τpre, τpost, Wmax, ΔApre, ΔApost = param
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

"""
    plasticity!(c::SpikingSynapse, param::vSTDPParameter, dt::Float32)

Perform update of synapses using plasticity rules based on the Spike Timing Dependent Plasticity (STDP) model.
This function updates pre-synaptic spike traces and post-synaptic membrane traces, and modifies synaptic weights using vSTDP rules.

# Arguments
- `c::SpikingSynapse`: The spiking synapse to be updated.
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
function plasticity!(c::SpikingSynapse, param::vSTDPParameter, dt::Float32)
    @unpack rowptr, colptr, I, J, index, W, W0, u, v, x, v_post, fireJ, t, g = c
    @unpack A_LTD, A_LTP, θ_LTD, θ_LTP, τu, τv, τx, Wmax, Wmin = param
    R(x::Float32) = x < 0.f0 ? 0.f0 : x


    @inbounds @fastmath for j in eachindex(fireJ) # Iterate over all columns, j: presynaptic neuron
        # update pre-synaptic spike trace
        x[j] += dt * (-x[j] + fireJ[j]) / τx # presynaptic neuron
        for s = colptr[j]:(colptr[j+1]-1) # Iterate over all values in column j, s: postsynaptic neuron connected to j
            # update post-synaptic membrane traces
            u[I[s]] += dt * (-u[I[s]] + v_post[I[s]]) / τu # postsynaptic neuron
            v[I[s]] += dt * (-v[I[s]] + v_post[I[s]]) / τv # postsynaptic neuron

            W[s] +=
                dt * (
                    -A_LTD * fireJ[j] * R(u[I[s]] - θ_LTD) +
                    A_LTP * x[j] * R(v_post[I[s]] - θ_LTP) * R(v[I[s]] - θ_LTD)
                )
        end
    end

    @unpack W1, μ = c.normalize
    @unpack τ, operator = c.normalize.param
    if τ > 0.0f0
        if ((t[1] + 10) % round(Int, τ / dt)) < dt
            @inbounds @fastmath for i = 1:(length(rowptr)-1) # Iterate over all postsynaptic neuron
                @simd for j = rowptr[i]:rowptr[i+1]-1 # all presynaptic neurons of i
                    W1[i] += W[index[j]]
                end
            end
        end
        if (t[1] % round(Int, τ / dt)) < dt
            @inbounds @fastmath for i = 1:(length(rowptr)-1) # Iterate over all postsynaptic neuron
                @simd for j = rowptr[i]:rowptr[i+1]-1 # all presynaptic neurons connected to neuron i
                    W[index[j]] = operator(W[index[j]], μ[i])
                end
            end
        end
    end
    @inbounds @simd for i in eachindex(W)
        W[i] = clamp.(W[i], Wmin, Wmax)
    end
end


"""
    plasticity!(c::SynapseNormalization, param::AdditiveNorm, dt::Float32)

Updates the synaptic weights using additive normalization. This function calculates 
the rate of change `μ` as the difference between initial weight `W0` and the current weight `W1`, 
normalized by `W1`. The weights are updated at intervals specified by time constant `τ`.

# Arguments
- `c`: An instance of SynapseNormalization.
- `param`: An instance of AdditiveNorm.
- `dt`: Simulation time step.
"""
function plasticity!(c::SynapseNormalization, param::AdditiveNorm, dt::Float32)
    @unpack W1, W0, μ, t = c
    @unpack τ, operator = param
    if ((t[1] + 5) % round(Int, τ / dt)) < dt
        @fastmath @inbounds @simd for i in eachindex(μ)
            μ[i] = (W0[i] - W1[i]) / W1[i]
        end
        fill!(W1, 0.0f0)
    end
end

"""
    plasticity!(c::SynapseNormalization, param::MultiplicativeNorm, dt::Float32)

Updates the synaptic weights using multiplicative normalization. This function calculates 
the rate of change `μ` as the ratio of initial weight `W0` to the current weight `W1`.
The weights are updated at intervals specified by time constant `τ`.

# Arguments
- `c`: An instance of SynapseNormalization.
- `param`: An instance of MultiplicativeNorm.
- `dt`: Simulation time step.
"""
function plasticity!(c::SynapseNormalization, param::MultiplicativeNorm, dt::Float32)
    @unpack W1, W0, μ, t = c
    @unpack τ, operator = param
    if ((t[1] + 5) % round(Int, τ / dt)) < dt
        @fastmath @inbounds @simd for i in eachindex(μ)
            μ[i] = W0[i] / W1[i]
        end
        fill!(W1, 0.0f0)
    end
end