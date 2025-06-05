using DSP
using Statistics
using Distributions

"""
    inter_spike_interval(spiketimes::Vector{Float32})

Calculate the inter-spike intervals (ISIs) for a given set of spike times.

# Arguments
- `spiketimes`: A vector of spike times for a single neuron.

# Returns
- `isis`: A vector of inter-spike intervals.
"""
function asynchronous_state(model, interval=nothing, pop=:Exc)
    population = getfield(model.pop, pop)
    interval = interval === nothing ? (0s:0.5s:get_time(model)) : interval
    bins, _ = SNN.bin_spiketimes(population; interval, do_sparse = false)
    # Calculate the Coefficient of Variation (CV) of ISIs
    isis = isi(population; interval)
    cv = std.(isis) ./ (mean.(isis) .+ 1e-6)  # Adding a small value to avoid division by zero
    cv[isnan.(cv)] .= 0.0  # Replace NaN values with 0.0
    cv = mean(cv)

    # Calculate the Fano Factor (FF)
    ff = var(bins) / mean(bins)  # Fano Factor

    ## Calculate the Synchrony Index (SI)
    si = mean(cov(bins, dims=2))

    return cv, ff, si
end

"""
    is_attractor_state(spiketimes::Spiketimes, interval::AbstractVector, N::Int)

Check if the network is in an attractor state by verifying that the average firing rate over the last N seconds of the simulation is a unimodal distribution.

# Arguments
- `spiketimes`: A vector of vectors containing the spike times of each neuron.
- `interval`: The time interval over which to compute the firing rate.
- `N`: The number of seconds over which to check the unimodality of the firing rate distribution.

# Returns
- `is_attractor`: A boolean indicating whether the network is in an attractor state.
"""
function is_attractor_state(
    pop::T,
    interval::AbstractVector;
    ratio::Real = 0.3,
    σ::Real = 10.0f0,
    false_value = 10,
) where {T<:SNN.AbstractPopulation}
    # Calculate the firing rate over the last N seconds

    rates, r = firing_rate(pop; interval, interpolate = true)
    ave_rate = mean(rates, dims = 2)[:, 1]
    kde = gaussian_kernel_estimate(ave_rate, σ, boundary = :continuous)
    # Check if the firing rate distribution is unimodal
    if (is_unimodal(kde, ratio) || is_unimodal(circshift(kde, length(kde) ÷ 2), ratio))
        # get the half width in σ
        peak, center = findmax(kde)
        return length(findall(x -> x > peak/2, kde))/σ, kde
    else
        return false_value, kde
    end
end


# Find bimodal value
# Here I use a simple algorithm that is described in :
# Journal of the Royal Statistical Society. Series B (Methodological)
# Using Kernel Density Estimates to Investigate Multimodality
# https://www.jstor.org/stable/2985156

# It consists in  using Normal kernels to approximate the data and then leverages a theorem on decreasing monotonicity of the number of maxima as function of the window span.

# # Kernel Density Estimation
# function KDE(t::Real, h::Real, ys)
#     ndf(x, h) = exp(-x^2 / h)
#     1 / length(data) * 1 / h * sum(ndf.(ys .- t, h))
# end

# # Distribution
# function globalKDE(h::Real, ys; xs::AbstractVector, distance::Function )
#     kde = zeros(Float64, length(xs))
#     @fastmath @inbounds for n = eachindex(xs)
#             kde[n] = KDE(xs[n], h, ys)
#     end
#     return kde
# end

#Get its maxima
function get_maxima(data)
    arg_maxima = []
    for x = 2:(length(data)-1)
        (data[x] > data[x-1]) && (data[x] > data[x+1]) && (push!(arg_maxima, x))
    end
    return arg_maxima
end

#Trash spurious values (below 30% of the true maximum)
function is_unimodal(kernel, ratio)
    maxima = get_maxima(kernel)
    z = maximum(kernel[maxima])
    real = []
    for n in maxima
        m = kernel[n]
        if (abs(m / z) > ratio)
            push!(real, m)
        end
    end
    if length(real) > 1
        return false
    else
        return true
    end
end


export is_unimodal,
    get_maxima,
    gaussian_kernel_estimate,
    gaussian_kernel,
    asynchronous_state

# #Trash spurious values (below 30% of the true maximum)
# function count_maxima(kernel, ratio)
#     maxima = get_maxima(kernel)
#     z = maximum(kernel[maxima])
#     real_maxima = []
#     for n in maxima
#         m = kernel[n]
#         if (abs(m / z) > ratio)
#             push!(real_maxima, m)
#         end
#     end
#     return length(real_maxima)
# end

# # Return the critical window (hence the bimodal factor)
# function critical_window(data; ratio = 0.3, max_b = 50, v_range = collect(-90:-35))
#     for h = 1:max_b
#         kernel = globalKDE(h, data, v_range = v_range)
#         bimodal = false
#         try
#             bimodal = isbimodal(kernel, ratio)
#         catch
#             bimodal = false
#             @error "Bimodal failed"
#         end
#         if !bimodal
#             return h
#         end
#     end
#     return max_b
# end

# # Return the critical window (hence the bimodal factor)
# function all_windows(data, ratio = 0.3; max_b = 50)
#     counter = zeros(max_b)
#     for h = 1:max_b
#         kernel = globalKDE(h, data)
#         counter[h] = count_maxima(kernel, ratio)
#     end
#     return counter
# end

"""
    gaussian_kernel(σ::Float64, length::Int)

Create a Gaussian kernel with standard deviation `σ` and specified `length`.
# Arguments
- `σ`: Standard deviation of the Gaussian kernel.
- `length`: Length of the kernel.
# Returns
- `kernel`: A vector representing the Gaussian kernel.
"""
function gaussian_kernel(σ::Real, ll::Int)
    t = range(-(ll ÷ 2), stop = ll ÷ 2, length = ll)
    kernel = exp.(-(t .^ 2) / (2 * σ^2))
    return kernel ./ sum(kernel)  # Normalize the kernel
end

"""
    gaussian_kernel_estimate(support_vector::Vector{Float64}, σ::Float64, length::Int)

Apply a Gaussian kernel estimate to a support vector with closed boundary conditions.

# Arguments
- `support_vector`: The input support vector.
- `σ`: Standard deviation of the Gaussian kernel.
- `length`: Length of the kernel.

# Returns
- `estimated_vector`: The estimated vector after applying the Gaussian kernel.
"""
function gaussian_kernel_estimate(support_vector::Vector, σ::Real; boundary = :continuous)

    # Apply the kernel using convolution
    # estimated_vector = conv(support_vector, kernel)
    kernel = gaussian_kernel(σ, length(support_vector))

    # Handle closed boundary conditions
    # Extend the support vector to handle boundaries
    ll = length(support_vector) ÷ 2
    if boundary == :continuous
        extension_left = support_vector[(end-ll):end]
        extension_right = support_vector[1:ll]
    elseif boundary == :closed
        extension_left = zeros(size(support_vector[(end-ll):end]))
        extension_right = zeros(size(support_vector[1:ll]))
    else
        error("Invalid boundary condition. Use :continuous or :closed.")
    end
    extended_vector = vcat(extension_left, support_vector, extension_right)

    # Apply the kernel to the extended vector
    extended_estimated_vector = conv(extended_vector, kernel)

    return extended_estimated_vector[2(1+ll):(end-2ll)]
end



export gaussian_kernel_estimate,
    gaussian_kernel



# # Example support vector
# support_vector = [2.0, 2.0, 4.0, 3.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 4.0, 5.0, 4.0, 3.0, 2.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 8.0, 10,]

# # Standard deviation of the Gaussian kernel
# σ = 1.0

# # Length of the kernel

# # Apply the Gaussian kernel estimate
# estimated_vector = gaussian_kernel_estimate(support_vector, 2.0, boundary=:continuous)
# rotated_array = circshift(estimated_vector, 10)
