using Random
using LinearAlgebra
using Parameters
using SpecialFunctions

"""
    create_spatial_structure(config)

Create a 2D spatial structure and dispose N points for each population.

# Arguments
- `config::Dict`: A dictionary containing configuration parameters, including `projections` and `Npop`.

# Returns
- `Pops::NamedTuple`: A named tuple containing the spatial points for each population.
"""
function create_spatial_structure(config)
    @unpack projections, Npop = config
    @unpack grid_size = projections
    Pops = Dict{Symbol,Vector}()
    for k in keys(Npop)
        points = [rand(2) .* grid_size for _ = 1:Npop[k]]
        Pops[k] = points
    end
    return Pops |> dict2ntuple
end

"""
    periodic_distance(point1, point2, grid_size)

Calculate the periodic distance between two points in a 2D grid.

# Arguments
- `point1::Vector{Float64}`: The coordinates of the first point.
- `point2::Vector{Float64}`: The coordinates of the second point.
- `grid_size::Float64`: The size of the grid.

# Returns
- `distance::Float64`: The periodic distance between the two points.
"""
function periodic_distance(point1, point2, grid_size)
    dx = abs(point1[1] - point2[1])
    dy = abs(point1[2] - point2[2])

    dx = min(dx, grid_size - dx)
    dy = min(dy, grid_size - dy)

    return sqrt(dx^2 + dy^2)
end

"""
    neurons_within_area(points, center, distance, grid_size)

Find the indices of neurons within a specified area around a center point.

# Arguments
- `points::Vector{Vector{Float64}}`: The coordinates of the neurons.
- `center::Vector{Float64}`: The coordinates of the center point.
- `distance::Float64`: The maximum distance from the center point.
- `grid_size::Float64`: The size of the grid.

# Returns
- `indices::Vector{Int}`: The indices of neurons within the specified area.
"""
function neurons_within_area(points, center, distance, grid_size)
    return [
        i for
        i = 1:length(points) if periodic_distance(points[i], center, grid_size) <= distance
    ]
end

"""
    neurons_outside_area(points, center, distance, grid_size)

Find the indices of neurons outside a specified area around a center point.

# Arguments
- `points::Vector{Vector{Float64}}`: The coordinates of the neurons.
- `center::Vector{Float64}`: The coordinates of the center point.
- `distance::Float64`: The minimum distance from the center point.
- `grid_size::Float64`: The size of the grid.

# Returns
- `indices::Vector{Int}`: The indices of neurons outside the specified area.
"""
function neurons_outside_area(points, center, distance, grid_size)
    return [
        i for
        i = 1:length(points) if periodic_distance(points[i], center, grid_size) > distance
    ]
end

"""
    compute_connections(pre::Symbol, post::Symbol, points; dc, pl, ϵ, grid_size, conn)

Compute the connections between two populations of neurons based on their spatial distance. This function will assign connections with probability `p_short` for short-range connections and `p_long` for long-range connections. The weights of the connections are determined by the `μ` parameter in the `conn` named tuple. 
The function uses a periodic boundary condition to calculate distances in a 2D grid.
The total number of connections per the post-synaptic neuron is: ϵ * N_pre * p_short + (1 - ϵ) * N_pre * p_long.

# Arguments
- `pre::Symbol`: The symbol representing the pre-synaptic population.
- `post::Symbol`: The symbol representing the post-synaptic population.
- `points::NamedTuple`: A named tuple containing the spatial points for each population.
- `dc::Float64`: The critical distance for short-range connections.
- `pl::Float64`: The probability of long-range connections.
- `ϵ::Float64`: The scaling factor for connection probabilities.
- `grid_size::Float64`: The size of the grid.
- `conn::NamedTuple`: A named tuple containing connection parameters, including `p` and `μ`.

# Returns
- `P::Matrix{Bool}`: A matrix indicating the presence of connections.
- `W::Matrix{Float32}`: A matrix containing the weights of the connections.
"""
function compute_connections(pre::Symbol, post::Symbol, points; dc, pl, ϵ, grid_size, conn)
    γs = 1 / (π * dc^2)
    γl = 1 / (1 - π * dc^2)
    p_short = (1 - pl) * γs * ϵ * conn.p
    p_long = (pl) * γl * ϵ * conn.p

    N_pre = length(getfield(points, pre))
    N_post = length(getfield(points, post))
    pre_points = getfield(points, pre)
    post_points = getfield(points, post)
    P = zeros(Bool, N_post, N_pre)
    W = zeros(Float32, N_post, N_pre)

    @inbounds for j = 1:N_pre
        for i = 1:N_post
            pre == post && i == j && continue
            distance = periodic_distance(post_points[i], pre_points[j], grid_size)
            if distance < dc
                if rand() < p_short
                    P[i, j] = true
                    W[i, j] = conn.μ
                end
            else
                if rand() < p_long
                    P[i, j] = true
                    W[i, j] = conn.μ
                end
            end
        end
    end

    return P, W
end

"""
    linear_network(N, σ_w=0.38, w_max=2.0)

Create a linear network with Gaussian-shaped connections.

# Arguments
- `N::Int`: The number of neurons.
- `σ_w::Float64`: The standard deviation of the Gaussian distribution.
- `w_max::Float64`: The maximum weight.

# Returns
- `W::Matrix{Float32}`: A matrix containing the weights of the connections.
"""
function linear_network(N; σ_w = 0.38, w_max = 2.0)
    # Function to calculate wθ^sE
    function wθ_sE(θ_j, θ_i, w_0, w_, σ_w)
        return w_0 +
               (w_ - w_0) * exp(-(min(abs(θ_j - θ_i), 2π - abs(θ_j - θ_i)))^2 / (2 * σ_w^2))
    end

    # Function to calculate w_0
    function w_0(w, σ_w)
        return w * σ_w * (erf(π / (sqrt(2) * σ_w)) - sqrt(2π)) /
               (σ_w * erf(π / (sqrt(2) * σ_w)) - sqrt(2π))
    end

    w_norm = w_0(w_max, σ_w)

    neuron_position = [i * 2π / N for i = 1:N]
    W = zeros(N, N)
    for i = 1:N
        for j = 1:N
            W[i, j] = wθ_sE(neuron_position[i], neuron_position[j], w_norm, w_max, σ_w)
            W[j, j] = 0.0f0
        end
    end
    return W
end

export create_spatial_structure,
    periodic_distance,
    compute_connections,
    neurons_within_area,
    neurons_outside_area,
    linear_network
