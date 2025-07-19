using Random
using LinearAlgebra
using Parameters
using SpecialFunctions

"""
    place_populations(config)

Create a 2D spatial structure and dispose N points for each population.

# Arguments
- `config::NamedTuple`: A NamedTuple containing configuration parameters, including `projections` and `Npop`.

# Returns
- `Pops::NamedTuple`: A named tuple containing the spatial points for each population.
"""
function place_populations(Npop, grid_size)
    Pops = Dict{Symbol,Vector}()
    for k in keys(Npop)
        !(typeof(Npop[k]) == Int64) && continue
        points = [rand(length(grid_size)) .* grid_size for _ = 1:Npop[k]]
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
    isa(grid_size, Vector) && @assert length(grid_size) == length(point1) "grid_size must match the dimension of points: $(point1), $(length(grid_size))"
    typeof(grid_size) <: Real && (grid_size = repeat([grid_size], length(point1)))
    @assert length(point1) == length(point2) "point1 and point2 must have the same dimension" 
    return sqrt(
            sum(
                map(eachindex(point1)) do n
                    min(abs(point1[n] - point2[n]), grid_size[n] - abs(point1[n] - point2[n]))
            end).^2
            )
end

"""
    neurons_within_circle(points, center, distance, grid_size)

Find the indices of neurons within a specified area around a center point.

# Arguments
- `points::Vector{Vector{Float64}}`: The coordinates of the neurons.
- `center::Vector{Float64}`: The coordinates of the center point.
- `distance::Float64`: The maximum distance from the center point.
- `grid_size::Float64`: The size of the grid.

# Returns
- `indices::Vector{Int}`: The indices of neurons within the specified area.
"""
function neurons_within_circle(points, center, distance, grid_size)
    map(x->periodic_distance(x, center, grid_size) <= distance, points)
end

function neurons_within(func::Function, kwargs...)
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
    compute_long_short_connections(pre::Symbol, post::Symbol, points; dc, pl, ϵ, grid_size, conn)

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
function compute_connections(pre::Symbol, post::Symbol, points; conn, spatial)
    @unpack grid_size = spatial
    @assert length(grid_size) == 2 "grid_size must be a vector of length 2"
    pre_points = getfield(points, pre)
    post_points = getfield(points, post)
    N_pre = length(getfield(points, pre))
    N_post = length(getfield(points, post))
    P = falses(N_post, N_pre)
    W = zeros(Float32, N_post, N_pre)

    if spatial.type == :critical_distance
        @unpack dc, ϵ, grid_size, p_long = spatial
        pl = getfield(spatial.p_long, pre)
        area = grid_size[1]*grid_size[2]
        γs = area / (π * dc^2)
        γl = area / (area - π * dc^2)
        p_short = (1 - pl) * γs * ϵ * conn.p
        p_long = (pl) * γl * ϵ * conn.p
        @inbounds for j = 1:N_pre
            for i = 1:N_post
                pre == post && i == j && continue
                distance = periodic_distance(post_points[i], pre_points[j], grid_size)
                if distance < dc
                    if rand() <= p_short
                        P[i, j] = true
                        W[i, j] = conn.μ
                    end
                else
                    if rand() <= p_long
                        P[i, j] = true
                        W[i, j] = conn.μ
                    end
                end
            end
        end
        return P, W
    end
    if spatial.type == :gaussian
        function gaussian_weight(pre, post=[0f0,0.0f0]; σx::Float32, σy::Float32, grid_size::Vector{Float32})
            # @fastmath @inbounds 
            begin
                x = periodic_distance(post[1], pre[1], grid_size[1])
                y = periodic_distance(post[2], pre[2], grid_size[2])
                return SNN.exp64(-(x/σx)^2 -(y/σy)^2)
            end
        end
        @unpack σs, grid_size, ϵ = spatial
        X, Y  = grid_size
        x = range(-X/2, stop=X/2, length=200) |> collect
        y = range(-Y/2, stop=Y/2, length=200) |> collect
        σx, σy = Float32.(getfield(σs, pre))
        γ = 1/mean(map(Iterators.product(x, y)) do point
            gaussian_weight(point;  σx, σy, grid_size)
        end)
        p =Float32(conn.p * ϵ * γ)
        randcache = rand(N_post, N_pre)
        for j = 1:N_pre
            for i = 1:N_post
                pre == post && i == j && continue
                p == 0 && continue
                randcache[i, j] > p && continue
                if randcache[i, j] <= p * gaussian_weight(pre_points[j], post_points[i]; σx=σx, σy=σy, grid_size)
                    P[i, j] =true
                    W[i, j] = conn.μ
                end
            end
        end
        @info "$pre => $post average conn weight: $(mean(W))"
        @info "$pre => $post average conn probability: $(mean(P))"
        return P, W
    end
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
function linear_network(N; σ_w = 0.38, w_max = 2.0, kwargs...)
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

"""
    spatial_activity(points, activity; N, L, grid_size=(x=0.1, y=0.1))

Compute the spatial average of activity data over a grid.

# Arguments
- `points::Tuple{Vector{Float64}, Vector{Float64}}`: A tuple containing two vectors `xs` and `ys`, which represent the x and y coordinates of the points.
- `activity::Matrix{Float64}`: A matrix where rows correspond to points and columns correspond to activity values over time.
- `N::Int`: The number of time steps to group together for averaging.
- `L::Float64`: The size of each grid cell in both x and y directions.
- `grid_size::NamedTuple{(:x, :y), Tuple{Float64, Float64}}` (optional): The total size of the grid in the x and y directions. Defaults to `(x=0.1, y=0.1)`.

# Returns
- `spatial_avg::Array{Float64, 3}`: A 3D array where the first two dimensions correspond to the grid cells in the x and y directions, and the third dimension corresponds to the time groups. Each element contains the average activity for the points within the corresponding grid cell and time group.
- `x_range::Vector{Float64}`: A vector representing the range of x coordinates for the grid cells.
- `y_range::Vector{Float64}`: A vector representing the range of y coordinates for the grid cells.

# Details
The function divides the spatial domain into a grid based on the `grid_size` and `L` parameters. For each grid cell, it computes the average activity of the points that fall within the cell over time groups defined by `N`. If no points are found in a grid cell, the average for that cell is skipped.

# Example
```julia
xs = [0.05, 0.15, 0.25, 0.35]
ys = [0.05, 0.15, 0.25, 0.35]
points = (xs, ys)
activity = rand(4, 100)  # Random activity data for 4 points over 100 time steps
N = 10
L = 0.1
grid_size = (x=0.4, y=0.4)

spatial_avg, x_range, y_range = spatial_activity(points, activity; N, L, grid_size)
```
"""
function spatial_activity(points, activity; N, L, grid_size=(x= [0,0.1], y= [0,0.1]))
    xs, ys = points
    _, num_values = size(activity)
    num_matrices = num_values ÷ N

    # Define the grid size
    @unpack x, y = grid_size
    x_range = ceil(Int,diff(collect(x))[1] / L)
    y_range = ceil(Int,diff(collect(y))[1] / L)

    spatial_avg = Array{Any,3}(undef, x_range, y_range, num_matrices)
    spatial_avg[:].=0
    for t in 1:num_matrices
        time_index = (1+(t-1) * N) : (t * N - 1)
        for j  in 1:x_range
            for k in 1:y_range
                # Find points within the current grid cell
                indices_x = findall(_x-> ((j-1)*L <= _x-x[1] < j*L), xs) |> Set
                indices_y = findall(_y-> ((k-1)*L <= _y-y[1] < k*L), ys) |> Set
                indices = intersect(indices_x, indices_y) |> collect
                isempty(indices) && continue
                spatial_avg[j, k, t] = mean(activity[indices, time_index])
            end
        end
    end
    x_range  = range(x[1], stop=x[end], length=x_range)
    y_range  = range(y[1], stop=y[end], length=y_range)
    return spatial_avg, x_range, y_range
end

export place_populations,
    periodic_distance,
    compute_connections,
    neurons_within_circle,
    neurons_outside_area,
    linear_network,
    spatial_activity
