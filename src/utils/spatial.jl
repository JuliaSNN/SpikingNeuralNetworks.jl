
# Function to create a 2D spatial structure and dispose N points
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

# Function to assign a value W to each point based on their spatial distance
function periodic_distance(point1, point2, grid_size)
    dx = abs(point1[1] - point2[1])
    dy = abs(point1[2] - point2[2])

    dx = min(dx, grid_size - dx)
    dy = min(dy, grid_size - dy)

    return sqrt(dx^2 + dy^2)
end

function neurons_within_area(points, center, distance, grid_size)
    return [
        i for
        i = 1:length(points) if periodic_distance(points[i], center, grid_size) <= distance
    ]
end

function neurons_outside_area(points, center, distance, grid_size)
    return [
        i for
        i = 1:length(points) if periodic_distance(points[i], center, grid_size) > distance
    ]
end


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

    # @info "Pre: $pre, Post: $post, p_short: $(N_post*p_short), p_long: $(N_post*p_long)"
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


export create_spatial_structure,
    periodic_distance, compute_connections, neurons_within_area, neurons_outside_area
