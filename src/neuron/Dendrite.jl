## Set physiology
SNN.@load_units
struct Physiology
    Ri::Float32 ## in Ω*cm
    Rd::Float32 ## in Ω*cm^2
    Cd::Float32 ## in pF/cm^2
end

HUMAN = Physiology(200 * Ω * cm, 38907 * Ω * cm^2, 0.5μF / cm^2)
MOUSE = Physiology(200 * Ω * cm, 1700Ω * cm^2, 1μF / cm^2)

"""
    G_axial(;Ri=Ri,d=d,l=l)
    Axial conductance of a cylinder of length l and diameter d
    return Conductance in nS
"""
function G_axial(; Ri = Ri, d = d, l = l)
    ((π * d * d) / (Ri * l * 4))
end

"""
    G_mem(;Rd=Rd,d=d,l=l)
    Membrane conductance of a cylinder of length l and diameter d
    return Conductance in nS
"""
function G_mem(; Rd = Rd, d = d, l = l)
    ((l * d * π) / Rd)
end

"""
    C_mem(;Cd=Cd,d=d,l=l)
    Capacitance of a cylinder of length l and diameter d
    return Capacitance in pF
"""
function C_mem(; Cd = Cd, d = d, l = l)
    (Cd * π * d * l)
end


"""
	Dendrite

A structure representing a dendritic compartment within a neuron model.

# Fields
- `Er::FT = -70.6mV`: Resting potential.
- `C::FT = 10pF`: Membrane capacitance.
- `gax::FT = 10nS`: Axial conductance.
- `gm::FT = 1nS`: Membrane conductance.
- `l::FT = 150um`: Length of the dendritic compartment.
- `d::FT = 4um`: Diameter of the dendrite.

The type `FT` represents Float32.
"""
Dendrite

@snn_kw struct Dendrite{VFT = Vector{Float32}}
    N::Int32 = 100
    Er::VFT = zeros(N)             # (mV) resting potential
    l::VFT = zeros(N) # μm distance from next compartment
    d::VFT = zeros(N) # μm dendrite diameter
    C::VFT = zeros(N)
    gax::VFT = zeros(N)# (nS) axial conductance
    gm::VFT = zeros(N)
end

function create_dendrite(N::Int, l; kwargs...)
    dendrites = Dendrite(N = N)
    for i = 1:N
        dendrite = create_dendrite(l; kwargs...)
        dendrites.Er[i] = -70.6f0
        dendrites.l[i] = dendrite.l
        dendrites.d[i] = dendrite.d
        dendrites.C[i] = dendrite.C
        dendrites.gax[i] = dendrite.gax
        dendrites.gm[i] = dendrite.gm
    end
    return dendrites
end

function create_dendrite(l; d::Real = 4um, s = :human)
    if isa(l, Tuple)
        l = rand(l[1]:1um:l[2])
    else
        l = l
    end
    l > 500um && error("Dendrite length must be less than 500um")
    @unpack Ri, Rd, Cd = s == :mouse ? MOUSE : HUMAN
    if l <= 0
        return (gm = 1.0f0, gax = 0.0f0, C = 1.0f0, l = -1, d = d)
    else
        return (
            gm = G_mem(Rd = Rd, d = d, l = l),
            gax = G_axial(Ri = Ri, d = d, l = l),
            C = C_mem(Cd = Cd, d = d, l = l),
            l = l,
            d = d,
        )
    end
end

proximal_distal = [(150um, 400um), (150um, 400um)]
proximal_proximal = [(150um, 300um), (150um, 300um)]
proximal = [(150um, 300um)]
all_lengths = [(150um, 400um)]

export create_dendrite,
    Dendrite,
    Physiology,
    HUMAN,
    MOUSE,
    proximal_distal,
    proximal_proximal,
    proximal,
    all_lengths
