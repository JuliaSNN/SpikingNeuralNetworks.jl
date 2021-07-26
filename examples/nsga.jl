ENV["PYTHON_JL_RUNTIME_PYTHON"] = Sys.which("python")
using Pkg
using PyCall
using OrderedCollections
using LinearAlgebra
#using UnicodePlots
using NSGAII
using SpikeSynchrony
include("../src/SpikingNeuralNetworks.jl")
include("../src/units.jl")
SNN = SpikingNeuralNetworks
using Evolutionary, Test, Random
using Plots
import DataStructures
using JLD
using Evolutionary, Test, Random

using Debugger
unicodeplots()


if isfile("ground_truth.jld")
    vmgtv = load("ground_truth.jld","vmgtv")
    ngt_spikes = load("ground_truth.jld","ngt_spikes")
    gt_spikes = load("ground_truth.jld","gt_spikes")

    ground_spikes = gt_spikes
    ngt_spikes = size(gt_spikes)[1]
    vmgtt = load("ground_truth.jld","vmgtt")

    #plot(Plot(vmgtv,vmgtt,w=1))

else

    py"""
    options = (
        325479788,
        324257146,
        476053392,
        623893177,
        623960880,
        482493761,
        471819401
    )
    specimen_id = options[1]
    target_num_spikes=7
    from neuronunit.allenapi import make_allen_tests_from_id
    sweep_numbers, data_set, sweeps = make_allen_tests_from_id.allen_id_to_sweeps(specimen_id)
    (vmm,stimulus,sn,spike_times) = make_allen_tests_from_id.get_model_parts_sweep_from_spk_cnt(
        target_num_spikes, data_set, sweep_numbers, specimen_id
    )
    """
    gt_spikes = py"spike_times"
    ground_spikes = gt_spikes

    ngt_spikes = size(gt_spikes)[1]
    vmgtv = py"vmm.magnitude"
    vmgtt = py"vmm.times"


    save("ground_truth.jld", "vmgtv", vmgtv,"vmgtt",vmgtt, "ngt_spikes", ngt_spikes,"gt_spikes",gt_spikes)

end

#plot(vmgtt[:],vmgtv[:])
plot(vmgtt[:],vmgtv[:]) |> display

ALLEN_DURATION = 2000 * ms
ALLEN_DELAY = 1000 * ms
#break_on(:error)
function get_ranges()
    ranges = DataStructures.OrderedDict{Char,Float32}()
    ranges = ("a"=>[0.002,0.3],"b"=>[0.02,0.36],"c"=>[-75,-35],"d"=>[0.005,16])#,"I"=>[100,9000])
    lower = []
    upper = []
    for (k,v) in ranges
        append!(lower,v[1])
        append!(upper,v[2])
    end
    lower,upper
end

function init_b(lower,upper)
    gene = []
    #chrome = Float32[size(lower)[1]]
    for (i,(l,u)) in enumerate(zip(lower,upper))
        p1 = rand(l:u, 1)
        append!(gene,p1)
        #chrome[i] = p1
    end
    gene
end

function initf(n)
    genesb = []
    for i in 1:n
        genes = init_b(lower,upper)
        append!(genesb,[genes])
    end
    genesb
end
lower,upper = get_ranges()

function raster_synchp(p)
    fire = p.records[:fire]
    spikes = Float32[]
    #neurons = Float32[]#, Float32[]
    for time = eachindex(fire)
        for neuron_id in findall(fire[time])
            push!(spikes,time)
        end
    end
    spikes
end
function loss(E,ngt_spikes,ground_spikes)
    spikes = raster_synchp(E)
    spikes = [s/1000.0 for s in spikes]
    maxt = findmax(sort!(unique(vcat(spikes,ground_spikes))))[1]
    if size(spikes)[1]>1
        t, S = SPIKE_distance_profile(spikes, ground_spikes;t0=0,tf = maxt)
        spkd = SpikeSynchrony.trapezoid_integral(t, S)/(t[end]-t[1]) # == SPIKE_distance(y1, y2)
    else
        spkd = 1.0
    end
    #delta = size(spikes)[1] - ngt_spikes
    #=
    if delta == 0
        #println()
        #println(" hit ")
        comp = 10.0
    else
        #println(abs(delta))
        comp = (1.0/abs(delta))
    end
    =#
    #spkd = 1.0-spkd
    #err = spkd #+ comp
    return spkd#+spkd

end
#=
function ConstraintVioloation(x)
    if size(spikes)[1] == size(ground_spikes)[1]
        return 0
    else
        return size(spikes)[1] + size(ground_spikes)[1]
    end
end
=#
function test_current(param,current,ngs)
    #println(param)
    CH = SNN.IZParameter(;a = param[1], b = param[2], c = param[3], d = param[4])
    E = SNN.IZ(;N = 1, param = CH)
    SNN.monitor(E, [:v])
    SNN.monitor(E, [:fire])
    ALLEN_DURATION = 2000 * ms
    ALLEN_DELAY = 1000 * ms

    E.I = [current*nA]#[param[5]*nA]
    SNN.sim!([E], []; dt =1*ms, delay=ALLEN_DELAY,stimulus_duration=ALLEN_DURATION,simulation_duration = ALLEN_DURATION+ALLEN_DELAY+500ms)
    #v = SNN.getrecord(E, :v)
    #SNN.vecplot(E, :v) |> display

    spikes = raster_synchp(E)
    spikes = [s/1000.0 for s in spikes]
    #println(spikes)
    nspk = size(spikes)[1]
    delta = nspk - ngs
    #println(nspk)
    #if delta > 0
    #    less_current = true
    #else
    #    less_current = false
    #end
    #println(delta)
    return delta
end
function test_c(check_values,param,ngt_spikes,current_dict)
    nspk = -1.0
    for i_c in check_values
        nspk = test_current(param,i_c,ngt_spikes)
        current_dict[nspk] = i_c
        if nspk == 0
            if 0 in keys(current_dict)
                return current_dict,0
            end
        end
    end
    return current_dict,nspk
end

function current_search(param)

    current_dict = Dict()
    minc = 0.0
    maxc = 90000.0
    step_size = (maxc-minc)/10.0
    check_values = minc:step_size:maxc
    current_dict = Dict()
    cnt = 0
    while (0 in keys(current_dict))==false

        current_dict,nspk = test_c(check_values,param,ngt_spikes,current_dict)
        over_s = Dict([(k,v) for (k,v) in current_dict if k>0])
        under_s = Dict([(k,v) for (k,v) in current_dict if k<0])
        if length(over_s)>0
            # find the lowest current that caused more than one spike
            # throw away part of dictionary that has no spikes
            # find minimum of value in the remaining dictionary
            new_top = findmin(collect(values(over_s)))[1]
            #new_bottom = minc-minc*(1.0/2.0)

        else
            new_top = maxc*2

        end
        if length(under_s)>0
            #new_top = maxc*2
            # find the lowest current that caused more than one spike
            # throw away part of dictionary that has no spikes
            # find minimum of value in the remaining dictionary
            new_bottom = findmax(collect(values(under_s)))[1]

        else
            new_bottom = minc-minc*(1.0/2.0)
        end

        step_size = abs(new_top-new_bottom)/10.0
        if step_size==0
            return findmin(collect(values(current_dict)))[1]
        end
        check_values = new_bottom:step_size:new_top

        cnt+=1
        if cnt >140
            return findmin(collect(values(current_dict)))[1]
        end
    end
    return current_dict[0]
end


function zz_(param)

    #param = x
    #println(param)
    #bincoded = NSGAII.encode(seed, bc)
    #NSGAII.decode!(bincoded, bc)
    #bc = BinaryCoding(6, [:Float32,:Float32,:Float32,:Float32,:Float32], lower, upper)

    #bc = BinaryCoding(size(lower)[1], lower, upper)
    #println(param)
    CH = SNN.IZParameter(;a = param[1], b = param[2], c = param[3], d = param[4])
    E = SNN.IZ(;N = 1, param = CH)
    SNN.monitor(E, [:v])
    SNN.monitor(E, [:fire])
    ALLEN_DURATION = 2000 * ms
    ALLEN_DELAY = 1000 * ms
    #T = (ALLEN_DURATION+ALLEN_DELAY)*ms
    E.I = [current_search(param)*nA]
    #print(param[5])
    #print(param)
    #println("fail")
    SNN.sim!([E], []; dt =1*ms, delay=ALLEN_DELAY,stimulus_duration=ALLEN_DURATION,simulation_duration = ALLEN_DURATION+ALLEN_DELAY+443ms)
    #v = SNN.getrecord(E, :v)
    #SNN.vecplot(E, :v) |> display
    #function loss(E,ngt_spikes,ground_spikes)

    error = loss(E,ngt_spikes,ground_spikes)
    #println(error)
    error
end


function z(x::AbstractVector)
    param = x
    CH = SNN.IZParameter(;a = param[1], b = param[2], c = param[3], d = param[4])
    E = SNN.IZ(;N = 1, param = CH)
    SNN.monitor(E, [:v])
    SNN.monitor(E, [:fire])
    ALLEN_DURATION = 2000 * ms
    ALLEN_DELAY = 1000 * ms
    E.I = [param[5]*nA]
    SNN.sim!([E], []; dt =1*ms, delay=ALLEN_DELAY,stimulus_duration=ALLEN_DURATION,simulation_duration = ALLEN_DURATION+ALLEN_DELAY+500ms)
    error = loss(E,ngt_spikes)
    error
end


#function rastrigin(x::AbstractVector{T}) where {T <: AbstractFloat}
#    n = length(x)
#    return 10n + sum([ x[i]^2 - 10cos(convert(T,2π*x[i])) for i in 1:n ])
#end

#function rastrigin(x::AbstractVector{T}) where {T <: AbstractFloat}
#    n = length(x)
#    return 10n + sum([ x[i]^2 - 10cos(convert(T,2π*x[i])) for i in 1:n ])
#end

function checkmodel(param)
    CH = SNN.IZParameter(;a = param[1], b = param[2], c = param[3], d = param[4])
    E = SNN.IZ(;N = 1, param = CH)
    SNN.monitor(E, [:v])
    SNN.monitor(E, [:fire])
    ALLEN_DURATION = 2000 * ms
    ALLEN_DELAY = 1000 * ms
    #E.I = [param[5]*nA]
    E.I = [current_search(param)*nA]

    SNN.sim!([E], []; dt =0.001*ms, delay=ALLEN_DELAY,stimulus_duration=ALLEN_DURATION,simulation_duration = ALLEN_DURATION+ALLEN_DELAY+500ms)
    #v = SNN.getrecord(E, :v)
    vec = SNN.vecplot(E, :v)

    vec |> display
end


lower,upper = get_ranges()
function initd()
    population = initf(10)
    garray = zeros((length(population)[1], length(population[1])))
    for (i,p) in enumerate(population)
        garray[i,:] = p
    end
    garray[1,:]
end
cb = Evolutionary.ConstraintBounds(lower,upper,lower,upper)
selections = [:roulette=>rouletteinv, :sus=>susinv, :rank=>ranklinear(1.5)]
crossovers = [:discrete=>discrete, :intermediate0=>intermediate(0.), :intermediate0_25=>intermediate(0.5), :line=>line(0.2)]
mutations = [:domrng0_5=>domainrange(fill(0.5,4)), :uniform=>uniform(3.0), :gaussian=>gaussian(0.6)]
etas = [0.25,0.35,0.5,0.75]#,0.8]
#temp_GA
meta_param_dict = Dict()
for (sn,ss) in selections, (xn,xovr) in crossovers, (mn,ms) in mutations, (ɛ) in etas
    xn == :discrete && (mn == :uniform || mn == :gaussian) && continue # bad combination
    xn == :line && mn == :gaussian && continue # bad combination
    temp_GA = GA(
        populationSize = 10,
        ɛ = ɛ,
        selection = ss,
        crossover = xovr,
        mutation = ms
        #ɛ = 0.125,
        #selection = ranklinear(1.5),
        #crossover = line(0.25),
        #mutation = gaussian(0.125)
    )
    println()
    println("GA:$(sn):$(xn):$(mn):ɛ=$ɛ) => F before crash...")
    println()

    result = Evolutionary.optimize(zz_, initd, temp_GA,
        Evolutionary.Options(iterations=12, successive_f_tol=25, show_trace=false, store_trace=false)
    )
    #print(result)
    fitness = minimum(result)
    meta_param_dict[:fitness] = Dict()
    meta_param_dict[:fitness][:"sn"] = sn
    meta_param_dict[:fitness][:"xn"] = xn
    meta_param_dict[:fitness][:"mn"] = mn
    meta_param_dict[:fitness][:"ɛ"] = ɛ

    println("GA:$(sn):$(xn):$(mn):ɛ=$ɛ) => F: $(minimum(result))")# C: $(Evolutionary.iterations(result))")

    extremum = Evolutionary.minimizer(result)
    meta_param_dict[:fitness][:"extremum"] = extremum
    save("meta_param_dict.jld", "meta_param_dict", meta_param_dict)#,"vmgtt",vmgtt, "ngt_spikes", ngt_spikes,"gt_spikes",gt_spikes)

    #checkmodel(extremum)
    #plot(vmgtt[:],vmgtv[:]) |> display
end

#=
function plot_pop(pop)
    pop = filter(indiv -> indiv.rank <= 1, pop) #keeps only the non-dominated solutions
    scatter3d(map(x -> x.y[1], pop), map(x -> x.y[2], pop),  map(x -> x.y[3], pop), markersize = 1) |> display
    sleep(0.1)
end


#bc1 = BinaryCoding(2, [:Float64,:Float64], lower[1:2], upper[1:2])
bc = BinaryCoding(4, [:Float64,:Float64,:Float64,:Float64], lower, upper)
=#
#=
function CV(x)
    #sumW = dot(x, w)
    println(x[1])
    println("gets here")
    return x[1]# <= c ? 0 : sumW - c
end

function two_bits_flip!(bits)
    for i = 1:2
        n = rand(1:length(bits))
        bits[n] = 1 - bits[n]
    end
end
function one_point_crossover!(parent_a, parent_b, child_a, child_b)
    n = length(parent_a)
    cut = rand(1:n-1)

    child_a[1:cut] .= parent_a[1:cut]
    child_a[cut+1:n] .= parent_b[cut+1:n]

    child_b[1:cut] .= parent_b[1:cut]
    child_b[cut+1:n] .= parent_a[cut+1:n]
end
function two_point_crossover!(bits_a, bits_b, child1, child2)
    cut_a = cut_b = rand(1:length(bits_a)-1)
    while(cut_b == cut_a)
        cut_b = rand(1:length(bits_a))
    end
    cut_a,cut_b = minmax(cut_b,cut_a)

    copyto!(child1, 1, bits_a, 1, cut_a-1)
    copyto!(child1, cut_a, bits_b, cut_a, cut_b-cut_a+1)
    copyto!(child1, cut_b+1, bits_a, cut_b+1, length(bits_a)-cut_b)

    copyto!(child2, 1, bits_b, 1, cut_a-1)
    copyto!(child2, cut_a, bits_a, cut_a, cut_b-cut_a+1)
    copyto!(child2, cut_b+1, bits_b, cut_b+1, length(bits_a)-cut_b)
end
function crossover!(ind_a, ind_b, child_a, child_b)
    two_point_crossover!(ind_a.x, ind_b.x, child_a.x, child_b.x)
end

#default_mutation!(p::Vector{Int}) = rand_swap!(p)
#default_mutation!(b::T) where T<:AbstractVector{Bool} = rand_flip!(b)

function rand_flip!(bits)
    print(typeof(bits))
    nb = length(bits.x)
    for i = 1:nb
        if rand() < 1/nb
            @inbounds bits[i] = 1 - bits[i]
        end
    end
end
=#
#mutate!(ind::Indiv, fmut!) =
#rand_flip!(ind.x)


#nsga_max(popsize, nbgen, z, init, fCV = CV, fcross = one_point_crossover!)
#nsga_max(popsize, nbgen, z, init, fCV = CV, fplot = plot_pop, plotevery = 5)
#popsize = 10
#nbgen = 10
#seed = initf(10)
#pf = nsga_max(popsize, nbgen, z, bc, pmut = 0.85, plotevery = 5,seed=seed, fcross = crossover!)#, fCV = ConstraintVioloation)#fmut = rand_flip!)#, fplot = plot_pop, plotevery = 1)#, fCV = CV)
#model = NSGAII.decode!(pf[1].x,bc)
#println(model.p)
#checkmodel(model.p)
#init = () -> NSGAII.BinaryCodedIndiv(bitrand(bc.nbbitstotal), zeros(bc.nbvar))
#ind = init()
#re_model = NSGAII.decode!(ind.x,bc)

#indiv -> (decode!(indiv, bc) ; z(indiv.p))
#println(indiv.p)
#_z = indiv -> (decode!(indiv, bc) ; z(indiv.p))


#param = getproperty.(result[1], :x)
#NSGAII.decode!(bincoded, bc)
#println(res)
#=
options = Evolutionary.Options(iterations=10, abstol=1e-5)


#bounds = ConstraintBounds(lower,upper,[],[])

method = GA(populationSize=10, crossoverRate=0.8, mutationRate=0.05, selection=susinv)#, crossover=MILX(), mutation=MIPM(lx,ux))
Random.seed!(10);
#population = Evolutionary.initial_population(method, model.p)

# These could be the same if we could restrict g below not to be an AbstractArray
function NonDifferentiable1(f, x::AbstractArray, F::Real = real(zero(eltype(x))); inplace = true)
    xnans = x_of_nans(x)
    NonDifferentiable1{typeof(F),typeof(xnans)}(f, F, xnans, [0,])
end
function NonDifferentiable1(f, x::AbstractArray, F::AbstractArray; inplace = true)
    f = !inplace && (F isa AbstractArray) ? f!_from_f(f, F, inplace) : f
    xnans = x_of_nans(x)
    NonDifferentiable1{typeof(F),typeof(xnans)}(f, F, xnans, [0,])
end

# this is the g referred to above!
NonDifferentiable1(f, g,        x::AbstractArray, F::Union{AbstractArray, Real} = real(zero(eltype(x)))) = NonDifferentiable1(f, x, F)
NonDifferentiable1(f, g, h,     x::TX, F) where TX  = NonDifferentiable(f, x, F)
NonDifferentiable1(f, g, fg, h, x::TX, F) where TX  = NonDifferentiable(f, x, F)

# Objective function
function rastrigin(x::AbstractVector{T}) where {T <: AbstractFloat}
    n = length(x)
    return 10n + sum([ x[i]^2 - 10cos(convert(T,2π*x[i])) for i in 1:n ])
end

# Parameters
N = 3
P = 100
initState = ()->rand(N)
println(initState)
println("initstate")
println()
=#
#=
#initState = garray[1]
# Testing: (μ/μ_I,λ)-σ-Self-Adaptation-ES
# with non-isotropic mutation operator y' := y + (σ_1 N_1(0, 1), ..., σ_N N_N(0, 1))
m = ES(mu = 15, lambda = P)
m.μ == 15
m.λ == P
result = Evolutionary.optimize( rastrigin,
    initState,
    ES(
        initStrategy = AnisotropicStrategy(N),
        recombination = average, srecombination = average,
        mutation = gaussian, smutation = gaussian,
        selection=:comma,
        μ = 15, λ = P
    ),Evolutionary.Options(iterations=1000, show_trace=false)
)
println("(15,$(P))-σ-SA-ES => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
#test_result(result, N, 1e-1)
print(result)
# Testing: CMA-ES
result = Evolutionary.optimize(rastrigin, initState, CMAES(mu = 15, lambda = P))
println("(15/15,$(P))-CMA-ES => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
#test_result(result, N, 1e-1)

selections = [:roulette=>rouletteinv, :sus=>susinv, :rank=>ranklinear(1.5)]
crossovers = [:discrete=>discrete, :intermediate0=>intermediate(0.), :intermediate0_25=>intermediate(0.5), :line=>line(0.2)]
mutations = [:domrng0_5=>domainrange(fill(0.5,N)), :uniform=>uniform(3.0), :gaussian=>gaussian(0.6)]

for (sn,ss) in selections, (xn,xovr) in crossovers, (mn,ms) in mutations
    xn == :discrete && (mn == :uniform || mn == :gaussian) && continue # bad combination
    xn == :line && mn == :gaussian && continue # bad combination
    result = Evolutionary.optimize( rastrigin, initState,
        GA(
            populationSize = P,
            ɛ = 0.1,
            selection = ss,
            crossover = xovr,
            mutation = ms
        ), Evolutionary.Options(iterations=1000, successive_f_tol=25)
    )
    println(result)
    println("GA:$(sn):$(xn):$(mn)(N=$(N),P=$(P),x=.8,μ=.1,ɛ=0.1) => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
end
# Testing: GA
#m = GA(epsilon = 10.0)
#@test m.ɛ == 10

selections = [:roulette=>rouletteinv, :sus=>susinv, :rank=>ranklinear(1.5)]
crossovers = [:discrete=>discrete, :intermediate0=>intermediate(0.), :intermediate0_25=>intermediate(0.5), :line=>line(0.2)]
mutations = [:domrng0_5=>domainrange(fill(0.5,N)), :uniform=>uniform(3.0), :gaussian=>gaussian(0.6)]

for (sn,ss) in selections, (xn,xovr) in crossovers, (mn,ms) in mutations
    xn == :discrete && (mn == :uniform || mn == :gaussian) && continue # bad combination
    xn == :line && mn == :gaussian && continue # bad combination
    result = Evolutionary.optimize( rastrigin, initState,
        GA(
            populationSize = P,
            ɛ = 0.1,
            selection = ss,
            crossover = xovr,
            mutation = ms
        ), Evolutionary.Options(iterations=1000, successive_f_tol=25)
    )
    println("GA:$(sn):$(xn):$(mn)(N=$(N),P=$(P),x=.8,μ=.1,ɛ=0.1) => F: $(minimum(result)), C: $(Evolutionary.iterations(result))")
end
=#


#=

function initd()
    population = initf(1)
    population
end
=#


#println(garray)
#aa = zero(eltype(first(garray)))
 #println(aa)
#nd = NonDifferentiable1(zz,first(garray))
#println(nd)
#(method::GA, options, objfun, population)
#GA_ = Evolutionary.GA()
#initState = ()->rand(N)
#N = size(seed[1])[1]
#constraints(z) = [sum(z[1:end-1]),z[2]]
#state = Evolutionary.initial_state(GA_, options, z, garray[1])

#Evolutionary.update_state!(z, constraints, state, garray[1], GA, 1)

#selections = [:roulette=>rouletteinv, :sus=>susinv, :rank=>ranklinear(1.5)]
#crossovers = [:discrete=>discrete, :intermediate0=>intermediate(0.), :intermediate0_25=>intermediate(0.5), :line=>line(0.2)]
#mutations = [:domrng0_5=>domainrange(fill(0.5,N)), :uniform=>uniform(3.0), :gaussian=>gaussian(0.6)]


#res2 = Evolutionary.optimize(z, bounds, method, options)


#result = Evolutionary.optimize(z, lower, upper, mthd, opts)
#function optimize(f, lower, upper, method::M,
#                  options::Options = Options(;default_options(method)...)
#                 ) where {M<:AbstractOptimizer}
#    bounds = ConstraintBounds(lower,upper,[],[])
#    optimize(f, bounds, method, options)
#end


@test minimum(result) ≈ 2.0 atol=1e-1
@test Evolutionary.minimizer(result) ≈ [0.5, 1] atol=1e-1

f1(x) = 2*x[1]+x[2]

#lx = [0.0, 0]
#ux = [1.6, 1]

#cons(x) = [1.25-x[1]^2-x[2], x[1]+x[2]]
#lc = [-Inf, -Inf]
#uc = [0.0, 1.6]

tc = [Float64, Float64, Float64, Float64, Float64]
c = MixedTypePenaltyConstraints(PenaltyConstraints(1e3, cb, z), tc)
#c = MixedTypePenaltyConstraints(WorstFitnessConstraints(cb, cons), tc)
#init = ()->Real[rand(Float64), rand(0:1)]

#-----------------------------------------------------------------

f8(z) = sum((z[4:end-1].-1).^2) - log(z[end]+1) + sum((z[i]-i)^2 for i in 1:3)

@test f8(Real[0.2, 1.280624, 1.954483, 1, 0, 0, 1]) ≈ 3.55746 atol=1e-6

cons(z) = [
    sum(z[1:end-1]),
    sum(v->v^2, z[[1,2,3,6]]),
    z[1]+z[4],
    z[2]+z[5],
    z[3]+z[6],
    z[1]+z[7],
    z[2]^2+z[5]^2,
    z[3]^2+z[6]^2,
    z[2]^2+z[6]^2,
]

lx = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ux = [Inf, Inf, Inf, 1.0, 1.0, 1.0, 1.0]

lc = Float64[]
uc = [5.0, 5.5, 1.2, 1.8, 2.5, 1.2, 1.64, 4.25, 4.64]

tc = [Float64, Float64, Float64, Int, Int, Int, Int]

cb = Evolutionary.ConstraintBounds(lx,ux,lc,uc)
# c = MixedTypePenaltyConstraints(PenaltyConstraints(1E, cb, cons), tc)
c = MixedTypePenaltyConstraints(WorstFitnessConstraints(cb, cons), tc)
init = ()->Real[rand(Float64,3); rand(0:1,4)]

opts = Evolutionary.Options(iterations=2000)
mthd = GA(populationSize=300, ɛ=0.05, crossoverRate=0.8, mutationRate=0.01, selection=susinv, crossover=MILX(0.0,0.5,0.3), mutation=MIPM(lx,ux))

Random.seed!(544);
result = Evolutionary.optimize(f8, c, init, mthd, opts)
@test minimum(result) < 4

#=
opts = Evolutionary.Options(iterations=100, show_every=1, show_trace=true, store_trace=true)
#opts = Evolutionary.Options(iterations=500, abstol=1e-5)
#mthd = GA(populationSize=10, crossoverRate=0.8, mutationRate=0.05, selection=susinv, crossover=MILX(), mutation=MIPM(lx,ux))
mthd = GA(populationSize = 120, ɛ = 0.03, crossoverRate=0.8, mutationRate=0.01, selection=rouletteinv)#, crossover=LX(0.0,4.0), mutation = PM(lx,ux,1.0))

tc = [Float32]#, Float64, Float64, Int, Int, Int, Int]

cb = Evolutionary.ConstraintBounds(lx,ux,lc,uc)
# c = MixedTypePenaltyConstraints(PenaltyConstraints(1e2, cb, cons), tc)
#c = MixedTypePenaltyConstraints(PenaltyConstraints(cb, cons), tc)

Random.seed!(10);
result = Evolutionary.optimize(evaluate, population[1], initf, mthd, opts)
#res = Evolutionary.optimize(fitness, model, algo, opts)

println()
println()
println()
println()

println("error $error")
println()
println()

function init_b()
    for i in 0:9
        gene = []
        for (l,u) in zip(lower,upper)
            p1 = rand(l:u, 1)
            append!(gene,p1)
        end
        println(size(gene))
        append!(genes,gene)
        bincoded = NSGAII.encode(gene, bin_code)
        append!(genesb,bincoded)
    end
    #bincoded = NSGAII.encode(gene_out, bc)
    #return bincoded

    genesb
end
#println(size(genes))
#println(size(genesb))



#bincoded = NSGAII.encode(seed, bin_code)
#=
export transdict
function transdict(x)
    bc = BinaryCoding(4, [:Int,:Int,:Int,:Int], py"lower_list", py"upper_list")

    #py"""
    #def map_dict(x):
    #    genes_out = x
    #    genes_out_dic = OrderedDict({k:genes_out[i] for i,k in enumerate(ranges.keys()) })
    #    return genes_out, genes_out_dic
    #"""
    decoded = x

    try
       decoded = NSGAII.decode(x,bc)
    catch
       decoded = x
    end
    _,god_dict = py"map_dict"(decoded)
    #bincoded = NSGAII.encode(genes_out, bc)
    return god_dict#, genes_out_dic
end
=#
##py"""
#lower_list = [v[0] for k,v in ranges.items()]
#upper_list = [v[1] for k,v in ranges.items()]
##"""

#export init_function
function init_function()
    #const
    bc = BinaryCoding(4, [:Int,:Int,:Int,:Int], py"lower_list", py"upper_list")

    py"""
    def pheno_map():

        lower_list = [v[0] for k,v in ranges.items()]
        upper_list = [v[1] for k,v in ranges.items()]
        gene_out = [random.uniform(lower, upper) for lower, upper in zip(lower_list, upper_list)]
        #gene_out_dic = OrderedDict({k:gene_out[i] for i,k in enumerate(ranges.keys()) })
        return gene_out#, gene_out_dic
    """
    gene_out = py"pheno_map"()
    bincoded = NSGAII.encode(gene_out, bc)
    return bincoded
end

function z(x)
    float_coded = NSGAII.decode(x, bc)

    #god = transdict(x)
    #contents = py"z_py"(py"evaluate",god)
    #return contents
end

#=
function references_for_H()
    bc = BinaryCoding(4, [:Int,:Int,:Int,:Int], py"lower_list", py"upper_list")

    py"""
    from sklearn.model_selection import ParameterGrid
    grid = ParameterGrid(ranges)
    genes = []
    for g in grid:
        genes.append(list(g.values()))
    genes_out_dic = OrderedDict({k:genes[i] for i,k in enumerate(ranges.keys()) })

    """
    genes_out = py"genes"
    bcd = []
    for g in genes_out
         bincoded = NSGAII.encode(g, bc)
         append!(bcd,bincoded)
    end
    return bcd
end
=#

export z
function z(x)
    god = transdict(x)
    contents = py"z_py"(py"evaluate",god)
    return contents
end



#bc = BinaryCoding(4, [:Int,:Int,:Int,:Int], py"lower_list", py"upper_list")

module HHNSGA
    #import Pkg;


    include("../src/SpikingNeuralNetworks.jl")
    include("../src/units.jl")
    include("../src/plot.jl")

    SNN = SpikingNeuralNetworks.SNN
    using Random
    using PyCall
    py"""
    import copy
    #import matplotlib
    #import matplotlib.pyplot as plt
    import numpy as np
    #matplotlib.get_backend()
    import sys
    import os
    sys.path.append(os.getcwd())
    """
    #include("../src/SpikingNeuralNetworks.jl")
    #SNN = SpikingNeuralNetworks.SNN")
    #import pickle
    #import numpy as np
    #import random
    #from sciunit.scores.collections import ScoreArray
    #from sciunit import TestSuite
    #import sciunit
    #import random
    #from neuronunit.tests.fi import RheobaseTestP
    #from collections import OrderedDict
    """
    #from neuronunit import get_neab
    #import pdb
    #pdb.set_trace()
    #cell_tests = pickle.load(open('multicellular_constraints.p','rb'))
    for test in cell_tests.values():
        if "Rheobase test" in test.keys():
            temp_test = {k:v for k,v in test.items()}
            break
    rt = temp_test["Rheobase test"]
    #rtp = RheobaseTestP(rt.observation)
    JUIZI = {
        'a': 0.02,
        'b': 0.2,
        'c': -65,
        'd': 8,
    }

    #ranges = OrderedDict({k:[v-0.5*np.abs(v),v+0.5*np.abs(v)] for k,v in copy.copy(JUIZI).items()})

    ranges = OrderedDict(temp)

    N = len(JUIZI)
    """
    #N = py"N"
    #ranges = py"ranges"
    #H1=[values(ranges)]
    #current_params = py"rt.params"
    #simple.attrs = py"JUIZI"
    py"""
    from izhi import IZModel

    def evaluate(test,god):
        model = IZModel(attrs=god)
        model.attrs = god
        rt = test['Rheobase test']

        rheo = rt.generate_prediction(model)

        scores_ = []
        test = {k:v for k,v in test.items() if v.observation is not None}
        for temp_test in test.values():
            if 'injected_square_current' not in temp_test.params.keys():
               temp_test.params['injected_square_current'] = {}
               temp_test.params['amplitude'] = rheo['value']

            if temp_test.name == "InjectedCurrentAPWidthTest":
               temp_test.params['injected_square_current']['amplitude'] = rheo['value']
               temp_test.params['amplitude'] = rheo['value']

            if temp_test.name == "Injected CurrentAPThresholdTest":
               temp_test.params['injected_square_current']['amplitude'] = rheo['value']
               temp_test.params['amplitude'] = rheo['value']

            if temp_test.name == "InjectedCurrentAPAmplitudeTest":
               temp_test.params['injected_square_current']['amplitude'] = rheo['value']
               temp_test.params['amplitude'] = rheo['value']

            if temp_test.name == "RheobaseTest":
               temp_test.params['injected_square_current']['amplitude'] = rheo['value']
               temp_test.params['amplitude'] = rheo['value']

        to_map = [(tt,model) for tt in test.values() ]
        def map_score(content):
            temp_test = content[0]
            model = content[1]
            temp_test.score_type = sciunit.scores.ZScore
            try:
                score = temp_test.judge(model)
                score = np.abs(score.log_norm_score)
            except:
                try:
                    score = np.abs(float(score.raw_score))
                except:
                    score = 100
            if isinstance(score, sciunit.scores.incomplete.InsufficientDataScore):
                score = 100
            return score

        scores_ = list(map(map_score,to_map))
        tt = TestSuite(list(test.values()))
        SA = ScoreArray(tt, scores_)
        errors = tuple(SA.values,)
        return errors
    """

    py"""
    def z_py(evaluate,god):

        cell_tests = pickle.load(open('multicellular_constraints.p','rb'))
        tests = cell_tests['Neocortex pyramidal cell layer 5-6']


        model_type="RAW"
        #from neuronunit.optimisation import make_sim_tests
        fps = ['a','b','c','d']
        #sim_tests, OM, target = make_sim_tests.test_all_objective_test(fps,model_type=model_type)
        SA = evaluate(tests,god)
        return SA
    """


    export transdict
    function transdict(x)
        bc = BinaryCoding(4, [:Int,:Int,:Int,:Int], py"lower_list", py"upper_list")

        #py"""
        #def map_dict(x):
        #    genes_out = x
        #    genes_out_dic = OrderedDict({k:genes_out[i] for i,k in enumerate(ranges.keys()) })
        #    return genes_out, genes_out_dic
        #"""
        decoded = x

        try
           decoded = NSGAII.decode(x,bc)
        catch
           decoded = x
        end
        _,god_dict = py"map_dict"(decoded)
        #bincoded = NSGAII.encode(genes_out, bc)
        return god_dict#, genes_out_dic
    end
    ##py"""
    #lower_list = [v[0] for k,v in ranges.items()]
    #upper_list = [v[1] for k,v in ranges.items()]
    ##"""

    export init_function
    function init_function()
        #const
        bc = BinaryCoding(4, [:Int,:Int,:Int,:Int], py"lower_list", py"upper_list")

        py"""
        def pheno_map():

            lower_list = [v[0] for k,v in ranges.items()]
            upper_list = [v[1] for k,v in ranges.items()]
            gene_out = [random.uniform(lower, upper) for lower, upper in zip(lower_list, upper_list)]
            #gene_out_dic = OrderedDict({k:gene_out[i] for i,k in enumerate(ranges.keys()) })
            return gene_out#, gene_out_dic
        """
        gene_out = py"pheno_map"()
        bincoded = NSGAII.encode(gene_out, bc)
        return bincoded
    end

    function references_for_H()
        bc = BinaryCoding(4, [:Int,:Int,:Int,:Int], py"lower_list", py"upper_list")

        py"""
        from sklearn.model_selection import ParameterGrid
        grid = ParameterGrid(ranges)
        genes = []
        for g in grid:
            genes.append(list(g.values()))
        genes_out_dic = OrderedDict({k:genes[i] for i,k in enumerate(ranges.keys()) })

        """
        genes_out = py"genes"
        bcd = []
        for g in genes_out
             bincoded = NSGAII.encode(g, bc)
             append!(bcd,bincoded)
        end
        return bcd
    end


    export z
    function z(x)
        god = transdict(x)
        contents = py"z_py"(py"evaluate",god)
        return contents
    end

end
#create_sysimage(:HHNSGA, sysimage_path="sys_HHNSGA.so", precompile_execution_file="precompile_HHNSGA.jl")
=#
