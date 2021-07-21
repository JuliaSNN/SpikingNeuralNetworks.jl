#using Pkg
#Pkg.update()
ENV["PYTHON_JL_RUNTIME_PYTHON"] = Sys.which("python")
using Pkg
using PyCall
using OrderedCollections
using LinearAlgebra
using UnicodePlots
using NSGAII
#export NSGAII
#export SNN
using SpikeSynchrony
include("../src/SpikingNeuralNetworks.jl")
include("../src/units.jl")
SNN = SpikingNeuralNetworks
using Evolutionary, Test, Random
using Plots
import DataStructures
using Plots
using UnicodePlots
unicodeplots()
using JLD


#py"specimen_id" = specimen_id
#varinfo(py)
if isfile("ground_truth.jld")
    vmgtv = load("ground_truth.jld","vmgtv")
    gt_spikes = load("ground_truth.jld","gt_spikes")
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
    specimen_id = options[0]
    target_num_spikes=10
    from neuronunit.allenapi import make_allen_tests_from_id
    sweep_numbers, data_set, sweeps = make_allen_tests_from_id.allen_id_to_sweeps(specimen_id)
    (vmm,stimulus,sn,spike_times) = make_allen_tests_from_id.get_model_parts_sweep_from_spk_cnt(
        target_num_spikes, data_set, sweep_numbers, specimen_id
    )
    """
    gt_spikes = py"spike_times"
    vmgtv = py"vmm.magnitude"
    vmgtt = py"vmm.times"
    save("ground_truth.jld", "vmgtv", vmgtv, "gt_spikes", gt_spikes)
end

#plot(Plots.fakedata(50,5),w=3)
#SNN.custom_vm(vmgtv)
#vmgt |> display
ALLEN_DURATION = 2000 * ms
ALLEN_DELAY = 1000 * ms

function get_ranges()
    ranges = DataStructures.OrderedDict{Char,Float32}()
    ranges = ("a"=>[0.02,0.1],"b"=>[0.2,0.26],"c"=>[-65,-50],"d"=>[0.05,8],"I"=>[700,6000])
    lower = []
    upper = []
    for (k,v) in ranges
        append!(lower,v[1])
        append!(upper,v[2])
    end
    lower,upper
end
#bin_code = BinaryCoding(4, [:Int,:Int,:Int,:Int,:Int],lower,upper)
#genes = []

function init_b(lower,upper)
    gene = []
    for (l,u) in zip(lower,upper)
        p1 = rand(l:u, 1)
        append!(gene,p1)
    end
    gene
end

function initf(n)#lower,upper)
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
function loss(E2,ground_spikes)
    spikes = raster_synchp(E2)
    maxt = findmax(sort!(unique(vcat(spikes,ground_spikes))))[1]
    if size(spikes)[1]>1
        t, S = SPIKE_distance_profile(spikes, ground_spikes;t0=0,tf = maxt)
        spkd = SpikeSynchrony.trapezoid_integral(t, S)/(t[end]-t[1]) # == SPIKE_distance(y1, y2)
    else
        spkd = 10.0
    end
    delta = size(spikes)[1] - size(ground_spikes)[1]
    #spkd
    println(size(spikes)[1] , size(ground_spikes)[1])

    #spkd = 1.0-spkd
    err = 1/delta
    println(err)
    return err

end

function z(param)
    #param = x
    #println(param)
    #bincoded = NSGAII.encode(seed, bc)
    #NSGAII.decode!(bincoded, bc)
    #bc = BinaryCoding(6, [:Float32,:Float32,:Float32,:Float32,:Float32], lower, upper)

    #bc = BinaryCoding(size(lower)[1], lower, upper)
    println(param)
    CH = SNN.IZParameter(;a = param[1], b = param[2], c = param[3], d = param[4])
    E2 = SNN.IZ(;N = 1, param = CH)
    SNN.monitor(E2, [:v])
    SNN.monitor(E2, [:fire])
    ALLEN_DURATION = 2000 * ms
    ALLEN_DELAY = 1000 * ms
    #T = (ALLEN_DURATION+ALLEN_DELAY)*ms
    E2.I = [param[5]*nA]
    #println("fail")
    SNN.sim!([E2], []; dt =1*ms, delay=ALLEN_DELAY,stimulus_duration=ALLEN_DURATION,simulation_duration = ALLEN_DURATION+ALLEN_DELAY+500ms)
    v = SNN.getrecord(E2, :v)
    SNN.vecplot(E2, :v) |> display
    error = loss(E2,gt_spikes)
    #println(typeof(error))
    #println("gets error: ")

    #println(typeof(error))

    error
end
lower,upper = get_ranges()
#population = initf(lower,upper)
#fitnesses = []
#for chromo in population
#    res = evaluate(chromo,gt_spikes)#
#    append!(fitnesses,res)
#    println(res)
#end
#using LinearAlgebra: dot

function CV(x)
    #sumW = dot(x, w)
    println(x[1])
    println("gets here")
    return x[1]# <= c ? 0 : sumW - c
end
#const bc = BinaryCoding(size(lower)[1], lower, upper)
#x = NSGAII.decode(bitrand(bc.nbbitstotal), bc)
const bc = BinaryCoding(5, [:Float32,:Float32,:Float32,:Float32,:Float32], lower, upper)

function plot_pop(P)
    clf() #clears the figure
    P = filter(indiv -> indiv.rank == 1, P) #keep only the non-dominated solutions
    plot(map(x -> x.y[1], P), map(x -> x.y[2], P), "bo", markersize = 1)
    sleep(0.1)
end
popsize = 20
nbgen = 20

#nsga_max(popsize, nbgen, z, init, fCV = CV, fplot = plot_pop, plotevery = 5)
result = nsga_max(popsize, nbgen, z, bc)#, fplot = plot_pop, plotevery = 1)#, fCV = CV)
println(result)
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
