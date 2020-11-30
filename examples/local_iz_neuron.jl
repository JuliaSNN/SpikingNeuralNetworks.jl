using Pkg
Pkg.update()
ENV["PYTHON_JL_RUNTIME_PYTHON"] = Sys.which("python")
Pkg.build("PyCall")
module HHNSGA
    using Pkg
    using Plots
    using UnicodePlots
    using OrderedCollections
    using LinearAlgebra
    using UnicodePlots
    using PyCall
    using NSGAII
    export NSGAII

    export SNN


    include("../src/SpikingNeuralNetworks.jl")
    include("../src/units.jl")
    include("../src/plot.jl")

    SNN = SpikingNeuralNetworks.SNN
    #create_sysimage(:SNN, sysimage_path="sys_SpikingNeuralNetworksso", precompile_execution_file="precompile_SpikingNeuralNetworks.jl")

    using Random
    using PyCall
    py"""
    import copy
    from neuronunit.optimisation import algorithms
    from neuronunit.optimisation import optimisations
    from neuronunit.optimisation.optimization_management import TSD
    from neuronunit.optimisation.optimization_management import OptMan
    #from neuronunit.optimisation import make_sim_tests
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    matplotlib.get_backend()
    import sys
    import os
    sys.path.append(os.getcwd())
    """
    py"""
    from neuronunit import tests
    from neuronunit.tests import fi
    from izhi import IZModel
    from neo import AnalogSignal
    import quantities as pqu
    try:
       from julia import Main
    except:
       import julia
       julia.install()
       from julia import Main
    Main.include("../src/SpikingNeuralNetworks.jl")
    Main.eval("SNN = SpikingNeuralNetworks.SNN")


    """
    ##model = py"IZModel()"
    simple = py"IZModel()"

    py"""
    import pickle
    import numpy as np
    import random
    from sciunit.scores.collections import ScoreArray
    from sciunit import TestSuite
    import sciunit
    import random

    from neuronunit.tests.fi import RheobaseTestP
    from collections import OrderedDict
    #from neuronunit import get_neab
    #import pdb
    #pdb.set_trace()
    cell_tests = pickle.load(open('multicellular_constraints.p','rb'))
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
    temp = {'a':[0.02,0.1],'b':[0.2,0.26],'c':[-65,-50],'d':[0.05,8]}
    ranges = OrderedDict(temp)

    N = len(JUIZI)
    """
    N = py"N"
    ranges = py"ranges"
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

        py"""
        def map_dict(x):
            genes_out = x
            genes_out_dic = OrderedDict({k:genes_out[i] for i,k in enumerate(ranges.keys()) })
            return genes_out, genes_out_dic
        """
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
    py"""
    lower_list = [v[0] for k,v in ranges.items()]
    upper_list = [v[1] for k,v in ranges.items()]
    """

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
