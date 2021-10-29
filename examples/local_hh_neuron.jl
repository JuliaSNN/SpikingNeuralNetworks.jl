
module HHNSGA
    using Pkg
    try
        using Plots
        using UnicodePlots
        using OrderedCollections
        using LinearAlgebra
        using UnicodePlots
        using PyCall
        using NSGAII
        export NSGAII
        #export nS

    catch
        include("install.jl")
    end
    export SNN


    include("../src/SpikingNeuralNetworks.jl")
    include("../src/units.jl")
    include("../src/plot.jl")
    SNN = SpikingNeuralNetworks.SNN
    using Random
    py"""
    import copy
    from neuronunit.optimisation import algorithms
    from neuronunit.optimisation import optimisations
    from neuronunit.optimisation.optimization_management import TSD
    from neuronunit.optimisation.optimization_management import OptMan
    import joblib

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
    from simple_with_injection import SimpleModel
    from neo import AnalogSignal
    import quantities as pq
    try:
       from julia import Main
    except:
       import julia
       julia.install()
       from julia import Main


    """
    simple = py"SimpleModel()"
    py"""
    import pickle
    import numpy as np
    import random
    from sciunit.scores.collections import ScoreArray
    from sciunit import TestSuite
    import sciunit

    from neuronunit.tests.fi import RheobaseTestP
    from collections import OrderedDict
    cell_tests = pickle.load(open('multicellular_constraints.p','rb'))
    for test in cell_tests.values():
        if "Rheobase test" in test.keys():
            temp_test = {k:v for k,v in test.items()}
            break
    rt = temp_test["Rheobase test"]
    #rtp = RheobaseTestP(rt.observation)
    JHH = {
    'Vr': -68.9346,
    'Cm': 0.0002,
    'gl': 1.0 * 1e-5,
    'El': -65.0,
    'EK': -90.0,
    'ENa': 50.0,
    'gNa': 0.02,
    'gK': 0.006,
    'Vt': -63.0
    }

    ranges = OrderedDict({k:[v-0.5*np.abs(v),v+0.5*np.abs(v)] for k,v in copy.copy(JHH).items()})
    N = len(JHH)
    model = SimpleModel()
    """
    N = py"N"
    ranges = py"ranges"
    H1=[values(ranges)]
    current_params = py"rt.params"
    simple.attrs = py"JHH"
    #using Py2Jl
    py"""

    def evaluate(test,god):
        model = SimpleModel()
        model.attrs = god
        rt = test['Rheobase test']
        #rtp = RheobaseTestP(rt.observation)

        rheo = rt.generate_prediction(model)
        #    errors = tuple([100.0,100.0,100.0,100.0,100.0,100.0,100.0,100.0])
        #    return errors
        scores_ = []
        #assert 'value' in rheo.keys()

        # for temp_test in test.values():
        #     #print(temp_test.name,temp_test.params)
        #     #print(temp_test.params['injected_square_current']['amplitude'])
        #     #if 'InjectedCurrentAPWidthTest' not in temp_test.params.keys():
        #     #   temp_test.params['injected_square_current'] = {}
        #     if temp_test.name in "InjectedCurrentAPWidthTest":
        #        temp_test.params['injected_square_current']['amplitude'] = rheo['value']
        #     if temp_test.name in "InjectedCurrentAPThresholdTest":
        #        temp_test.params['injected_square_current']['amplitude'] = rheo['value']
        #     if temp_test.name in "InjectedCurrentAPThresholdTest":
        #        temp_test.params['injected_square_current']['amplitude'] = rheo['value']
        #     if temp_test.name in "RheobaseTest":
        #        temp_test.params['injected_square_current']['amplitude'] = rheo['value']

        for key, temp_test in test.items():
            #if key == 'Rheobase test':
            # def to_map(key_val):
            temp_test.score_type = sciunit.scores.ZScore
            if key in ['Injected current AP width test', 'Injected current AP threshold test', 'Injected current AP amplitude test', 'Rheobase test']:
                temp_test.params['amplitude'] = rheo['value']


            try:
                score = temp_test.judge(model)
                score = np.abs(score.log_norm_score)

            except:
                try:
                    score = np.abs(float(score.raw_score))
                except:
                    score = 100#np.inf
            if isinstance(score, sciunit.scores.incomplete.InsufficientDataScore):
                score = 100#np.inf
                #if score == 100:
                print(temp_test.name)
                print(temp_test.params)
                if not rheo['value'] is None:

                    import pdb
                    pdb.set_trace()
            scores_.append(score)

        tt = TestSuite(list(test.values()))
        SA = ScoreArray(tt, scores_)
        errors = tuple(SA.values,)


        return errors
    """



    py"""
    def z_py(evaluate,god):
        cell_tests = pickle.load(open('multicellular_constraints.p','rb'))
        tests = cell_tests[list(cell_tests.keys())[0]]
        SA = evaluate(tests,god)
        return SA
    """


    function references_for_H()
        py"""
        grid = ParameterGrid(ranges)
        genes = []
        for g in grid:
            genes.append(list(g.values()))
        """
        genes = py"genes"
        return genes
    end

    export init_function2
    function init_function2(x)
        py"""
        def one(x):
            if str('x') in locals():
                genes_out = x
                genes_out_dic = OrderedDict({k:genes_out[i] for i,k in enumerate(ranges.keys()) })
            else:
                lower_list = [v[0] for k,v in ranges.items()]
                upper_list = [v[1] for k,v in ranges.items()]
                genes_out = [random.uniform(lower, upper) for lower, upper in zip(lower_list, upper_list)]
                genes_out_dic = OrderedDict({k:genes_out[i] for i,k in enumerate(ranges.keys()) })

            return genes_out, genes_out_dic
        """
        decoded = x

        try
           decoded = NSGAII.decode(x, HHNSGA.bc)
        catch
           decoded = x
           println("no")
        end
        genes_out, genes_out_dic = py"one"(decoded)
        #@show(genes_out)

        bincoded = NSGAII.encode(genes_out, bc)
        #@show(bincoded)
        return bincoded, genes_out_dic#py"genes_out",py"genes_out_dic"
    end
    py"""
    lower_list = [v[0] for k,v in ranges.items()]
    upper_list = [v[1] for k,v in ranges.items()]
    #print(len(lower_list))
    """

    const bc = BinaryCoding(9, [:Int,:Int,:Int,:Int,:Int,:Int,:Int,:Int,:Int], py"lower_list", py"upper_list")
    export init_function
    function init_function()
        py"""
        def one():
            import random
            lower_list = [v[0] for k,v in ranges.items()]
            upper_list = [v[1] for k,v in ranges.items()]
            genes_out = [random.uniform(lower, upper) for lower, upper in zip(lower_list, upper_list)]
            genes_out_dic = OrderedDict({k:genes_out[i] for i,k in enumerate(ranges.keys()) })
            return genes_out, genes_out_dic
        """
        genes_out, genes_out_dic = py"one"()
        #@show(genes_out)
        bincoded = NSGAII.encode(genes_out, bc)
        #@show(bincoded)
        return bincoded
    end

    export z
    function z(x)
        genes_out,god = init_function2(x)
        #x = genes_out
        # The slow part can it be done in parallel.
        contents = py"z_py"(py"evaluate",god)
        @show(contents)
        println("gene fitness calculated")
        return contents
    end

end
