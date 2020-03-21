

module HHNSGA
    using Pkg
    # ENV["PYTHON"]="/home/user/anaconda3/bin/python"
    # Pkg.build("PyCall")
    Pkg.add("UnicodePlots")
    #using Distributed
    try
        using Plots
        using UnicodePlots
        using OrderedCollections
    catch
        import Pkg; Pkg.add("Plots")
        Pkg.add("OrderedCollections")
        import Pkg; Pkg.add("UnicodePlots")
        using OrderedCollections

    end
    import Pkg; Pkg.add("ProgressMeter")
    using NSGAIII
    export NSGAIII
    export nS
    using Pkg
    Pkg.resolve()

    try
        using LinearAlgebra
        using UnicodePlots
        using PyCall
        using NSGAIII
    catch
        using Pkg
        Pkg.add("PyCall")
        Pkg.add("UnicodePlots")

        Pkg.build("PyCall")
        Pkg.clone("https://github.com/gsoleilhac/NSGAIII.jl")
        using PyCall
        using UnicodePlots
    end
    include("../src/SpikingNeuralNetworks.jl")
    include("../src/units.jl")
    include("../src/plot.jl")
    SNN = SpikingNeuralNetworks.SNN
    export SNN

    using Random
    py"""
    import copy
    from neuronunit.optimisation import algorithms
    from neuronunit.optimisation import optimisations

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
    from neuronunit.tests.fi import RheobaseTestP
    from collections import OrderedDict
    cell_tests = pickle.load(open('multicellular_constraints.p','rb'))
    for test in cell_tests.values():
        if "Rheobase test" in test.keys():
            temp_test = {k:v for k,v in test.items()}
            break
    rt = temp_test["Rheobase test"]

    rtest = RheobaseTestP(observation=rt.observation)
    #                        name='RheobaseTest')
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
    #JHH1 = {k:(v-0.01*v) for k,v in JHH.items()}
    #JHH2 = {k:(v+0.01*v) for k,v in JHH.items()}
    ranges = OrderedDict({k:[v-0.01*np.abs(v),v+0.01*np.abs(v)] for k,v in copy.copy(JHH).items()})
    N = len(JHH)
    model = SimpleModel()
    """
    N = py"N"

    #
    ranges = py"ranges"
    H1=[values(ranges)]

    current_params = py"rt.params"
    print(current_params)
    simple.attrs = py"JHH"

    py"""
    from neuronunit.optimisation import exhaustive_search
    from sklearn.model_selection import ParameterGrid

    grid = ParameterGrid(ranges)
    genes = []
    for g in grid: genes.append(g.values)
    #dic_grid = es.create_grid(mp_in =ranges,npoints = 10, free_params =ranges.keys())

    """
    py"""
    def evaluate(test,god):
        model = SimpleModel()
        model.attrs = god
        rt = test['Rheobase test']

        rheo = rt.generate_prediction(model)
        from sciunit.scores.collections import ScoreArray
        from sciunit import TestSuite
        import sciunit
        scores_ = []

        for key, temp_test in test.items():
            if key == 'Rheobase test':
                temp_test.score_type = sciunit.scores.ZScore
            if key in ['Injected current AP width test', 'Injected current AP threshold test', 'Injected current AP amplitude test', 'Rheobase test']:
                rt.params['amplitude'] = rheo['value']

            try:
                score = temp_test.judge(model)
                score = np.abs(score.log_norm_score)

            except:
                score = 100#np.inf
            if isinstance(score, sciunit.scores.incomplete.InsufficientDataScore):
                score = 100#np.inf
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
                import random
                lower_list = [v[0] for k,v in ranges.items()]
                upper_list = [v[1] for k,v in ranges.items()]
                genes_out = [random.uniform(lower, upper) for lower, upper in zip(lower_list, upper_list)]
                genes_out_dic = OrderedDict({k:genes_out[i] for i,k in enumerate(ranges.keys()) })
            return genes_out, genes_out_dic
        """
        genes_out, genes_out_dic = py"one"(x)
        return genes_out, genes_out_dic#py"genes_out",py"genes_out_dic"
    end

    export z
    function z(x)
        genes_out,god = init_function2(x)
        x = genes_out
        # The slow part can it be done in parallel.
        contents = py"z_py"(py"evaluate",god)
        contents = [ convert(Float64,c) for c in contents ]
        return contents
    end

end
