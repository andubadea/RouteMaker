import multiprocessing as mp
from data.parser import CityParser, parse_scenario
from scenario.tools import makescen
# This model is used to find a heuristic solution
from models.heuristic.model import HeuristicModel as HModel
# This model is used to find the best bound after a warm start
from models.bound.model import BoundModel as BModel

mp.set_start_method('fork')

city = CityParser('Vienna')
scen_idx = city.get_scenario_names(None).index('Flight_intention_120_1')
name, scenario = parse_scenario(city.scenarios[scen_idx])

param_dict = {'scenario' : scenario, 
        'city' : city, 
        'time_horizon' : 7200, 
        'time_step' : 1,
        'fl_num' : 10, 
        'fl_size' : 15.24, 
        'C' : 1, 
        'time_window' : 20, 
        'v_cruise' : 15, 
        'v_turn' : 4.78, 
        'v_up' : 5, 
        'v_down' : 3, 
        'a_hoz' : 3, 
        'yaw_r' : 55, 
        'num_paths' : 5, 
        'overlap' : True,
        'scen_name' : name, 
        'num_cpus' : 4, 
        'seed' : 42, 
        'force_path_gen' : False   
}

# First, run the heuristic model with the NoRel heuristic
print(f'################### {name} ###################')
print('\n******* Heuristic model *******\n')
hmodel = HModel(param_dict)

# Output the MPS file of the heuristic model
hmodel.outputmps()

print('> Setting heuristic model parameters...')
hmodel.problem.model.setParam('Threads', 8)
hmodel.problem.model.setParam('Method', 1)
hmodel.problem.model.setParam('Presolve', 2)
hmodel.problem.model.setParam('NoRelHeurTime', 3600*1)
# hmodel.problem.model.setParam('Heuristics', 1)
hmodel.problem.model.setParam('TimeLimit', 3600*24)

hmodel.solve()

# Create the hmodel scenario
print('> Creating heuristic model scn file...')
makescen(hmodel)
quit()

# Then, run the bound model with the heuristic solution as a hot start
print('\n******* Bound model *******\n')
bmodel = BModel(param_dict, hmodel)

# Output the MPS file of the bound model
# bmodel.outputmps()

# Set global problem parameters for the bound model
print('> Setting bound model parameters...')
bmodel.problem.model.setParam('Threads', 16)
bmodel.problem.model.setParam('MIPGap', 1e-3)
bmodel.problem.model.setParam('Method', 1)
bmodel.problem.model.setParam('Presolve', 2)
bmodel.problem.model.setParam('MIPFocus', 2)
bmodel.problem.model.setParam('TimeLimit', 3600*8)

# Solve it
bmodel.solve()

# Create the bmodel scenario
print('> Creating bound model scn file...')
makescen(bmodel)
print('\n################### Done! ###################\n\n')