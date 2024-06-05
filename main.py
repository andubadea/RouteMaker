import multiprocessing as mp

from data.parser import CityParser, parse_scenario
from scenario.tools import makescen
altonly = False

# This is the model that uses edge flows
#from models.edges.model import EdgeModel as Model

# This is the model that uses node flows
# from models.nodes.model import NodeModel as Model

# This is the model that uses node flows and relaxed constraints
from models.nodesrel.model import NodeRelModel as Model

mp.set_start_method('fork')

city = CityParser('Vienna')
scen_idx = city.get_scenario_names(None).index('Flight_intention_30_1')
name, scenario = parse_scenario(city.scenarios[scen_idx])


model = Model(scenario=scenario, 
            city = city, 
            time_horizon=7200, 
            time_step=1,
            fl_num=10, 
            fl_size=15.24, 
            C=1, 
            time_window=20, 
            v_cruise=15, 
            v_turn=4.78, 
            v_up=5, 
            v_down=3, 
            a_hoz=3, 
            yaw_r=55, 
            num_paths=5, 
            overlap=True,
            scen_name=name, 
            num_cpus=4, 
            seed=42, 
            force_path_gen = False
            )

# Output the MPS file
# model.outputmps()

# Set global problem parameters
print('> Setting global problem parameters...')
model.problem.model.setParam('Threads', 4)
model.problem.model.setParam('MIPGap', 1e-3)
model.problem.model.setParam('Method', 1)
model.problem.model.setParam('Presolve', 2)
model.problem.model.setParam('MIPFocus', 1)
#model.problem.model.setParam('NoRelHeurTime', 600)
#model.problem.model.setParam('Heuristics', 1)
model.problem.model.setParam('TimeLimit', 1200)

# Solve it
model.solve()

# Create the scenario
print('> Creating scn file...')
makescen(model, altonly = altonly)
print('\n################### Done! ###################\n\n')