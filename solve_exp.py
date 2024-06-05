import multiprocessing as mp

from data.parser import CityParser, parse_scenario
from scenario.tools import makescen
from models.nodesrel.model import NodeRelModel as Model

mp.set_start_method('fork')

city = CityParser('Vienna')
tw_list = [20,40,60]

scen_idx = city.get_scenario_names(None).index('Flight_intention_120_1')
name, scenario = parse_scenario(city.scenarios[scen_idx])

model = Model(scenario=scenario, 
              city = city, 
              time_horizon=7200, 
              time_step=1,
              fl_num=10, 
              fl_size=15.24, 
              C=1, 
              time_window=40, 
              v_cruise=15, 
              v_turn=4.78, 
              v_up=5, 
              v_down=3, 
              a_hoz=3, 
              yaw_r=55, 
              max_flight_time=1800, 
              overlap=True,
              scen_name=name, 
              num_cpus=4, 
              seed=42, 
              force_path_gen = False
              )

# Output the MPS file
# model.outputmps()

# Set model parameters
hours = 1
model.problem.model.setParam('Threads', 4)
model.problem.model.setParam('MIPGap', 1e-3)
model.problem.model.setParam('Method', 1)
#model.problem.model.setParam('SolutionLimit', 3)
model.problem.model.setParam('Presolve', 2)
model.problem.model.setParam('MIPFocus', 1)
#model.problem.model.setParam('NoRelHeurTime', 3600*hours/2)
#model.problem.model.setParam('Heuristics', 1)
model.problem.model.setParam('TimeLimit', 3600*hours)

# Solve it
model.solve()

# Create the scenario
print('> Creating scn file...')
makescen(model)
print('\n################### Done! ###################\n\n')