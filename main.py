import multiprocessing as mp

#from models.full.model import FullModel as Model
from models.paths.model import PathModel as Model
from models.paths.pcache import PathCacheMaker
from data.parser import CityParser, parse_scenario
from scenario.tools import makescen

mp.set_start_method('fork')

city = CityParser('Vienna')
name, scenario = parse_scenario(city.scenarios[6])

model = Model(scenario=scenario, city = city, time_horizon=7200, time_step=1,
    fl_num=10, fl_size=15.24, C=6, time_window=60, v_cruise=15, v_turn=4.78, 
    v_up=5, v_down=3, a_hoz=3.5, yaw_r=55, max_flight_time=1800, overlap=True,
    scen_name=name, num_cpus=4, seed=42, force_path_gen = False
    )

# Output the MPS file
# model.outputmps()

# Set model parameters
model.problem.model.setParam('Threads', 4)
model.problem.model.setParam('MIPGap', 1e-3)
model.problem.model.setParam('Method', 1)
model.problem.model.setParam('SolutionLimit', 3)
model.problem.model.setParam('TimeLimit', 3600)
model.problem.model.setParam('Heuristics', 1)

# Solve it
model.solve()

# Create the scenario
print('> Creating scn file...')
makescen(model)
print('\n################### Done! ###################\n\n')