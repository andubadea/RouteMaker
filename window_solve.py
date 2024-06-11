import multiprocessing as mp

from data.parser import CityParser, parse_scenario
from scenario.tools import makescen
from models.window.model import WindowModel as WModel

mp.set_start_method('fork')

city = CityParser('Vienna')

scen_idx = city.get_scenario_names(None).index('Flight_intention_30_1')
name, scenario = parse_scenario(city.scenarios[scen_idx])
print(f'################### {name} ###################')

param_dict = {'scenario' : scenario, 
        'city' : city, 
        'time_horizon' : 7200, 
        'planning_time_step' : 1800,
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

wmodel = WModel(param_dict)
wmodel.solve()
makescen(wmodel)