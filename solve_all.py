import sys
import multiprocessing as mp

from data.parser import CityParser, parse_scenario
from scenario.tools import makescen
from models.window.model import WindowModel as WModel

mp.set_start_method('fork')

city = CityParser('Vienna')
if len(sys.argv) > 1 and sys.argv[1] == '-h':
    heuristic = True
else:
    heuristic = False

tws = [5,10,20,30]

for tw in tws:
    print(f'@@@@@@@@@@@@@@@@@ Flow time window: {tw}s @@@@@@@@@@@@@@@@@')
    for scenfile in city.scenarios:
        name, scenario = parse_scenario(scenfile)
        if '_60_' not in name:
            continue
        print(f'################### {name} ###################')

        if heuristic:
            model_params = [
                ('Threads', 8),
                ('Method', 1),
                ('NoRelHeurTime', 3600),
                ('TimeLimit', 3600),
                ('Presolve', 2),
                ('MIPGap', 1e-3),
                ('MIPFocus', 2),
            ]

            param_dict = {'scenario' : scenario, 
                'city' : city, 
                'time_horizon' : 7200, 
                'min_time' : 1200,
                'planning_time_step' : 2100,
                'planning_overlap' : 300,
                'fl_num' : 10, 
                'fl_size' : 15.24, 
                'C' : 1, 
                'time_window' : tw, 
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
                'force_path_gen' : False,
                'model_params' : model_params
            }
        else:
            model_params = [
                ('Threads', 8),
                ('Method', 1),
                ('NoRelHeurTime', 3600*2),
                ('TimeLimit', 3600*48),
                ('Presolve', 2),
                ('MIPGap', 1e-3),
                ('MIPFocus', 2),
            ]

            param_dict = {'scenario' : scenario, 
                'city' : city, 
                'time_horizon' : 7200, 
                'min_time' : None,
                'planning_time_step' : 7200,
                'planning_overlap' : 7200,
                'fl_num' : 10, 
                'fl_size' : 15.24, 
                'C' : 1, 
                'time_window' : tw, 
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
                'force_path_gen' : False,
                'model_params' : model_params
            }
            
        wmodel = WModel(param_dict)
        wmodel.solve()
        print('> Creating scn file...')
        makescen(wmodel)
        print('\n################### Done! ###################\n\n')   
    print('################### Double done! ###################')
print('################### Triple done! ###################')