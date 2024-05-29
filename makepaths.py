import multiprocessing as mp

from data.parser import CityParser, parse_scenario
from scenario.tools import makescen

from models.paths.pcache import PathMaker


mp.set_start_method('fork')

city = CityParser('Vienna')
name, scenario = parse_scenario(city.scenarios[6])

#Generate all cache files
for scen in city.scenarios:
    name, scenario = parse_scenario(scen)
    print(f'------ {name} ------')
    
    # Generate the cache files but don't solve the model
    _ = PathMaker(scenario=scenario,
        G=city.G,
        nodes=city.nodes,
        edges=city.edges,
        city = city,
        time_horizon=7200,
        time_step=1,
        fl_num=10,
        fl_size=20,
        C=6,
        time_window=60,
        v_cruise=15,
        v_turn=5,
        v_up=5,
        v_down=3,
        a_hoz=3,
        yaw_r=55,
        max_flight_time=1800,
        overlap=True,
        scen_name=name,
        num_cpus=16,
        seed=42,
        force_path_gen = True
        )