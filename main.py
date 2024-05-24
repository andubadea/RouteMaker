#from models.full.model import FullModel as Model
from models.paths.model import PathModel as Model
from models.paths.pcache import PathCacheMaker
from data.parser import CityParser, parse_scenario
import multiprocessing as mp

mp.set_start_method('fork')

city = CityParser('Vienna')
scenario = parse_scenario(city.scenarios[0], 10800)
#small_scenario = dict([[acid, scenario[acid]] 
#                       for acid in list(scenario.keys())[:10]])

# # Create the cache
# cachemaker = PathCacheMaker(
#     G=city.G,
#     nodes=city.nodes,
#     edges=city.edges,
#     v_cruise=15,
#     v_turn=5,
#     v_up=5,
#     v_down=3,
#     a_hoz=3,
#     yaw_r=55,
#     num_cpus=15,
#     seed=42
#     )

# cachemaker.create_cache(list(zip([parse_scenario(scen) 
#                             for scen in city.scenarios], city.scenario_names)))



# Create the model
model = Model(scenario=scenario,
    G=city.G,
    nodes=city.nodes,
    edges=city.edges,
    time_horizon=7200,
    time_step=1,
    fl_num=10,
    fl_size=20,
    C=5,
    time_window=60,
    v_cruise=15,
    v_turn=5,
    v_up=5,
    v_down=3,
    a_hoz=3,
    yaw_r=55,
    max_flight_time=1800,
    overlap=True,
    scen_name='test',
    num_cpus=16,
    seed=42,
    force_path_gen = True
    )

# Solve it
model.solve()