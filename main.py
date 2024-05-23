#from models.full.model import FullModel as Model
from models.paths.model import PathModel as Model
from models.paths.pcache import PathCacheMaker
from data.parser import CityParser, parse_scenario
import multiprocessing as mp

mp.set_start_method('fork')

city = CityParser('Vienna')
#scenario = parse_scenario(city.scenarios[0])
#small_scenario = {'D1':scenario['D1'], 'D2':scenario['D2']}

# Create the cache
cachemaker = PathCacheMaker(
    G=city.G,
    nodes=city.nodes,
    edges=city.edges,
    v_cruise=15,
    v_turn=5,
    v_up=5,
    v_down=3,
    a_hoz=3,
    yaw_r=55,
    num_cpus=15,
    seed=42
    )

cachemaker.create_cache(list(zip([parse_scenario(scen) 
                            for scen in city.scenarios], city.scenario_names)))



# # Create the model
# model = Model(scenario=small_scenario,
#     G=city.G,
#     nodes=city.nodes,
#     edges=city.edges,
#     time_horizon=7200,
#     time_step=1,
#     fl_num=10,
#     fl_size=20,
#     C=2,
#     time_window=60,
#     v_cruise=15,
#     v_turn=5,
#     v_up=5,
#     v_down=3,
#     a_hoz=3,
#     yaw_r=55,
#     max_flight_time=1800,
#     overlap=True,
#     scen_name='test',
#     num_cpus=2,
#     seed=42,
#     force_path_gen = False
#     )

# # Solve it
# print(model.problem.solve())