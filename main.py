#from models.full.model import FullModel as Model
from models.paths.model import PathModel as Model
from data.parser import CityParser, parse_scenario

city = CityParser('Vienna')
# Do first scenario
scenario = parse_scenario(city.scenarios[0])

# Create the model
model = Model(scenario=scenario,
    G=city.G,
    nodes=city.nodes,
    edges=city.edges,
    time_horizon=7200,
    time_step=1,
    fl_num=10,
    fl_size=20,
    C=2,
    time_window=60,
    v_cruise=15,
    v_turn=5,
    v_up=5,
    v_down=3,
    a_hoz=3,
    yaw_r=55,
    max_flight_time=1800,
    overlap=True,
    scen_name='test2',
    num_cpus=8,
    seed=42,
    force_path_gen = True
    )

# Solve it
#print(model.problem.solve())