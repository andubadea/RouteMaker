import geopandas as gpd
import osmnx as ox
from models.full.params import Parameters
from data.parser import CityParser, parse_scenario

city = CityParser('Vienna')
# Do first scenario
scenario = parse_scenario(city.scenarios[0])

# Test parameters
test = Parameters(
    scenario,
    city.G,
    city.nodes,
    city.edges,
    time_horizon=7200,
    time_step=1,
    fl_num=10,
    fl_size=20,
    C = 2,
    time_window=600,
    v_cruise=15,
    v_turn=5,
    v_up = 5,
    v_down = 3,
    a_hoz = 3,
    max_flight_time=1800,
    overlap=False
)