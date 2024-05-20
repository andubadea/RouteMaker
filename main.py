import geopandas as gpd
import osmnx as ox
from models.full.params import Parameters as FullParameters
from models.full.problem import ProblemGlobal as FullProblem
from data.parser import CityParser, parse_scenario

city = CityParser('Vienna')
# Do first scenario
scenario = parse_scenario(city.scenarios[0])

# Let's try to solve its

# Test parameters
test_params = FullParameters(
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

# Create the problem
problem = FullProblem(test_params, 'Test')
# Solve it
print(problem.solve())