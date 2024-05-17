import geopandas as gpd
import osmnx as ox
from models.full.params import Parameters

city = 'Vienna'

nodes = gpd.read_file(f'data/cities/{city}/streets.gpkg', layer='nodes')
edges = gpd.read_file(f'data/cities/{city}/streets.gpkg', layer='edges')

# set the indices 
edges.set_index(['u', 'v', 'key'], inplace=True)
nodes.set_index(['osmid'], inplace=True)

# ensure that it has the correct value
nodes['x'] = nodes['geometry'].apply(lambda x: x.x)
nodes['y'] = nodes['geometry'].apply(lambda x: x.y)

G = ox.graph_from_gdfs(nodes, edges)

with open('data/scenarios/Flight_intention_120_1.txt') as f:
    scen_lines = f.readlines()

scenario = {}
for line in scen_lines:
    s = line.split(';')
    dep_time = int(s[2].split(':')[0]) * 3600 + int(s[2].split(':')[1]) * 60 +\
        int(s[2].split(':')[0])
    origin = int(s[3])
    destination = int(s[4])
    scenario[s[0]] = [dep_time, origin, destination]

test = Parameters(
    scenario,
    G,
    nodes,
    edges,
    time_horizon=7200,
    time_step=1,
    fl_num=10,
    fl_size=20,
    C = 2,
    time_window=60,
    v_cruise=15,
    v_turn=5,
    v_up = 5,
    v_down = 3,
    a_hoz = 3,
    max_flight_time=1800,
    overlap=False
)