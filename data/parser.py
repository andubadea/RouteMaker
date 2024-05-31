import osmnx as ox
import geopandas as gpd
import os
import networkx as nx
from typing import Tuple

class CityParser():
    """Load the city and parse its scenarios.
    """
    def __init__(self, city) -> None:
        self.name = city
        self.G, self.nodes, self.edges = self.load_city(city)
        self.scenarios = self.get_scenario_list(city)
        self.scenario_names = self.get_scenario_names(city)
    
    def load_city(self, city:str) -> Tuple[nx.MultiDiGraph, 
                                        gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Load the specified city.

        Args:
            city (str): 
                The city name.

        Returns:
            Tuple[nx.MultiDiGraph, gpd.GeoDataFrame, gpd.GeoDataFrame]:
                The nx graph, nodes and edges gdfs.
        """
        # Get the directory name
        dirname = os.path.dirname(__file__)
        gpkg_path = os.path.join(dirname, f'cities/{city}/streets.gpkg')
        
        # Load the edges and nodes
        nodes = gpd.read_file(gpkg_path, layer='nodes')
        edges = gpd.read_file(gpkg_path, layer='edges')

        # Set the indices correctly
        edges.set_index(['u', 'v', 'key'], inplace=True)
        nodes.set_index(['osmid'], inplace=True)

        # Ensure that the geometries have the correct values
        nodes['x'] = nodes['geometry'].apply(lambda x: x.x)
        nodes['y'] = nodes['geometry'].apply(lambda x: x.y)

        # Create the graph
        G = ox.graph_from_gdfs(nodes, edges)
        
        return G, nodes, edges
    
    def get_scenario_list(self, city:str) -> list:
        """Returns a list of scenarios associated with this city

        Returns:
            list: List of paths to scenarios associated with this city.
        """
        dirname = os.path.dirname(__file__)
        scendirname = os.path.join(dirname, f'cities/{city}/scenarios')
        return [os.path.join(scendirname, x) for x in os.listdir(scendirname) 
                if '.txt' in x]
        
    def get_scenario_names(self, city:str) -> list:
        """Returns the names of the files in the scenario folder.
        """
        if city is None:
            city = self.name
        dirname = os.path.dirname(__file__)
        scendirname = os.path.join(dirname, f'cities/{city}/scenarios')
        return [x.replace('.txt','') for x in os.listdir(scendirname) 
                if '.txt' in x]
        
def parse_scenario(scenario_path:os.PathLike, num_ac:int = 0) -> dict:
    """Parses a scenario into a dictionary with each entry representing a flight.
    Each entry is of shape acid:[dep time, origin, destination].

    Args:
        scenario_path (os.PathLike): Path to scenario
        num_ac (int): Number of aircraft to extract from scenario

    Returns:
        dict: Scenario dict.
    """
    # Open the scenario
    with open(scenario_path) as f:
        scen_lines = f.readlines()
        scen_name = os.path.basename(f.name).replace('.txt','')
        
    if num_ac == 0 or num_ac > len(scen_lines):
        num_ac = len(scen_lines)
    else:
        scen_name += f'_{num_ac}'

    # Create the dict
    scenario = {}
    for line in scen_lines[:num_ac]:
        s = line.split(';')
        dep_time = int(s[2].split(':')[0]) * 3600 + int(s[2].split(':')[1]) * 60 +\
            int(s[2].split(':')[2])
        origin = int(s[3])
        destination = int(s[4])
        scenario[s[0]] = [dep_time, origin, destination]
    
    return scen_name, scenario